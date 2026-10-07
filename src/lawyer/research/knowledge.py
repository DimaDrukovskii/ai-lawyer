"""База знаний: результат research, версионируется по run_id и лежит в git.

Принцип: в «доверенное» попадает только то, что (1) подтверждено независимым верификатором
И (2) опирается хотя бы на один источник уровня primary/official_text/marketplace.
Оценка мнения агента (kind=opinion) в доверенное не попадает никогда. Это правило —
код, а не просьба к модели.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..schemas import (
    Claim,
    ClaimCheck,
    ClaimKind,
    Gap,
    SourceRef,
    Tier,
    Verdict,
    VerifiedZone,
    ZoneFinding,
)
from ..tools.domains import normalize_url
from .zones import ResearchContext, Zone

TRUSTED_TIERS = frozenset({Tier.PRIMARY, Tier.OFFICIAL_TEXT, Tier.MARKETPLACE})


class Status(StrEnum):
    TRUSTED = "trusted"
    REJECTED = "rejected"  # wrong / outdated / exaggerated
    UNVERIFIED = "unverified"


class KnowledgeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    zone_id: str
    claim_id: str
    text: str
    kind: ClaimKind
    condition: str = ""
    numbers: list[str] = []
    sources: list[SourceRef] = []
    status: Status
    verdict: Verdict
    correction: str = ""
    as_of: str


def classify_claim(
    claim: Claim, check: ClaimCheck | None, fetched_urls: Iterable[str] = ()
) -> Status:
    """Доверие = вердикт confirmed + первоисточник, который верификатор РЕАЛЬНО открыл.

    fetched_urls берётся из журнала инструментов верификатора (код), а не из его ответа:
    модель не может «подтвердить» источник, к которому не ходила.
    """
    if check is None or check.verdict is Verdict.UNVERIFIABLE:
        return Status.UNVERIFIED
    if check.verdict is not Verdict.CONFIRMED:
        return Status.REJECTED
    if claim.kind is ClaimKind.OPINION:
        return Status.UNVERIFIED
    opened = {normalize_url(u) for u in fetched_urls}
    if not any(s.tier in TRUSTED_TIERS and normalize_url(s.url) in opened for s in claim.sources):
        return Status.UNVERIFIED  # подтверждено, но без открытого первоисточника — не доверяем
    return Status.TRUSTED


def has_trusted(finding: ZoneFinding, verified: VerifiedZone) -> bool:
    checks = {c.claim_id: c for c in verified.checks}
    return any(
        classify_claim(c, checks.get(c.id), verified.fetched_urls) is Status.TRUSTED
        for c in finding.claims
    )


def build_records(
    findings: dict[str, ZoneFinding], verified: dict[str, VerifiedZone], as_of: date
) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    for zone_id, finding in findings.items():
        checks = {c.claim_id: c for c in verified[zone_id].checks}
        fetched = verified[zone_id].fetched_urls
        for claim in finding.claims:
            check = checks.get(claim.id)
            records.append(
                KnowledgeRecord(
                    zone_id=zone_id,
                    claim_id=claim.id,
                    text=claim.text,
                    kind=claim.kind,
                    condition=claim.condition,
                    numbers=claim.numbers,
                    sources=claim.sources,
                    status=classify_claim(claim, check, fetched),
                    verdict=check.verdict if check else Verdict.UNVERIFIABLE,
                    correction=check.correction if check else "",
                    as_of=as_of.isoformat(),
                )
            )
    return records


# ---------------------------------------------------------------- рендер


def _src(s: SourceRef) -> str:
    label = s.title or s.url
    date_part = f", {s.doc_date}" if s.doc_date else ""
    return f"[{label}]({s.url}) ({s.tier.value}{date_part})"


def _bullet(r: KnowledgeRecord) -> str:
    cond = f"\n  - *Применимо, если:* {r.condition}" if r.condition else ""
    nums = f"\n  - *Числа:* {'; '.join(r.numbers)}" if r.numbers else ""
    srcs = "\n".join(f"  - {_src(s)}" for s in r.sources) or "  - (источников нет)"
    fix = f"\n  - *Поправка верификатора:* {r.correction}" if r.correction else ""
    return f"- **[{r.claim_id}]** {r.text}{cond}{nums}{fix}\n{srcs}"


def render_zone(
    zone: Zone,
    finding: ZoneFinding,
    records: list[KnowledgeRecord],
    ctx: ResearchContext,
    run_id: str,
) -> str:
    def section(title: str, status: Status) -> str:
        items = [r for r in records if r.status is status]
        body = "\n".join(_bullet(r) for r in items) if items else "_нет_"
        return f"## {title}\n{body}"

    unverified_notes = "\n".join(f"- {u}" for u in finding.unverified) or "_нет_"
    return "\n\n".join(
        [
            f"# {zone.title}",
            f"> Актуально на {ctx.as_of.isoformat()} · налоговый год {ctx.tax_year} · "
            f"прогон `{run_id}` · уверенность исследователя: {finding.confidence}",
            f"**Вопрос зоны:** {zone.question}",
            f"## Сводка исследователя\n{finding.summary}",
            section("Подтверждено (доверенное)", Status.TRUSTED),
            section("Отклонено верификатором", Status.REJECTED),
            section("Не подтверждено / мнение агента", Status.UNVERIFIED),
            f"## Что исследователь не смог подтвердить\n{unverified_notes}",
        ]
    )


def write_knowledge(
    knowledge_root: Path,
    *,
    run_id: str,
    ctx: ResearchContext,
    zones: list[Zone],
    findings: dict[str, ZoneFinding],
    verified: dict[str, VerifiedZone],
    open_gaps: list[Gap],
    failed: dict[str, str],
    promote: bool = True,
) -> Path:
    """promote=False: прогон сохраняется, но LATEST не трогаем (частичный или с упавшими зонами)."""
    run_dir = knowledge_root / run_id
    (run_dir / "zones").mkdir(parents=True, exist_ok=True)
    records = build_records(findings, verified, ctx.as_of)
    by_zone: dict[str, list[KnowledgeRecord]] = {}
    for r in records:
        by_zone.setdefault(r.zone_id, []).append(r)

    zone_by_id = {z.id: z for z in zones}
    for zone_id, finding in findings.items():
        zone = zone_by_id.get(zone_id) or Zone(
            id=zone_id, title=zone_id, question=finding.summary[:200]
        )
        (run_dir / "zones" / f"{zone_id}.md").write_text(
            render_zone(zone, finding, by_zone.get(zone_id, []), ctx, run_id), encoding="utf-8"
        )

    (run_dir / "claims.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in records], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (run_dir / "gaps.json").write_text(
        json.dumps(
            {"open": [g.model_dump(mode="json") for g in open_gaps], "failed_zones": failed},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    counts = {s: sum(r.status is s for r in records) for s in Status}
    index = [
        f"# База знаний · прогон `{run_id}`",
        f"Актуально на **{ctx.as_of.isoformat()}**, налоговый год **{ctx.tax_year}**. "
        f"Аудитория: {ctx.audience}.",
        f"Утверждений: доверенных {counts[Status.TRUSTED]}, отклонено {counts[Status.REJECTED]}, "
        f"не подтверждено {counts[Status.UNVERIFIED]}. Открытых пробелов: {len(open_gaps)}. "
        f"Упавших зон: {len(failed)}.",
        "## Зоны",
        *(f"- [{zone_by_id[z].title if z in zone_by_id else z}](zones/{z}.md)" for z in findings),
    ]
    if failed:
        index += ["## Не удались", *(f"- `{z}`: {err}" for z, err in failed.items())]
    if not promote:
        index.insert(
            2, "> **Не назначен актуальным (LATEST не изменён):** прогон частичный или с ошибками."
        )
    (run_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    if promote:
        tmp = knowledge_root / "LATEST.tmp"
        tmp.write_text(run_id + "\n", encoding="utf-8")
        tmp.replace(knowledge_root / "LATEST")  # атомарно: review не увидит полузаписанный файл
    return run_dir


# ---------------------------------------------------------------- чтение (для review)


def latest_run(knowledge_root: Path) -> str | None:
    marker = knowledge_root / "LATEST"
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else None


def load_trusted(knowledge_root: Path, run_id: str | None = None) -> list[KnowledgeRecord]:
    run = run_id or latest_run(knowledge_root)
    if run is None:
        return []
    path = knowledge_root / run / "claims.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in map(KnowledgeRecord.model_validate, raw) if r.status is Status.TRUSTED]


def _prompt_line(r: KnowledgeRecord) -> str:
    cond = f" [если: {r.condition}]" if r.condition else ""
    return f"[{r.claim_id}] {r.text}{cond} (актуально на {r.as_of})"


def select_for_prompt(
    records: Iterable[KnowledgeRecord], max_chars: int = 24_000
) -> tuple[list[KnowledgeRecord], bool]:
    """Что поместится в промпт судьи и была ли обрезка. Ссылаться можно только на показанное."""
    shown: list[KnowledgeRecord] = []
    used = 0
    for r in records:
        used += len(_prompt_line(r)) + 1
        if used > max_chars:
            return shown, True
        shown.append(r)
    return shown, False


def format_for_prompt(records: Iterable[KnowledgeRecord], max_chars: int = 24_000) -> str:
    shown, truncated = select_for_prompt(records, max_chars)
    lines = [_prompt_line(r) for r in shown]
    if truncated:
        lines.append("[…база знаний обрезана по лимиту…]")
    return "\n".join(lines) or "(доверенных утверждений в базе знаний нет)"
