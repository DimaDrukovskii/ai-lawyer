"""Мультиагентный research: разведка → состязательная проверка → критики полноты → дозакрытие.

Слои (по методике урока):
  1. N параллельных исследователей, по одному на зону (first-source only)
  2. На каждую зону второй агент «исходно не верю» перепроверяет каждую ссылку
  3. Три критика полноты с разными оптиками (календарь / форензик / дизайн проверок)
  4. Критические и высокие пробелы уходят обратно в research отдельными агентами
Каждый этап кэшируется в out/research/<run_id>/ — упавший прогон можно продолжить.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..deps import Deps
from ..schemas import CriticReport, Gap, Severity, VerifiedZone, ZoneFinding
from .critics import actionable_gaps, critique_all, digest
from .knowledge import has_trusted, write_knowledge
from .researcher import research_zone
from .verifier import verify_zone
from .zones import ResearchContext, Zone, safe_id

M = TypeVar("M", bound=BaseModel)
Event = Callable[[str], None]

MAX_GAPS_PER_ROUND = 6


@dataclass(frozen=True)
class RunResult:
    run_id: str
    knowledge_dir: Path
    findings: dict[str, ZoneFinding]
    verified: dict[str, VerifiedZone]
    critics: list[CriticReport]
    open_gaps: list[Gap]
    failed: dict[str, str] = field(default_factory=dict)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # оборванная запись не оставит битый кэш


def _cached(path: Path, model_cls: type[M], enabled: bool) -> M | None:
    if not enabled or not path.exists():
        return None
    try:
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError, OSError):
        return None  # битый кэш равен отсутствию кэша: пересчитаем зону


def _save(path: Path, obj: BaseModel) -> None:
    _write_atomic(path, obj.model_dump_json(indent=1))


def _adopt_run_meta(
    cache_dir: Path, ctx: ResearchContext, resume: bool, emit: Event
) -> ResearchContext:
    """Возобновляя прогон, берём налоговый год и дату из него, а не из текущих аргументов.

    Иначе кэш прошлого года окажется подписан «актуально на сегодня» с новым годом.
    """
    meta_path = cache_dir / "run.json"
    if resume and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            saved = replace(
                ctx, tax_year=int(meta["tax_year"]), as_of=date.fromisoformat(meta["as_of"])
            )
        except (ValueError, KeyError, TypeError, OSError):
            saved = None
        if saved is not None:
            if saved != ctx:
                emit(
                    f"[resume] беру из прогона: налоговый год {saved.tax_year}, дата {saved.as_of}"
                )
            return saved
    _write_atomic(meta_path, json.dumps({"tax_year": ctx.tax_year, "as_of": ctx.as_of.isoformat()}))
    return ctx


async def _process_zone(
    deps: Deps, zone: Zone, ctx: ResearchContext, cache_dir: Path, resume: bool, emit: Event
) -> tuple[ZoneFinding, VerifiedZone]:
    finding = _cached(cache_dir / f"{zone.id}.finding.json", ZoneFinding, resume)
    regenerated = finding is None
    if finding is None:
        emit(f"[research] {zone.id}")
        finding = await research_zone(deps, zone, ctx)
        _save(cache_dir / f"{zone.id}.finding.json", finding)

    # вердикты относятся к конкретным утверждениям: новый finding делает старый verified недействительным
    verified = (
        None
        if regenerated
        else _cached(cache_dir / f"{zone.id}.verified.json", VerifiedZone, resume)
    )
    if verified is None:
        emit(f"[verify]   {zone.id} ({len(finding.claims)} утверждений)")
        verified = await verify_zone(deps, finding, ctx)
        _save(cache_dir / f"{zone.id}.verified.json", verified)
    return finding, verified


async def _process_many(
    deps: Deps, zones: list[Zone], ctx: ResearchContext, cache_dir: Path, resume: bool, emit: Event
) -> tuple[dict[str, tuple[ZoneFinding, VerifiedZone]], dict[str, str]]:
    async def one(zone: Zone) -> tuple[str, tuple[ZoneFinding, VerifiedZone] | str]:
        try:
            return zone.id, await _process_zone(deps, zone, ctx, cache_dir, resume, emit)
        except Exception as exc:  # граница зоны: любая её поломка не должна стоить всего прогона
            emit(f"[fail]     {zone.id}: {type(exc).__name__}")
            return zone.id, f"{type(exc).__name__}: {exc}"[:300]

    done: dict[str, tuple[ZoneFinding, VerifiedZone]] = {}
    failed: dict[str, str] = {}
    for zone_id, outcome in await asyncio.gather(*(one(z) for z in zones)):
        if isinstance(outcome, str):
            failed[zone_id] = outcome
        else:
            done[zone_id] = outcome
    return done, failed


def _gap_to_zone(gap: Gap, round_no: int) -> Zone:
    return Zone(
        id=safe_id(f"gap-r{round_no}-{gap.id}"),
        title=f"Дозакрытие: {gap.angle}",
        question=gap.question,
        must_cover=[],
        scope="primary",
    )


def _dedup_gaps(gaps: list[Gap]) -> list[Gap]:
    seen: set[str] = set()
    out: list[Gap] = []
    for g in gaps:
        key = g.question.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


async def run_research(
    deps: Deps,
    zones: list[Zone],
    ctx: ResearchContext,
    *,
    run_id: str,
    out_root: Path,
    knowledge_root: Path,
    rounds: int = 1,
    resume: bool = True,
    promote: bool = True,
    emit: Event = lambda _msg: None,
) -> RunResult:
    cache_dir = out_root / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    ctx = _adopt_run_meta(cache_dir, ctx, resume, emit)
    all_zones = list(zones)

    emit(f"[start] {len(zones)} зон параллельно, прогон {run_id}")
    done, failed = await _process_many(deps, zones, ctx, cache_dir, resume, emit)

    critics: list[CriticReport] = []
    open_gaps: list[Gap] = []
    for round_no in range(1, rounds + 1):
        if not done:
            break
        emit(f"[critics] раунд {round_no}")
        critics = await critique_all(
            deps,
            digest(
                all_zones,
                {z: f for z, (f, _) in done.items()},
                {z: v for z, (_, v) in done.items()},
            ),
            ctx,
        )
        actionable = actionable_gaps(critics)
        if not actionable:
            break
        gaps, overflow = actionable[:MAX_GAPS_PER_ROUND], actionable[MAX_GAPS_PER_ROUND:]
        emit(f"[gaps] {len(gaps)} пробелов уходят в research, без очереди осталось {len(overflow)}")
        gap_zones = [_gap_to_zone(g, round_no) for g in gaps]
        all_zones += gap_zones
        more, more_failed = await _process_many(deps, gap_zones, ctx, cache_dir, resume, emit)
        done = {**done, **more}
        failed = {**failed, **more_failed}
        # Пробел закрыт, только если дозакрытие дало хотя бы одно доверенное утверждение:
        # «агент отработал без ошибки» не значит «вопрос выяснен».
        closed = {zid for zid, (f, v) in more.items() if has_trusted(f, v)}
        open_gaps += overflow + [
            g for g, z in zip(gaps, gap_zones, strict=True) if z.id not in closed
        ]

    # неактуальные (low/medium) пробелы не ресёрчим, но сохраняем как открытые
    open_gaps += [
        g for r in critics for g in r.gaps if g.severity in {Severity.MEDIUM, Severity.LOW}
    ]

    findings = {z: f for z, (f, _) in done.items()}
    verified = {z: v for z, (_, v) in done.items()}
    open_gaps = _dedup_gaps(open_gaps)
    knowledge_dir = write_knowledge(
        knowledge_root,
        run_id=run_id,
        ctx=ctx,
        zones=all_zones,
        findings=findings,
        verified=verified,
        open_gaps=open_gaps,
        failed=failed,
        promote=promote and not failed and bool(findings),
    )
    emit(f"[done] база знаний: {knowledge_dir}")
    return RunResult(run_id, knowledge_dir, findings, verified, critics, open_gaps, failed)
