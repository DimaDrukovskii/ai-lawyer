from __future__ import annotations

import asyncio

from ..agent import run_agent
from ..deps import Deps
from ..schemas import CriticReport, Gap, Severity, VerifiedZone, ZoneFinding
from ..structured import parse_structured
from .zones import ResearchContext, Zone

# Три разные оптики поверх всей картины (слой 3 методики)
LENSES = ("calendar", "forensic", "checks")


def digest(
    zones: list[Zone],
    findings: dict[str, ZoneFinding],
    verified: dict[str, VerifiedZone],
    *,
    max_claim_chars: int = 240,
) -> str:
    """Компактная выжимка для критиков: зона → сводка → утверждения с вердиктами."""
    parts: list[str] = []
    for zone in zones:
        finding = findings.get(zone.id)
        if finding is None:
            parts.append(f"## {zone.id} — {zone.title}\n(исследование не удалось)")
            continue
        verdicts = {c.claim_id: c.verdict.value for c in verified[zone.id].checks}
        lines = [
            f"- [{c.id}] ({verdicts.get(c.id, '?')}, {c.kind.value}) {c.text[:max_claim_chars]}"
            for c in finding.claims
        ]
        parts.append(f"## {zone.id} — {zone.title}\n{finding.summary}\n" + "\n".join(lines))
    return "\n\n".join(parts)


async def critique(deps: Deps, lens: str, digest_text: str, ctx: ResearchContext) -> CriticReport:
    s = deps.settings
    system = deps.prompt(
        f"critic_{lens}", tax_year=ctx.tax_year, as_of=ctx.as_of.isoformat(), audience=ctx.audience
    )
    result = await run_agent(
        deps.provider,
        model=s.model_strong,
        system=system,
        user=f"lens: {lens}\n<digest>\n{digest_text}\n</digest>",
        registry=None,  # критики работают по выжимке, без похода в сеть
        max_tokens=3000,
    )
    report = await parse_structured(
        deps.provider, model=s.model_fast, model_cls=CriticReport, text=result.text
    )
    return report.model_copy(update={"lens": lens})


async def critique_all(deps: Deps, digest_text: str, ctx: ResearchContext) -> list[CriticReport]:
    return list(await asyncio.gather(*(critique(deps, lens, digest_text, ctx) for lens in LENSES)))


def actionable_gaps(reports: list[CriticReport], limit: int = 6) -> list[Gap]:
    """Критические и высокие пробелы без дублей, не больше limit штук."""
    seen: set[str] = set()
    out: list[Gap] = []
    for report in reports:
        for gap in report.gaps:
            key = gap.question.strip().lower()
            if gap.severity in {Severity.CRITICAL, Severity.HIGH} and key not in seen:
                seen.add(key)
                out.append(gap.model_copy(update={"id": f"{report.lens}-{gap.id}"}))
    out.sort(key=lambda g: g.severity is not Severity.CRITICAL)
    return out[:limit]
