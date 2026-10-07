from __future__ import annotations

from ..agent import run_agent
from ..deps import Deps
from ..schemas import ClaimCheck, Verdict, VerifiedZone, ZoneFinding
from ..structured import parse_structured
from .zones import ResearchContext


def _complete(verified: VerifiedZone, finding: ZoneFinding) -> VerifiedZone:
    """Каждое утверждение обязано получить вердикт; пропущенные считаем непроверяемыми."""
    known = {c.claim_id for c in verified.checks}
    known_in_finding = {c.id for c in finding.claims}
    checks = [c for c in verified.checks if c.claim_id in known_in_finding]
    for claim in finding.claims:
        if claim.id not in known:
            checks.append(
                ClaimCheck(
                    claim_id=claim.id,
                    verdict=Verdict.UNVERIFIABLE,
                    note="верификатор не вынес вердикт",
                )
            )
    return verified.model_copy(update={"zone_id": finding.zone_id, "checks": checks})


async def verify_zone(deps: Deps, finding: ZoneFinding, ctx: ResearchContext) -> VerifiedZone:
    """Второй агент с установкой «исходно не верю»: открывает каждую ссылку сам."""
    s = deps.settings
    system = deps.prompt(
        "verifier", tax_year=ctx.tax_year, as_of=ctx.as_of.isoformat(), audience=ctx.audience
    )
    user = f"zone_id: {finding.zone_id}\n<finding>\n{finding.model_dump_json(indent=1)}\n</finding>"
    result = await run_agent(
        deps.provider,
        model=s.model_strong,  # верификатор сильнее исследователя: он ловит его ошибки
        system=system,
        user=user,
        registry=deps.registry,
        max_steps=s.max_tool_steps + 4,
        max_tokens=4000,
    )
    verified = await parse_structured(
        deps.provider, model=s.model_fast, model_cls=VerifiedZone, text=result.text
    )
    return _complete(verified, finding)
