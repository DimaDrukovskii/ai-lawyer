from __future__ import annotations

from ..agent import run_agent
from ..deps import Deps
from ..schemas import Claim, SourceRef, ZoneFinding
from ..structured import parse_structured
from ..tools.domains import classify
from .zones import ResearchContext, Zone


def _normalize(finding: ZoneFinding, zone_id: str, max_claims: int) -> ZoneFinding:
    """Приводим вывод модели к инвариантам: id зоны, уникальные id утверждений, tier по URL.

    Tier нельзя доверять модели: он считается из домена источника.
    """
    claims: list[Claim] = []
    for i, claim in enumerate(finding.claims[:max_claims], 1):
        sources = [s.model_copy(update={"tier": classify(s.url)}) for s in claim.sources]
        claims.append(
            claim.model_copy(update={"id": f"{zone_id}.c{i}", "sources": _dedup(sources)})
        )
    return finding.model_copy(update={"zone_id": zone_id, "claims": claims})


def _dedup(sources: list[SourceRef]) -> list[SourceRef]:
    seen: set[str] = set()
    out: list[SourceRef] = []
    for s in sources:
        if s.url not in seen:
            seen.add(s.url)
            out.append(s)
    return out


def _user_message(zone: Zone) -> str:
    must = "\n".join(f"- {m}" for m in zone.must_cover) or "- (на твоё усмотрение, но полно)"
    return (
        f"zone_id: {zone.id}\n"
        f"Зона: {zone.title}\n"
        f"Главный вопрос: {zone.question}\n"
        f"Обязательно покрыть:\n{must}\n"
        f"Рекомендуемый scope поиска: {zone.scope}"
    )


async def research_zone(deps: Deps, zone: Zone, ctx: ResearchContext) -> ZoneFinding:
    s = deps.settings
    system = deps.prompt(
        "researcher",
        tax_year=ctx.tax_year,
        as_of=ctx.as_of.isoformat(),
        audience=ctx.audience,
        max_claims=s.max_claims_per_zone,
    )
    result = await run_agent(
        deps.provider,
        model=s.model_fast,
        system=system,
        user=_user_message(zone),
        registry=deps.registry,
        max_steps=s.max_tool_steps,
        max_tokens=4000,
    )
    finding = await parse_structured(
        deps.provider, model=s.model_fast, model_cls=ZoneFinding, text=result.text
    )
    return _normalize(finding, zone.id, s.max_claims_per_zone)
