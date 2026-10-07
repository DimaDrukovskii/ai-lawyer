from __future__ import annotations

from ..agent import ToolLogEntry, run_agent
from ..deps import Deps
from ..schemas import ClaimCheck, Verdict, VerifiedZone, ZoneFinding
from ..structured import parse_structured
from ..tools.domains import normalize_url
from .zones import ResearchContext

# Чем меньше число, тем строже вердикт. При дублях побеждает самый строгий:
# порядок ответа модели не должен решать, доверено утверждение или отклонено.
_STRICTNESS = {
    Verdict.WRONG: 0,
    Verdict.OUTDATED: 0,
    Verdict.EXAGGERATED: 0,
    Verdict.UNVERIFIABLE: 1,
    Verdict.CONFIRMED: 2,
}


def _complete(verified: VerifiedZone, finding: ZoneFinding) -> VerifiedZone:
    """Ровно один вердикт на каждое утверждение; пропущенные считаем непроверяемыми."""
    valid_ids = {c.id for c in finding.claims}
    strictest: dict[str, ClaimCheck] = {}
    for check in verified.checks:
        if check.claim_id not in valid_ids:
            continue
        current = strictest.get(check.claim_id)
        if current is None or _STRICTNESS[check.verdict] < _STRICTNESS[current.verdict]:
            strictest[check.claim_id] = check
    checks = [
        strictest.get(c.id)
        or ClaimCheck(
            claim_id=c.id, verdict=Verdict.UNVERIFIABLE, note="верификатор не вынес вердикт"
        )
        for c in finding.claims
    ]
    return verified.model_copy(update={"zone_id": finding.zone_id, "checks": checks})


def opened_urls(tool_log: tuple[ToolLogEntry, ...]) -> list[str]:
    """Страницы, которые верификатор действительно загрузил: по журналу инструментов."""
    seen: dict[str, str] = {}
    for entry in tool_log:
        url = str(entry.arguments.get("url", ""))
        if entry.name == "fetch_url" and entry.ok and url:
            seen.setdefault(normalize_url(url), url)
    return list(seen.values())


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
        max_tokens=s.max_output_tokens,
    )
    verified = await parse_structured(
        deps.provider, model=s.model_fast, model_cls=VerifiedZone, text=result.text
    )
    # fetched_urls задаёт код, а не модель: что бы ни написал верификатор в ответе, затираем
    return _complete(verified, finding).model_copy(
        update={"fetched_urls": opened_urls(result.tool_log)}
    )
