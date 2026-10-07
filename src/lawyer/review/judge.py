from __future__ import annotations

from pydantic import BaseModel

from ..agent import run_agent
from ..deps import Deps
from ..research.knowledge import KnowledgeRecord, format_for_prompt
from ..schemas import ExtractedDocs, Finding
from ..structured import parse_structured
from .docs_io import Case, pack

UNCONFIRMED_PREFIX = "[НЕ ПОДТВЕРЖДЕНО БАЗОЙ] "


class _JudgeOut(BaseModel):
    findings: list[Finding] = []


def sanitize(findings: list[Finding], known_claim_ids: set[str]) -> list[Finding]:
    """Находка о праве без ссылки на доверенное утверждение не выдаётся за установленный факт.

    Выдуманные моделью claim_id отбрасываем; если опоры не осталось — помечаем явно.
    """
    out: list[Finding] = []
    for f in findings:
        valid = [c for c in f.claim_ids if c in known_claim_ids]
        grounded = bool(valid)
        update: dict[str, object] = {"claim_ids": valid, "origin": "judge"}
        if not grounded or len(valid) != len(f.claim_ids):
            update["needs_human"] = True
        if not grounded and not f.title.startswith(UNCONFIRMED_PREFIX):
            update["title"] = UNCONFIRMED_PREFIX + f.title
        out.append(f.model_copy(update=update))
    return out


async def judge(
    deps: Deps,
    case: Case,
    extracted: ExtractedDocs,
    check_findings: list[Finding],
    knowledge: list[KnowledgeRecord],
    checklist: str,
) -> list[Finding]:
    s = deps.settings
    user = "\n".join(
        [
            pack(case),
            f"<extracted>\n{extracted.model_dump_json(indent=1)}\n</extracted>",
            "<checks>\n"
            + "\n".join(f"- [{f.severity.value}] {f.title}: {f.detail}" for f in check_findings)
            + "\n</checks>",
            f"<checklist>\n{checklist}\n</checklist>",
            f"<knowledge>\n{format_for_prompt(knowledge)}\n</knowledge>",
        ]
    )
    result = await run_agent(
        deps.provider,
        model=s.model_strong,
        system=deps.prompt("judge"),
        user=user,
        registry=None,
        max_tokens=5000,
    )
    out = await parse_structured(
        deps.provider, model=s.model_fast, model_cls=_JudgeOut, text=result.text
    )
    return sanitize(out.findings, {r.claim_id for r in knowledge})
