from __future__ import annotations

from ..agent import run_agent
from ..deps import Deps
from ..schemas import ExtractedDocs
from ..structured import parse_structured
from .docs_io import Case, pack


async def extract_figures(deps: Deps, case: Case) -> ExtractedDocs:
    """LLM только переписывает числа из документов; считает и сверяет потом код (checks.py)."""
    s = deps.settings
    result = await run_agent(
        deps.provider,
        model=s.model_strong,
        system=deps.prompt("extract"),
        user=pack(case),
        registry=None,
        max_tokens=3500,
    )
    extracted = await parse_structured(
        deps.provider, model=s.model_fast, model_cls=ExtractedDocs, text=result.text
    )
    return extracted.model_copy(update={"unreadable": [*extracted.unreadable, *case.unreadable]})
