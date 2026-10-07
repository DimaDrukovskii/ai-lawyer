from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from ..agent import run_agent
from ..deps import Deps
from ..structured import parse_structured
from .docs_io import Case, pack

MAX_QUESTIONS = 5


class _IntakeOut(BaseModel):
    missing: list[str] = []


async def find_missing(deps: Deps, case: Case) -> list[str]:
    """Вопросы, без ответа на которые вывод будет неверным (не больше MAX_QUESTIONS)."""
    s = deps.settings
    result = await run_agent(
        deps.provider,
        model=s.model_strong,
        system=deps.prompt("intake"),
        user=pack(case),
        registry=None,
        max_tokens=1500,
    )
    out = await parse_structured(
        deps.provider, model=s.model_fast, model_cls=_IntakeOut, text=result.text
    )
    return [q.strip() for q in out.missing if q.strip()][:MAX_QUESTIONS]


def collect_answers(questions: list[str], ask: Callable[[str], str]) -> str:
    """Задаёт вопросы по очереди. Возвращает блок для дописывания в context.local.md."""
    lines = [f"Q: {q}\nA: {ask(q + ' > ').strip()}" for q in questions]
    return "\n\n## Уточнения\n" + "\n".join(lines) + "\n"
