from __future__ import annotations

from datetime import date

from ..agent import run_agent
from ..deps import Deps
from ..schemas import Finding, NonEmptyModel
from ..structured import parse_structured
from .checks import SEVERITY_ORDER

DISCLAIMER = (
    "> **Это автоматический second opinion, а не заключение налогового консультанта и не "
    "юридическая услуга.** Он ищет риски и формулирует вопросы к специалисту. Решения о подаче "
    "документов принимай вместе с бухгалтером или налоговым консультантом."
)


class ReportParts(NonEmptyModel):
    summary: str = ""
    questions_for_accountant: list[str] = []
    next_steps: list[str] = []


async def build_parts(deps: Deps, profile_json: str, findings: list[Finding]) -> ReportParts:
    s = deps.settings
    listing = (
        "\n".join(
            f"- [{f.severity.value}] {f.title}: {f.detail} {f.price_of_error}".strip()
            for f in findings
        )
        or "- (находок нет)"
    )
    result = await run_agent(
        deps.provider,
        model=s.model_strong,
        system=deps.prompt("report"),
        user=f"<profile>\n{profile_json}\n</profile>\n<findings>\n{listing}\n</findings>",
        registry=None,
        max_tokens=2000,
    )
    return await parse_structured(
        deps.provider, model=s.model_fast, model_cls=ReportParts, text=result.text
    )


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_report(
    *,
    case_name: str,
    findings: list[Finding],
    parts: ReportParts,
    open_questions: list[str],
    unreadable: list[str],
    unverified_rules: list[str],
    knowledge_run: str | None,
    today: date,
) -> str:
    ordered = sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.id))
    table = ["| № | Находка | Серьёзность | Цена ошибки | Источник |", "|---|---|---|---|---|"]
    for i, f in enumerate(ordered, 1):
        src = (
            ", ".join(f.claim_ids)
            if f.claim_ids
            else ("расчёт по документам" if f.origin == "check" else "—")
        )
        human = " ⚠ проверить у специалиста" if f.needs_human else ""
        table.append(
            f"| {i} | {_cell(f.title)}{human} | {f.severity.value} | {_cell(f.price_of_error or '—')} | {_cell(src)} |"
        )
    details = [f"### {i}. {f.title}\n{f.detail}" for i, f in enumerate(ordered, 1)]

    def bullets(items: list[str], empty: str = "_нет_") -> str:
        return "\n".join(f"- {x}" for x in items) if items else empty

    basis = [
        f"- База знаний: прогон `{knowledge_run}`"
        if knowledge_run
        else "- База знаний: **не найдена**, выводы о праве не подкреплены",
        f"- Правила проверок без подтверждения research-прогоном: {', '.join(unverified_rules) or 'нет'}",
    ]
    return (
        "\n\n".join(
            [
                f"# Second opinion: {case_name}",
                f"_Дата отчёта: {today.isoformat()}_",
                DISCLAIMER,
                f"## Резюме\n{parts.summary or '_нет_'}",
                "## Находки\n" + ("\n".join(table) if ordered else "_Находок нет._"),
                "## Детали\n" + ("\n\n".join(details) if details else "_нет_"),
                f"## Вопросы бухгалтеру\n{bullets(parts.questions_for_accountant)}",
                f"## Что делать\n{bullets(parts.next_steps)}",
                f"## Чего не хватало для полного вывода\n{bullets(open_questions)}",
                f"## Не удалось прочитать\n{bullets(unreadable)}",
                "## Основание\n" + "\n".join(basis),
            ]
        )
        + "\n"
    )
