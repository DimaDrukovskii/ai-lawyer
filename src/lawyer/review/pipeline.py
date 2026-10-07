"""Проверка документов клиента: приёмка → извлечение → детерминированные сверки → судья → отчёт."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..deps import Deps
from ..research.knowledge import latest_run, load_trusted
from ..schemas import ExtractedDocs, Finding
from .checks import SEVERITY_ORDER, run_checks
from .docs_io import Case, load_case
from .extract import extract_figures
from .grounding import grounding_findings
from .intake import collect_answers, find_missing
from .judge import judge
from .report import build_parts, render_report
from .rules import RuleBook


@dataclass(frozen=True)
class ReviewPaths:
    knowledge_root: Path
    rules_file: Path
    checklist_file: Path
    out_dir: Path


@dataclass(frozen=True)
class ReviewResult:
    report_md: str
    findings: list[Finding]
    extracted: ExtractedDocs
    open_questions: list[str]
    report_path: Path


async def run_review(
    deps: Deps,
    case_dir: Path,
    paths: ReviewPaths,
    *,
    ask: Callable[[str], str] | None = None,
    knowledge_run: str | None = None,
    holidays: frozenset[date] = frozenset(),
    today: date | None = None,
) -> ReviewResult:
    case: Case = load_case(case_dir)

    # 0. приёмка: вопросы, без которых вывод неверен
    open_questions = await find_missing(deps, case)
    if open_questions and ask is not None:
        case.local_context_path.write_text(
            (
                case.local_context_path.read_text(encoding="utf-8")
                if case.local_context_path.exists()
                else ""
            )
            + collect_answers(open_questions, ask),
            encoding="utf-8",
        )
        open_questions = []
        case = load_case(case_dir)  # контекст дополнен ответами

    # 1. числа из документов (LLM только переписывает)
    extracted = await extract_figures(deps, case)

    # 2. детерминированные проверки (считает код)
    rules = RuleBook.from_yaml(paths.rules_file)
    check_findings = [
        *run_checks(case.profile, extracted, rules, holidays),
        *grounding_findings(extracted, case.docs),
    ]

    # 3. судья: то, что код не покрывает, строго на доверенной базе знаний
    knowledge = load_trusted(paths.knowledge_root, knowledge_run)
    judged = await judge(
        deps,
        case,
        extracted,
        check_findings,
        knowledge,
        paths.checklist_file.read_text(encoding="utf-8"),
    )
    findings = sorted([*check_findings, *judged], key=lambda f: (SEVERITY_ORDER[f.severity], f.id))

    # 4. отчёт: таблицу собирает код, резюме и вопросы пишет модель
    parts = await build_parts(deps, case.profile.model_dump_json(), findings)
    report = render_report(
        case_name=case.name,
        findings=findings,
        parts=parts,
        open_questions=open_questions,
        unreadable=list(extracted.unreadable),
        unverified_rules=rules.unverified_keys(),
        knowledge_run=knowledge_run or latest_run(paths.knowledge_root),
        today=today or date.today(),
    )
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = paths.out_dir / f"{case.name}_report.md"
    report_path.write_text(report, encoding="utf-8")
    (paths.out_dir / f"{case.name}_findings.json").write_text(
        "[" + ",\n".join(f.model_dump_json() for f in findings) + "]", encoding="utf-8"
    )
    return ReviewResult(report, findings, extracted, open_questions, report_path)
