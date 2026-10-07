from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from lawyer.deps import Deps
from lawyer.llm.mock import MockProvider
from lawyer.research.knowledge import KnowledgeRecord, Status
from lawyer.review import ReviewPaths, run_review
from lawyer.review.docs_io import DocReadError, load_case, read_doc
from lawyer.review.judge import UNCONFIRMED_PREFIX, sanitize
from lawyer.review.report import DISCLAIMER
from lawyer.schemas import ClaimKind, Finding, Severity, Verdict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def paths(tmp_path: Path) -> ReviewPaths:
    return ReviewPaths(
        knowledge_root=tmp_path / "knowledge",
        rules_file=ROOT / "rules" / "usn.yaml",
        checklist_file=ROOT / "skills" / "usn-marketplace-review" / "checklist.md",
        out_dir=tmp_path / "out",
    )


class TestDocs:
    def test_demo_case_loads_without_readme(self, demo_case):
        case = load_case(demo_case)
        assert set(case.docs) == {"declaration_synthetic.txt", "kudir_synthetic.txt"}
        assert case.profile.entity_type == "ip" and case.profile.tax_year == 2025

    def test_missing_profile_has_helpful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="profile.yaml"):
            load_case(tmp_path)

    def test_unsupported_format_is_unreadable_not_crash(self, demo_case):
        (demo_case / "docs" / "scan.png").write_bytes(b"\x89PNG")
        case = load_case(demo_case)
        assert any("scan.png" in u for u in case.unreadable)

    def test_xlsx_is_flattened_to_text(self, tmp_path):
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "КУДиР"
        ws.append(["Доходы за год", 5200000])
        ws.append([None, None])
        path = tmp_path / "k.xlsx"
        wb.save(path)
        text = read_doc(path)
        assert "лист: КУДиР" in text and "Доходы за год\t5200000" in text

    def test_garbage_pdf_raises_docreaderror(self, tmp_path):
        bad = tmp_path / "x.pdf"
        bad.write_bytes(b"not a pdf")
        with pytest.raises(DocReadError):
            read_doc(bad)


class TestSanitizeJudge:
    def known(self):
        return {"vat.c1"}

    def f(self, **kw) -> Finding:
        return Finding(id="j1", title="Т", severity=Severity.HIGH, detail="д", **kw)

    def test_grounded_finding_keeps_claims_and_is_marked_judge(self):
        (out,) = sanitize([self.f(claim_ids=["vat.c1"])], self.known())
        assert out.claim_ids == ["vat.c1"] and out.origin == "judge" and not out.needs_human

    def test_invented_claim_id_is_dropped_and_flagged(self):
        (out,) = sanitize([self.f(claim_ids=["vat.c1", "fake.c9"])], self.known())
        assert out.claim_ids == ["vat.c1"] and out.needs_human

    def test_ungrounded_finding_cannot_pose_as_fact(self):
        (out,) = sanitize([self.f(claim_ids=["fake.c9"])], self.known())
        assert out.title.startswith(UNCONFIRMED_PREFIX) and out.needs_human and out.claim_ids == []

    def test_prefix_is_not_applied_twice(self):
        (once,) = sanitize([self.f()], set())
        (twice,) = sanitize([once], set())
        assert twice.title.count(UNCONFIRMED_PREFIX) == 1


class TestPipeline:
    async def test_demo_case_finds_the_three_planted_errors(self, deps, demo_case, paths):
        result = await run_review(deps, demo_case, paths, today=date(2026, 10, 7))

        ids = {f.id for f in result.findings}
        assert {"chk-arith-h1", "chk-kudir-доходы-year", "chk-deadline"} <= ids
        assert result.findings[0].severity is Severity.HIGH

        report = result.report_md
        assert DISCLAIMER in report
        assert "Дата отчёта: 2026-10-07" in report
        assert "не подтверждено research-прогоном" in report  # оговорка про непроверенные правила
        assert "База знаний: **не найдена**" in report
        assert result.report_path.exists()
        saved = json.loads((paths.out_dir / "demo_findings.json").read_text())
        assert len(saved) == len(result.findings)

    async def test_judge_findings_are_merged_and_sanitized(self, settings, demo_case, paths):
        kb = paths.knowledge_root / "r1"
        kb.mkdir(parents=True)
        record = KnowledgeRecord(
            zone_id="usn-income", claim_id="usn-income.c1", text="норма", kind=ClaimKind.LAW_NORM,
            status=Status.TRUSTED, verdict=Verdict.CONFIRMED, as_of="2026-10-07",
        )  # fmt: skip
        (kb / "claims.json").write_text(json.dumps([record.model_dump(mode="json")]))
        (paths.knowledge_root / "LATEST").write_text("r1\n")

        judged = {
            "findings": [
                {"id": "j1", "title": "Доход по выплатам", "severity": "critical", "detail": "…",
                 "claim_ids": ["usn-income.c1"]},
                {"id": "j2", "title": "Выдумка", "severity": "high", "detail": "…", "claim_ids": ["nope.c1"]},
            ]
        }  # fmt: skip
        provider = MockProvider({"judge": lambda s, m: json.dumps(judged, ensure_ascii=False)})
        result = await run_review(Deps(provider=provider, settings=settings), demo_case, paths)

        by_id = {f.id: f for f in result.findings}
        assert by_id["j1"].claim_ids == ["usn-income.c1"] and not by_id["j1"].needs_human
        assert by_id["j2"].title.startswith(UNCONFIRMED_PREFIX) and by_id["j2"].needs_human
        assert result.findings[0].id == "j1"  # critical первым
        assert "База знаний: прогон `r1`" in result.report_md
        assert "⚠ проверить у специалиста" in result.report_md

    async def test_open_questions_without_ask_go_to_report(self, settings, demo_case, paths):
        provider = MockProvider(
            {"intake": lambda s, m: json.dumps({"missing": ["Какая схема: FBO или FBS?"]})}
        )
        result = await run_review(
            Deps(provider=provider, settings=settings), demo_case, paths, ask=None
        )
        assert result.open_questions == ["Какая схема: FBO или FBS?"]
        assert "Какая схема: FBO или FBS?" in result.report_md

    async def test_answers_are_saved_locally_and_not_in_tracked_context(
        self, settings, demo_case, paths
    ):
        provider = MockProvider(
            {"intake": lambda s, m: json.dumps({"missing": ["Есть ли сотрудники?"]})}
        )
        result = await run_review(
            Deps(provider=provider, settings=settings), demo_case, paths, ask=lambda q: "нет"
        )
        local = (demo_case / "context.local.md").read_text()
        assert "Q: Есть ли сотрудники?" in local and "A: нет" in local
        assert "Уточнения" not in (demo_case / "context.md").read_text()
        assert result.open_questions == []

    async def test_report_table_escapes_pipes(self, settings, demo_case, paths):
        judged = {"findings": [{"id": "j1", "title": "A | B", "severity": "low", "detail": "d"}]}
        provider = MockProvider({"judge": lambda s, m: json.dumps(judged)})
        result = await run_review(Deps(provider=provider, settings=settings), demo_case, paths)
        assert "A \\| B" in result.report_md
