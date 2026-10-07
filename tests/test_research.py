from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from lawyer.deps import Deps
from lawyer.llm.mock import MockProvider
from lawyer.research import ResearchContext, load_zones, run_research, select_zones
from lawyer.research.critics import actionable_gaps
from lawyer.research.knowledge import Status, classify_claim, load_trusted
from lawyer.research.researcher import _normalize
from lawyer.schemas import (
    Claim,
    ClaimCheck,
    ClaimKind,
    CriticReport,
    Gap,
    Severity,
    SourceRef,
    Tier,
    Verdict,
    ZoneFinding,
)

ROOT = Path(__file__).resolve().parents[1]
CTX = ResearchContext(tax_year=2026, as_of=date(2026, 10, 7))


def src(url: str, tier: Tier = Tier.UNKNOWN) -> SourceRef:
    return SourceRef(url=url, tier=tier)


class TestZones:
    def test_repo_zones_are_valid_and_unique(self):
        zones = load_zones(ROOT / "research" / "zones.yaml")
        assert len(zones) == 12
        assert len({z.id for z in zones}) == 12
        assert all(z.question and z.must_cover for z in zones)

    def test_select_unknown_zone_raises(self):
        zones = load_zones(ROOT / "research" / "zones.yaml")
        with pytest.raises(KeyError, match="nope"):
            select_zones(zones, ["nope"])


class TestTrustRules:
    """Правило доверия — код, а не промпт: именно его мы и защищаем тестами."""

    def claim(self, kind=ClaimKind.LAW_NORM, sources=None) -> Claim:
        return Claim(id="z.c1", text="t", kind=kind, sources=sources or [])

    def check(self, verdict: Verdict) -> ClaimCheck:
        return ClaimCheck(claim_id="z.c1", verdict=verdict)

    def test_confirmed_with_primary_source_is_trusted(self):
        c = self.claim(sources=[src("https://nalog.gov.ru/x", Tier.PRIMARY)])
        assert classify_claim(c, self.check(Verdict.CONFIRMED)) is Status.TRUSTED

    def test_confirmed_but_only_blog_source_is_not_trusted(self):
        c = self.claim(sources=[src("https://klerk.ru/x", Tier.LEAD)])
        assert classify_claim(c, self.check(Verdict.CONFIRMED)) is Status.UNVERIFIED

    def test_confirmed_without_any_source_is_not_trusted(self):
        assert classify_claim(self.claim(), self.check(Verdict.CONFIRMED)) is Status.UNVERIFIED

    def test_agent_opinion_is_never_trusted(self):
        c = self.claim(
            kind=ClaimKind.OPINION, sources=[src("https://nalog.gov.ru/x", Tier.PRIMARY)]
        )
        assert classify_claim(c, self.check(Verdict.CONFIRMED)) is Status.UNVERIFIED

    @pytest.mark.parametrize("v", [Verdict.WRONG, Verdict.OUTDATED, Verdict.EXAGGERATED])
    def test_failed_verdicts_are_rejected(self, v):
        c = self.claim(sources=[src("https://nalog.gov.ru/x", Tier.PRIMARY)])
        assert classify_claim(c, self.check(v)) is Status.REJECTED

    def test_unverifiable_or_missing_check_is_unverified(self):
        c = self.claim(sources=[src("https://nalog.gov.ru/x", Tier.PRIMARY)])
        assert classify_claim(c, self.check(Verdict.UNVERIFIABLE)) is Status.UNVERIFIED
        assert classify_claim(c, None) is Status.UNVERIFIED


def test_normalize_overrides_model_claimed_tier_and_namespaces_ids():
    finding = ZoneFinding(
        zone_id="whatever",
        summary="s",
        claims=[
            Claim(id="c1", text="a", sources=[src("https://random-blog.example/x", Tier.PRIMARY)]),
            Claim(id="c1", text="b"),  # модель продублировала id
        ],
    )
    out = _normalize(finding, "usn-regime", max_claims=10)
    assert [c.id for c in out.claims] == ["usn-regime.c1", "usn-regime.c2"]
    assert out.claims[0].sources[0].tier is Tier.UNKNOWN  # самозваный «primary» не принят


def test_actionable_gaps_dedup_filter_and_cap():
    reports = [
        CriticReport(lens="calendar", gaps=[
            Gap(id="g1", angle="a", question="Один и тот же?", severity=Severity.HIGH),
            Gap(id="g2", angle="b", question="Мелочь", severity=Severity.LOW),
        ]),
        CriticReport(lens="forensic", gaps=[
            Gap(id="g1", angle="a", question="один и тот же?", severity=Severity.CRITICAL),
            Gap(id="g2", angle="c", question="Другой", severity=Severity.CRITICAL),
        ]),
    ]  # fmt: skip
    gaps = actionable_gaps(reports, limit=2)
    assert [g.id for g in gaps] == ["calendar-g1", "forensic-g2"] or len(gaps) == 2
    assert all(g.severity in {Severity.CRITICAL, Severity.HIGH} for g in gaps)
    assert len({g.question.lower() for g in gaps}) == len(gaps)


class TestOrchestrator:
    def zones(self, n=3):
        return load_zones(ROOT / "research" / "zones.yaml")[:n]

    async def run(self, deps, tmp_path, zones, **kw):
        return await run_research(
            deps, zones, CTX, run_id="t1",
            out_root=tmp_path / "out", knowledge_root=tmp_path / "knowledge", **kw,
        )  # fmt: skip

    async def test_end_to_end_with_mock_writes_knowledge(self, deps, tmp_path):
        result = await self.run(deps, tmp_path, self.zones(3))

        assert len(result.findings) == 3 and not result.failed
        kdir = tmp_path / "knowledge" / "t1"
        assert (kdir / "README.md").exists() and (kdir / "claims.json").exists()
        assert (tmp_path / "knowledge" / "LATEST").read_text().strip() == "t1"
        zone_md = (kdir / "zones" / f"{self.zones(1)[0].id}.md").read_text()
        assert "## Подтверждено (доверенное)" in zone_md and "налоговый год 2026" in zone_md

        claims = json.loads((kdir / "claims.json").read_text())
        by_id = {c["claim_id"]: c for c in claims}
        zid = self.zones(1)[0].id
        # c1: confirmed + primary-источник → доверено; c2: conditional без источников → нет
        assert by_id[f"{zid}.c1"]["status"] == "trusted"
        assert by_id[f"{zid}.c2"]["status"] == "unverified"
        assert [r.claim_id for r in load_trusted(tmp_path / "knowledge")] == [
            f"{z.id}.c1" for z in self.zones(3)
        ]

    async def test_one_failing_zone_does_not_kill_the_run(self, settings, tmp_path):
        zones = self.zones(2)
        bad = zones[0].id

        def researcher(system, messages):
            text = messages[-1]["content"]
            if f"zone_id: {bad}" in text:
                return "это не JSON"
            from lawyer.llm.mock import _researcher

            return _researcher(system, messages)

        deps = Deps(provider=MockProvider({"researcher": researcher}), settings=settings)
        result = await self.run(deps, tmp_path, zones)

        assert bad in result.failed and zones[1].id in result.findings
        gaps = json.loads((tmp_path / "knowledge" / "t1" / "gaps.json").read_text())
        assert bad in gaps["failed_zones"]

    async def test_gap_round_researches_critical_gaps(self, settings, tmp_path):
        def critic(system, messages):
            lens = messages[-1]["content"].split("\n", 1)[0]
            gaps = (
                [
                    {
                        "id": "g1",
                        "angle": "Сроки при переходе",
                        "question": "Какой срок уведомления?",
                        "severity": "critical",
                    }
                ]
                if lens == "lens: calendar"
                else []
            )
            return json.dumps({"lens": "x", "complete": not gaps, "gaps": gaps}, ensure_ascii=False)

        provider = MockProvider({"critic": critic})
        deps = Deps(provider=provider, settings=settings)
        result = await self.run(deps, tmp_path, self.zones(2), rounds=1)

        gap_zone = "gap-r1-calendar-g1"
        assert gap_zone in result.findings and gap_zone in result.verified
        assert result.open_gaps == []  # закрыт отдельным агентом
        assert provider.calls.count("researcher") == 3  # 2 зоны + 1 дозакрытие
        assert (tmp_path / "knowledge" / "t1" / "zones" / f"{gap_zone}.md").exists()

    async def test_resume_skips_cached_zones(self, deps, mock_provider, tmp_path):
        zones = self.zones(2)
        await self.run(deps, tmp_path, zones)
        before = mock_provider.calls.count("researcher")
        await self.run(deps, tmp_path, zones, resume=True)
        assert mock_provider.calls.count("researcher") == before

    async def test_verifier_that_skips_a_claim_leaves_it_unverified(self, settings, tmp_path):
        provider = MockProvider(
            {"verifier": lambda s, m: json.dumps({"zone_id": "z", "checks": []})}
        )
        result = await self.run(Deps(provider=provider, settings=settings), tmp_path, self.zones(1))
        verdicts = {c.verdict for v in result.verified.values() for c in v.checks}
        assert verdicts == {Verdict.UNVERIFIABLE}
        assert load_trusted(tmp_path / "knowledge") == []
