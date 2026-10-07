from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from lawyer.review.checks import (
    check_contribution_cap,
    check_filing_deadline,
    check_kudir_vs_declaration,
    check_min_tax,
    check_rate,
    check_tax_arithmetic,
    next_working_day,
    round_rub,
    run_checks,
)
from lawyer.review.rules import Rule, RuleBook
from lawyer.schemas import (
    ClientProfile,
    ExtractedDocs,
    KudirFigures,
    PeriodFigures,
    Severity,
    UsnFigures,
)

D = Decimal


def profile(**kw) -> ClientProfile:
    base = dict(entity_type="ip", usn_object="income", region="Москва", tax_year=2025)
    return ClientProfile(**{**base, **kw})


RULES = RuleBook(
    {
        "usn_rate_income_base": Rule(value=6),
        "usn_rate_income_minus_expenses_base": Rule(value=15),
        "usn_min_tax_percent": Rule(value=1),
        "contrib_deduction_cap_percent_default": Rule(value=50),
        "contrib_deduction_cap_percent_ip_no_employees": Rule(value=100, verified=True),
        "declaration_deadline": Rule(value={"ooo": "03-25", "ip": "04-25"}),
    }
)


def decl(**periods: PeriodFigures) -> UsnFigures:
    return UsnFigures(periods=periods)


class TestRounding:
    def test_half_up_per_art_52(self):
        assert round_rub(D("100.49")) == D("100")
        assert round_rub(D("100.50")) == D("101")  # банковское округление дало бы 100

    def test_next_working_day_skips_weekend_and_holidays(self):
        assert next_working_day(date(2026, 4, 25)) == date(2026, 4, 27)  # суббота → понедельник
        assert next_working_day(date(2026, 4, 27), frozenset({date(2026, 4, 27)})) == date(
            2026, 4, 28
        )
        assert next_working_day(date(2026, 4, 24)) == date(2026, 4, 24)


class TestArithmetic:
    def test_correct_income_tax_passes(self):
        d = decl(
            h1=PeriodFigures(income=D(2_500_000), rate_percent=D(6), tax_calculated=D(150_000))
        )
        assert check_tax_arithmetic(profile(), d) == []

    def test_understated_tax_is_flagged_with_price(self):
        d = decl(
            h1=PeriodFigures(income=D(2_500_000), rate_percent=D(6), tax_calculated=D(125_000))
        )
        (f,) = check_tax_arithmetic(profile(), d)
        assert f.id == "chk-arith-h1" and f.severity is Severity.HIGH
        assert "25 000" in f.price_of_error and "Занижение" in f.price_of_error

    def test_one_ruble_tolerance(self):
        d = decl(q1=PeriodFigures(income=D(1_000_000), rate_percent=D(6), tax_calculated=D(60_001)))
        assert check_tax_arithmetic(profile(), d) == []

    def test_income_minus_expenses_uses_difference_and_floors_at_zero(self):
        p = profile(usn_object="income_minus_expenses")
        ok = decl(
            year=PeriodFigures(
                income=D(1_000_000),
                expenses=D(800_000),
                rate_percent=D(15),
                tax_calculated=D(30_000),
            )
        )
        loss = decl(
            year=PeriodFigures(
                income=D(100), expenses=D(500), rate_percent=D(15), tax_calculated=D(0)
            )
        )
        assert check_tax_arithmetic(p, ok) == [] and check_tax_arithmetic(p, loss) == []

    def test_missing_data_is_silent_not_failed(self):
        assert check_tax_arithmetic(profile(), decl(h1=PeriodFigures(income=D(1)))) == []


class TestKudirVsDeclaration:
    def test_mismatch_year_income(self):
        d = decl(year=PeriodFigures(income=D(5_000_000)))
        k = KudirFigures(income={"year": D(5_200_000)})
        (f,) = check_kudir_vs_declaration(profile(), d, k)
        assert f.id == "chk-kudir-доходы-year"
        assert "200 000" in f.detail and "занижена" in f.price_of_error

    def test_equal_within_tolerance(self):
        d = decl(year=PeriodFigures(income=D(5_000_000)))
        assert (
            check_kudir_vs_declaration(profile(), d, KudirFigures(income={"year": D("5000000.60")}))
            == []
        )

    def test_expenses_checked_only_for_income_minus_expenses(self):
        d = decl(year=PeriodFigures(income=D(10), expenses=D(5)))
        k = KudirFigures(income={"year": D(10)}, expenses={"year": D(9)})
        assert check_kudir_vs_declaration(profile(), d, k) == []
        assert (
            len(check_kudir_vs_declaration(profile(usn_object="income_minus_expenses"), d, k)) == 1
        )


class TestContributionCap:
    def test_ip_without_employees_may_deduct_all(self):
        d = decl(year=PeriodFigures(tax_calculated=D(100_000), contributions_deducted=D(100_000)))
        assert check_contribution_cap(profile(), d, RULES) == []

    def test_ooo_over_half_is_flagged_and_caveat_added(self):
        d = decl(year=PeriodFigures(tax_calculated=D(100_000), contributions_deducted=D(60_000)))
        (f,) = check_contribution_cap(profile(entity_type="ooo"), d, RULES)
        assert f.id == "chk-contrib-year" and "10 000" in f.price_of_error
        assert "не подтверждено" in f.detail  # правило default не verified

    def test_ip_with_employees_uses_default_cap(self):
        d = decl(year=PeriodFigures(tax_calculated=D(100_000), contributions_deducted=D(60_000)))
        assert len(check_contribution_cap(profile(has_employees=True), d, RULES)) == 1

    def test_missing_rule_reports_inability_instead_of_passing(self):
        d = decl(year=PeriodFigures(tax_calculated=D(100_000), contributions_deducted=D(999_999)))
        (f,) = check_contribution_cap(profile(), d, RuleBook({}))
        assert f.id == "chk-contrib-norule" and f.needs_human and f.severity is Severity.LOW

    def test_not_applicable_for_income_minus_expenses(self):
        d = decl(year=PeriodFigures(tax_calculated=D(1), contributions_deducted=D(9)))
        assert check_contribution_cap(profile(usn_object="income_minus_expenses"), d, RULES) == []


class TestMinTax:
    def test_below_minimum_flagged(self):
        d = UsnFigures(
            periods={"year": PeriodFigures(income=D(10_000_000))}, tax_for_year=D(50_000)
        )
        (f,) = check_min_tax(profile(usn_object="income_minus_expenses"), d, RULES)
        assert "50 000" in f.price_of_error  # минимум 100 000, заплачено 50 000

    def test_at_or_above_minimum_passes(self):
        d = UsnFigures(
            periods={"year": PeriodFigures(income=D(10_000_000))}, tax_for_year=D(100_000)
        )
        assert check_min_tax(profile(usn_object="income_minus_expenses"), d, RULES) == []


class TestDeadline:
    def d(self, filed: date) -> UsnFigures:
        return UsnFigures(filed_on=filed)

    def test_ip_deadline_moves_from_saturday_to_monday(self):
        assert check_filing_deadline(profile(), self.d(date(2026, 4, 27)), RULES) == []
        (f,) = check_filing_deadline(profile(), self.d(date(2026, 4, 28)), RULES)
        assert "2026-04-27" in f.detail and "1 дн" in f.detail

    def test_ooo_uses_march_25(self):
        # 25 марта 2026 — среда
        assert (
            check_filing_deadline(profile(entity_type="ooo"), self.d(date(2026, 3, 25)), RULES)
            == []
        )
        assert (
            len(check_filing_deadline(profile(entity_type="ooo"), self.d(date(2026, 3, 26)), RULES))
            == 1
        )

    def test_holidays_extend_deadline_when_provided(self):
        h = frozenset({date(2026, 4, 27)})
        assert check_filing_deadline(profile(), self.d(date(2026, 4, 28)), RULES, h) == []

    def test_no_filing_date_is_silent(self):
        assert check_filing_deadline(profile(), UsnFigures(), RULES) == []


class TestRate:
    def test_higher_and_lower_rates_need_human(self):
        d = decl(q1=PeriodFigures(rate_percent=D(8)), year=PeriodFigures(rate_percent=D(5)))
        findings = check_rate(profile(), d, RULES)
        assert {f.id for f in findings} == {"chk-rate-high-8", "chk-rate-low-5"}
        assert all(f.needs_human for f in findings)

    def test_base_rate_is_clean(self):
        assert check_rate(profile(), decl(q1=PeriodFigures(rate_percent=D(6))), RULES) == []


class TestRuleBook:
    def test_null_value_means_rule_absent(self):
        rb = RuleBook({"x": Rule(value=None, note="заполнить")})
        assert rb.get("x") is None and rb.unverified_keys() == []

    def test_unverified_keys_and_caveat(self):
        rb = RuleBook({"a": Rule(value=1), "b": Rule(value=2, verified=True)})
        assert rb.unverified_keys() == ["a"]
        assert rb.caveat("a") and rb.caveat("b") == ""

    def test_repo_rules_file_loads(self):
        from pathlib import Path

        rb = RuleBook.from_yaml(Path(__file__).parents[1] / "rules" / "usn.yaml")
        assert rb.get("declaration_deadline") is not None
        assert rb.get("income_limit_rub") is None  # ждёт research


def test_run_checks_sorts_by_severity_and_skips_kudir_when_absent():
    d = UsnFigures(
        periods={
            "h1": PeriodFigures(income=D(2_500_000), rate_percent=D(5), tax_calculated=D(125_000)),
        },
        filed_on=date(2026, 5, 20),
    )
    findings = run_checks(profile(), ExtractedDocs(declaration=d), RULES)
    severities = [f.severity for f in findings]
    assert severities == sorted(
        severities, key=lambda s: ["critical", "high", "medium", "low"].index(s.value)
    )
    assert any(f.id == "chk-deadline" for f in findings)


@pytest.mark.parametrize("extracted", [ExtractedDocs(), ExtractedDocs(kudir=KudirFigures())])
def test_run_checks_without_declaration_returns_nothing(extracted):
    assert run_checks(profile(), extracted, RULES) == []
