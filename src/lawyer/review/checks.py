"""Детерминированные проверки: чистые функции без LLM. Цифры считает код, не модель.

Правило: нет данных или нет правила — проверка честно говорит «не могу проверить»
либо молчит, но никогда не «проходит» молча.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from ..schemas import (
    ClientProfile,
    ExtractedDocs,
    Finding,
    KudirFigures,
    Period,
    Severity,
    UsnFigures,
)
from .rules import RuleBook

TOLERANCE_RUB = Decimal("1")  # в декларации целые рубли, в КУДиР — копейки
PERIOD_LABEL: dict[str, str] = {
    "q1": "1 квартал",
    "h1": "полугодие",
    "m9": "9 месяцев",
    "year": "год",
}
PERIOD_ORDER: tuple[Period, ...] = ("q1", "h1", "m9", "year")
SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}


def round_rub(x: Decimal) -> Decimal:
    """Округление налога по ст. 52 НК: до рубля, 50 копеек и больше — вверх."""
    return x.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def next_working_day(d: date, holidays: frozenset[date] = frozenset()) -> date:
    while d.weekday() >= 5 or d in holidays:
        d += timedelta(days=1)
    return d


def _fmt(x: Decimal) -> str:
    return f"{x:,.2f}".replace(",", " ")


def _label(period: str) -> str:
    return PERIOD_LABEL[period]


# ---------------------------------------------------------------- отдельные проверки


def check_tax_arithmetic(profile: ClientProfile, decl: UsnFigures) -> list[Finding]:
    findings: list[Finding] = []
    for period in PERIOD_ORDER:
        fig = decl.periods.get(period)
        if (
            fig is None
            or fig.income is None
            or fig.rate_percent is None
            or fig.tax_calculated is None
        ):
            continue
        if profile.usn_object == "income":
            base = fig.income
        elif fig.expenses is not None:
            base = max(fig.income - fig.expenses, Decimal(0))
        else:
            continue
        expected = round_rub(base * fig.rate_percent / Decimal(100))
        diff = fig.tax_calculated - expected
        if abs(diff) <= TOLERANCE_RUB:
            continue
        under = diff < 0
        # При «доходы минус расходы» налог ниже расчётного может быть законным: базу уменьшает
        # убыток прошлых лет, а в извлечённых данных его нет. Это вопрос, а не ошибка.
        maybe_loss = under and profile.usn_object == "income_minus_expenses"
        loss_note = (
            " Если у клиента есть убыток прошлых лет, он уменьшает базу: уточни у бухгалтера."
            if maybe_loss
            else ""
        )
        findings.append(
            Finding(
                id=f"chk-arith-{period}",
                title=f"Налог за {_label(period)} посчитан неверно",
                severity=Severity.MEDIUM if maybe_loss else Severity.HIGH,
                needs_human=maybe_loss,
                detail=(
                    f"Исчисленный налог {_fmt(fig.tax_calculated)} ₽, а по базе {_fmt(base)} ₽ "
                    f"и ставке {fig.rate_percent}% получается {_fmt(expected)} ₽ "
                    f"(разница {_fmt(diff)} ₽)." + loss_note
                ),
                price_of_error=(
                    f"Занижение налога на {_fmt(-diff)} ₽ нарастающим итогом: риск доначисления, "
                    "пеней и штрафа."
                    if under
                    else f"Завышение налога на {_fmt(diff)} ₽: переплата."
                ),
            )
        )
    return findings


def check_kudir_vs_declaration(
    profile: ClientProfile, decl: UsnFigures, kudir: KudirFigures
) -> list[Finding]:
    findings: list[Finding] = []
    pairs: list[tuple[str, dict[Period, Decimal | None], Callable[[Period], Decimal | None]]] = [
        ("доходы", kudir.income, lambda p: decl.periods[p].income if p in decl.periods else None),
    ]
    if profile.usn_object == "income_minus_expenses":
        pairs.append(
            (
                "расходы",
                kudir.expenses,
                lambda p: decl.periods[p].expenses if p in decl.periods else None,
            )
        )
    for what, kudir_values, decl_value in pairs:
        for period in PERIOD_ORDER:
            k, d = kudir_values.get(period), decl_value(period)
            if k is None or d is None or abs(k - d) <= TOLERANCE_RUB:
                continue
            findings.append(
                Finding(
                    id=f"chk-kudir-{what}-{period}",
                    title=f"КУДиР и декларация расходятся по статье «{what}» за {_label(period)}",
                    severity=Severity.HIGH,
                    detail=(
                        f"В КУДиР {what} {_fmt(k)} ₽, в декларации {_fmt(d)} ₽, "
                        f"разница {_fmt(k - d)} ₽. Показатели обязаны совпадать."
                    ),
                    price_of_error=(
                        "Если верна КУДиР, налоговая база в декларации занижена."
                        if what == "доходы" and k > d
                        else "Нужно понять, какой из документов верен, и подать уточнённую декларацию."
                    ),
                )
            )
    return findings


def check_contribution_cap(
    profile: ClientProfile, decl: UsnFigures, rules: RuleBook
) -> list[Finding]:
    if profile.usn_object != "income":
        return []
    has_data = any(f.contributions_deducted is not None for f in decl.periods.values())
    if not has_data:
        return []
    key = (
        "contrib_deduction_cap_percent_ip_no_employees"
        if profile.entity_type == "ip" and not profile.has_employees
        else "contrib_deduction_cap_percent_default"
    )
    rule = rules.get(key)
    if rule is None:
        return [
            Finding(
                id="chk-contrib-norule",
                title="Не могу проверить предельный вычет взносов",
                severity=Severity.LOW,
                detail=f"В rules/usn.yaml нет правила {key}. Нужен research (зона contributions).",
                needs_human=True,
            )
        ]
    cap_pct = Decimal(str(rule.value))
    findings: list[Finding] = []
    for period in PERIOD_ORDER:
        fig = decl.periods.get(period)
        if fig is None or fig.tax_calculated is None or fig.contributions_deducted is None:
            continue
        cap = round_rub(fig.tax_calculated * cap_pct / Decimal(100))
        if fig.contributions_deducted - cap > TOLERANCE_RUB:
            excess = fig.contributions_deducted - cap
            findings.append(
                Finding(
                    id=f"chk-contrib-{period}",
                    title=f"Вычет взносов за {_label(period)} превышает предел",
                    severity=Severity.HIGH,
                    detail=(
                        f"Взносы к вычету {_fmt(fig.contributions_deducted)} ₽ при исчисленном налоге "
                        f"{_fmt(fig.tax_calculated)} ₽ и пределе {cap_pct}% ({_fmt(cap)} ₽). "
                        f"Превышение {_fmt(excess)} ₽." + rules.caveat(key)
                    ),
                    price_of_error=f"Занижение налога на {_fmt(excess)} ₽ за период.",
                )
            )
    return findings


def check_min_tax(profile: ClientProfile, decl: UsnFigures, rules: RuleBook) -> list[Finding]:
    if profile.usn_object != "income_minus_expenses":
        return []
    year = decl.periods.get("year")
    rule = rules.get("usn_min_tax_percent")
    if year is None or year.income is None or decl.tax_for_year is None or rule is None:
        return []
    expected = round_rub(year.income * Decimal(str(rule.value)) / Decimal(100))
    if expected - decl.tax_for_year <= TOLERANCE_RUB:
        return []
    return [
        Finding(
            id="chk-min-tax",
            title="Налог за год ниже минимального",
            severity=Severity.HIGH,
            detail=(
                f"Налог за год {_fmt(decl.tax_for_year)} ₽, минимальный налог {rule.value}% от дохода "
                f"{_fmt(year.income)} ₽ = {_fmt(expected)} ₽." + rules.caveat("usn_min_tax_percent")
            ),
            price_of_error=f"Недоплата {_fmt(expected - decl.tax_for_year)} ₽ за год.",
        )
    ]


def check_filing_deadline(
    profile: ClientProfile,
    decl: UsnFigures,
    rules: RuleBook,
    holidays: frozenset[date] = frozenset(),
) -> list[Finding]:
    rule = rules.get("declaration_deadline")
    if decl.filed_on is None or rule is None:
        return []
    month, day = (int(x) for x in str(rule.value[profile.entity_type]).split("-"))
    nominal = date(profile.tax_year + 1, month, day)
    deadline = next_working_day(nominal, holidays)
    if decl.filed_on <= deadline:
        return []
    return [
        Finding(
            id="chk-deadline",
            title="Декларация подана после срока",
            severity=Severity.HIGH,
            detail=(
                f"Подана {decl.filed_on.isoformat()}, срок {deadline.isoformat()} "
                f"(номинально {nominal.isoformat()}, перенос на рабочий день). "
                f"Просрочка {(decl.filed_on - deadline).days} дн. "
                "Нерабочие праздничные дни учитываются только если переданы в проверку."
                + rules.caveat("declaration_deadline")
            ),
            price_of_error="Штраф за несвоевременную подачу; размер и условия — см. базу знаний (зона penalties).",
        )
    ]


def check_rate(profile: ClientProfile, decl: UsnFigures, rules: RuleBook) -> list[Finding]:
    key = (
        "usn_rate_income_base"
        if profile.usn_object == "income"
        else "usn_rate_income_minus_expenses_base"
    )
    rule = rules.get(key)
    if rule is None:
        return []
    base = Decimal(str(rule.value))
    rates = {f.rate_percent for f in decl.periods.values() if f.rate_percent is not None}
    findings: list[Finding] = []
    for rate in sorted(rates):
        if rate > base:
            findings.append(
                Finding(
                    id=f"chk-rate-high-{rate}",
                    title=f"Ставка {rate}% выше базовой {base}%",
                    severity=Severity.MEDIUM,
                    detail=(
                        "Повышенная ставка допустима только при превышении установленного порога. "
                        "Убедись, что порог превышен и ставка применена с правильного периода."
                        + rules.caveat(key)
                    ),
                    needs_human=True,
                )
            )
        elif rate < base:
            findings.append(
                Finding(
                    id=f"chk-rate-low-{rate}",
                    title=f"Ставка {rate}% ниже базовой {base}%",
                    severity=Severity.LOW,
                    detail=(
                        f"Пониженная ставка возможна по закону региона (регион клиента: "
                        f"{profile.region}). Проверь закон региона и условия применения."
                        + rules.caveat(key)
                    ),
                    needs_human=True,
                )
            )
    return findings


# ---------------------------------------------------------------- сборка


def coverage_gaps(extracted: ExtractedDocs) -> list[Finding]:
    """Проверка, которая не смогла выполниться, обязана об этом сказать.

    Иначе пустое или усечённое извлечение выглядит как «ошибок нет».
    """

    def gap(id_: str, title: str, detail: str, severity: Severity) -> Finding:
        return Finding(id=id_, title=title, severity=severity, detail=detail, needs_human=True)

    out: list[Finding] = []
    decl = extracted.declaration
    if decl is None:
        out.append(
            gap(
                "chk-coverage-declaration",
                "Декларация не распознана: расчётные проверки не выполнены",
                "Из документов не удалось извлечь показатели декларации (нет файла, скан без "
                "текстового слоя или ответ модели не разобран). Арифметика налога, взносы, "
                "ставка, срок и сверка с КУДиР НЕ проверялись. Это не означает отсутствие ошибок.",
                Severity.HIGH,
            )
        )
    elif not decl.periods:
        out.append(
            gap(
                "chk-coverage-periods",
                "В декларации не извлечено ни одного периода",
                "Арифметика налога, предел вычета взносов и ставка НЕ проверялись.",
                Severity.HIGH,
            )
        )
    kudir = extracted.kudir
    if kudir is None or not (kudir.income or kudir.expenses):
        out.append(
            gap(
                "chk-coverage-kudir",
                "КУДиР не распознан: сверка с декларацией не выполнена",
                "Совпадение доходов и расходов КУДиР и декларации НЕ проверялось.",
                Severity.MEDIUM,
            )
        )
    return out


def run_checks(
    profile: ClientProfile,
    extracted: ExtractedDocs,
    rules: RuleBook,
    holidays: frozenset[date] = frozenset(),
) -> list[Finding]:
    findings: list[Finding] = coverage_gaps(extracted)
    decl = extracted.declaration
    if decl is not None:
        findings += check_tax_arithmetic(profile, decl)
        findings += check_contribution_cap(profile, decl, rules)
        findings += check_min_tax(profile, decl, rules)
        findings += check_filing_deadline(profile, decl, rules, holidays)
        findings += check_rate(profile, decl, rules)
        if extracted.kudir is not None:
            findings += check_kudir_vs_declaration(profile, decl, extracted.kudir)
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.id))
