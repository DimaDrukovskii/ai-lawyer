"""Заземление извлечения: каждое число, которое модель «переписала» из документа, должно
в этом документе действительно встречаться.

Без этого принцип «LLM только переписывает, считает код» держится на честном слове: выдуманное
или подсунутое текстом документа число попадало бы в детерминированные проверки как факт.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from decimal import Decimal

from ..schemas import ExtractedDocs, Finding, Severity

_SPACES = re.compile(r"[\s  ]")
MIN_DIGITS = 4  # короткие числа (6, 15, 30) встречаются в любом тексте и ничего не доказывают
MAX_LISTED = 10


def _values(extracted: ExtractedDocs) -> Iterator[tuple[str, Decimal]]:
    decl = extracted.declaration
    if decl is not None:
        for period, fig in decl.periods.items():
            for attr in ("income", "expenses", "tax_calculated", "contributions_deducted"):
                value = getattr(fig, attr)
                if value is not None:
                    yield f"декларация, {period}, {attr}", value
        if decl.tax_for_year is not None:
            yield "декларация, tax_for_year", decl.tax_for_year
    kudir = extracted.kudir
    if kudir is not None:
        for what, values in (("income", kudir.income), ("expenses", kudir.expenses)):
            for period, value in values.items():
                if value is not None:
                    yield f"КУДиР, {period}, {what}", value


def ungrounded_fields(extracted: ExtractedDocs, docs: dict[str, str]) -> list[str]:
    haystack = _SPACES.sub("", "\n".join(docs.values()))
    missing: list[str] = []
    for label, value in _values(extracted):
        needle = str(int(abs(value)))
        if len(needle) >= MIN_DIGITS and needle not in haystack:
            missing.append(f"{label} = {value}")
    return missing


def grounding_findings(extracted: ExtractedDocs, docs: dict[str, str]) -> list[Finding]:
    missing = ungrounded_fields(extracted, docs)
    if not missing:
        return []
    shown = "; ".join(missing[:MAX_LISTED]) + (" …" if len(missing) > MAX_LISTED else "")
    return [
        Finding(
            id="chk-ungrounded",
            title="Часть извлечённых чисел не найдена в тексте документов",
            severity=Severity.MEDIUM,
            detail=(
                f"Не найдено в документах ({len(missing)}): {shown}. Либо модель ошиблась при "
                "извлечении, либо число вычислено, а не записано в документе. Результаты проверок, "
                "опирающихся на эти значения, ненадёжны: сверь их с оригиналом."
            ),
            needs_human=True,
        )
    ]
