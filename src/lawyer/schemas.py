"""Доменные модели. Все иммутабельны (frozen): изменения — через model_copy(update=...)."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict


class _Model(BaseModel):
    # extra="ignore": LLM любит добавлять лишние поля, это не повод падать
    model_config = ConfigDict(frozen=True, extra="ignore")


# ---------------------------------------------------------------- источники и утверждения


class Tier(StrEnum):
    PRIMARY = "primary"  # ФНС, Минфин, pravo.gov.ru, суды: нормы и разъяснения
    OFFICIAL_TEXT = "official_text"  # КонсультантПлюс/Гарант: текст норм, не первоисточник
    MARKETPLACE = "marketplace"  # оферты и справка WB / Ozon / Яндекс Маркета
    LEAD = "lead"  # блоги и СМИ: ТОЛЬКО как наводка, число из блога не принимается
    UNKNOWN = "unknown"


class ClaimKind(StrEnum):
    LAW_NORM = "law_norm"  # норма закона, действует безусловно
    CONDITIONAL = "conditional"  # применение зависит от факта, которого мы не знаем
    OPINION = "opinion"  # вывод самого агента, не норма


class Verdict(StrEnum):
    CONFIRMED = "confirmed"
    WRONG = "wrong"
    OUTDATED = "outdated"
    EXAGGERATED = "exaggerated"
    UNVERIFIABLE = "unverifiable"


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceRef(_Model):
    url: str
    title: str = ""
    doc_date: str = ""  # дата документа / редакции, как указана на источнике
    quote: str = ""  # короткая дословная цитата, по которой можно перепроверить
    tier: Tier = Tier.UNKNOWN


class Claim(_Model):
    id: str
    text: str
    kind: ClaimKind = ClaimKind.LAW_NORM
    condition: str = ""  # для CONDITIONAL: от какого факта зависит
    numbers: list[str] = []  # все суммы/пороги/проценты/даты в утверждении
    sources: list[SourceRef] = []


class ZoneFinding(_Model):
    zone_id: str
    summary: str
    claims: list[Claim] = []
    confidence: Literal["high", "medium", "low"] = "medium"
    unverified: list[str] = []


class ClaimCheck(_Model):
    claim_id: str
    verdict: Verdict
    correction: str = ""
    evidence_urls: list[str] = []
    note: str = ""


class VerifiedZone(_Model):
    zone_id: str
    checks: list[ClaimCheck] = []
    note: str = ""


class Gap(_Model):
    id: str
    angle: str
    question: str
    severity: Severity = Severity.HIGH


class CriticReport(_Model):
    lens: str
    complete: bool = False
    gaps: list[Gap] = []


# ---------------------------------------------------------------- профиль клиента и документы


def _to_decimal(v: object) -> Decimal | None:
    if v is None or isinstance(v, Decimal):
        return v
    if isinstance(v, bool):
        raise ValueError("bool is not a money value")
    if isinstance(v, int | float):
        return Decimal(str(v))
    if isinstance(v, str):
        s = re.sub(r"[^\d.\-]", "", v.replace(" ", "").replace(" ", "").replace(",", "."))
        if s in {"", "-", "."}:
            return None
        return Decimal(s)
    raise ValueError(f"cannot parse money: {v!r}")


Money = Annotated[Decimal | None, BeforeValidator(_to_decimal)]
Period = Literal["q1", "h1", "m9", "year"]  # нарастающим итогом: 1 кв., полугодие, 9 мес., год


class ClientProfile(_Model):
    entity_type: Literal["ip", "ooo"]
    usn_object: Literal["income", "income_minus_expenses"]
    region: str
    tax_year: int
    has_employees: bool = False
    marketplaces: list[str] = []  # ["wb", "ozon", "ym"]
    vat: Literal["exempt", "payer", "unknown"] = "unknown"


class PeriodFigures(_Model):
    income: Money = None  # доходы нарастающим итогом
    expenses: Money = None  # расходы нарастающим итогом (только «доходы минус расходы»)
    rate_percent: Money = None
    tax_calculated: Money = None  # исчисленный налог ДО вычета взносов
    contributions_deducted: Money = None  # взносы, уменьшающие налог («доходы»)


class UsnFigures(_Model):
    """Нормализованные показатели декларации по УСН. Привязка к строкам формы — в extract."""

    periods: dict[Period, PeriodFigures] = {}
    tax_for_year: Money = None  # налог за год к зачёту авансов (с учётом минимального налога)
    filed_on: date | None = None
    form_version: str = ""


class KudirFigures(_Model):
    income: dict[Period, Money] = {}
    expenses: dict[Period, Money] = {}


class Evidence(_Model):
    field: str
    doc: str
    quote: str


class ExtractedDocs(_Model):
    declaration: UsnFigures | None = None
    kudir: KudirFigures | None = None
    evidence: list[Evidence] = []
    unreadable: list[str] = []


class Finding(_Model):
    id: str
    title: str
    severity: Severity
    detail: str
    price_of_error: str = ""
    origin: Literal["check", "judge"] = "check"
    claim_ids: list[str] = []  # id утверждений из базы знаний, на которые опирается вывод
    needs_human: bool = False
