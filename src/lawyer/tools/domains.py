"""Иерархия источников. Главный предохранитель от «ИИ процитировал блог как закон».

Правило из методики: блоги допустимы только как наводка; любое число из блога обязано
быть подтверждено на первоисточнике. Список — стартовый, расширяется по мере research.
"""

from __future__ import annotations

from urllib.parse import urlparse

from ..schemas import Tier

PRIMARY_HOSTS = (
    "nalog.gov.ru", "nalog.ru", "minfin.gov.ru", "pravo.gov.ru", "sozd.duma.gov.ru",
    "duma.gov.ru", "kremlin.ru", "government.ru", "sfr.gov.ru", "cbr.ru", "vsrf.ru",
    "sudrf.ru", "arbitr.ru", "economy.gov.ru", "regulation.gov.ru", "gosuslugi.ru",
    "rosstat.gov.ru",
)  # fmt: skip
OFFICIAL_TEXT_HOSTS = ("consultant.ru", "garant.ru")
MARKETPLACE_HOSTS = ("ozon.ru", "wildberries.ru", "market.yandex.ru")
# (хост, префикс пути): публичные оферты и справка Яндекса живут на общем домене
MARKETPLACE_PATHS = (("yandex.ru", "/legal/"), ("yandex.ru", "/support/market"))
LEAD_HOSTS = (
    "klerk.ru", "glavbukh.ru", "kontur.ru", "e-kontur.ru", "1c.ru", "buh.ru", "vc.ru",
    "habr.com", "sudact.ru", "rospravosudie.com", "audit-it.ru", "taxcom.ru", "moedelo.org",
)  # fmt: skip

# Хосты для оператора site: в поисковом запросе (длина запроса к Яндексу ≤ 400 символов)
SEARCH_HOSTS: dict[str, tuple[str, ...]] = {
    "primary": (
        "nalog.gov.ru", "minfin.gov.ru", "pravo.gov.ru", "consultant.ru", "garant.ru",
        "sozd.duma.gov.ru", "sfr.gov.ru", "vsrf.ru", "arbitr.ru",
    ),
    "marketplace": MARKETPLACE_HOSTS,
    "any": (),
}  # fmt: skip

SCOPE_TIERS: dict[str, frozenset[Tier]] = {
    "primary": frozenset({Tier.PRIMARY, Tier.OFFICIAL_TEXT}),
    "marketplace": frozenset({Tier.MARKETPLACE}),
    "any": frozenset(Tier),
}


def _host_matches(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith("." + suffix)


def classify(url: str) -> Tier:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return Tier.UNKNOWN
    for host_suffix, path_prefix in MARKETPLACE_PATHS:
        if _host_matches(host, host_suffix) and parsed.path.startswith(path_prefix):
            return Tier.MARKETPLACE
    groups = (
        (Tier.PRIMARY, PRIMARY_HOSTS),
        (Tier.OFFICIAL_TEXT, OFFICIAL_TEXT_HOSTS),
        (Tier.MARKETPLACE, MARKETPLACE_HOSTS),
        (Tier.LEAD, LEAD_HOSTS),
    )
    for tier, hosts in groups:
        if any(_host_matches(host, h) for h in hosts):
            return tier
    return Tier.UNKNOWN


def is_fetch_allowed(url: str, *, allow_unknown: bool) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    return allow_unknown or classify(url) is not Tier.UNKNOWN
