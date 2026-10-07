"""Иерархия источников. Главный предохранитель от «ИИ процитировал блог как закон».

Правило из методики: блоги допустимы только как наводка; любое число из блога обязано
быть подтверждено на первоисточнике. Список — стартовый, расширяется по мере research.

Уровень доверия выдаётся не целому домену, а там, где это нужно, — поддомену или пути:
на ozon.ru лежат карточки товаров от продавцов, на garant.ru и consultant.ru — статьи и
новости рядом с текстами норм. Такие страницы первоисточником не являются.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import ParseResult, urlparse

from ..schemas import Tier

PRIMARY_HOSTS = (
    "nalog.gov.ru", "nalog.ru", "minfin.gov.ru", "pravo.gov.ru", "sozd.duma.gov.ru",
    "duma.gov.ru", "kremlin.ru", "government.ru", "sfr.gov.ru", "cbr.ru", "vsrf.ru",
    "sudrf.ru", "arbitr.ru", "economy.gov.ru", "regulation.gov.ru", "gosuslugi.ru",
    "rosstat.gov.ru",
)  # fmt: skip
# Текст норм: только документная часть, не новости и не статьи
OFFICIAL_TEXT_HOSTS = ("base.garant.ru",)
OFFICIAL_TEXT_PATHS = (("consultant.ru", "/document/"),)
# Только кабинеты и справка продавца: www.ozon.ru, www.wildberries.ru, market.yandex.ru
# — это витрины с пользовательским контентом. Список уточнить по итогам первого research.
MARKETPLACE_HOSTS = (
    "docs.ozon.ru", "seller.ozon.ru", "seller-edu.ozon.ru",
    "dev.wildberries.ru", "seller.wildberries.ru", "partner.market.yandex.ru",
)  # fmt: skip
MARKETPLACE_PATHS = (("yandex.ru", "/legal/"), ("yandex.ru", "/support/market"))
LEAD_HOSTS = (
    "klerk.ru", "glavbukh.ru", "kontur.ru", "e-kontur.ru", "1c.ru", "buh.ru", "vc.ru",
    "habr.com", "sudact.ru", "rospravosudie.com", "audit-it.ru", "taxcom.ru", "moedelo.org",
    "garant.ru", "consultant.ru", "ozon.ru", "wildberries.ru", "market.yandex.ru",
)  # fmt: skip

# Хосты для оператора site: в поисковом запросе (длина запроса к Яндексу ≤ 400 символов)
SEARCH_HOSTS: dict[str, tuple[str, ...]] = {
    "primary": (
        "nalog.gov.ru", "minfin.gov.ru", "pravo.gov.ru", "consultant.ru", "base.garant.ru",
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


def _parse(url: str) -> ParseResult | None:
    """urlparse бросает ValueError на мусоре вроде 'http://[::1'; URL приходят от модели."""
    try:
        return urlparse(url)
    except ValueError:
        return None


def classify(url: str) -> Tier:
    parsed = _parse(url)
    if parsed is None:
        return Tier.UNKNOWN
    host = (parsed.hostname or "").lower()
    if not host:
        return Tier.UNKNOWN
    path_rules = (
        (Tier.MARKETPLACE, MARKETPLACE_PATHS),
        (Tier.OFFICIAL_TEXT, OFFICIAL_TEXT_PATHS),
    )
    for tier, rules in path_rules:
        if any(_host_matches(host, h) and parsed.path.startswith(p) for h, p in rules):
            return tier
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


def normalize_url(url: str) -> str:
    """Ключ для сравнения ссылок: без схемы, www, фрагмента и завершающего слеша."""
    parsed = _parse(url.strip())
    if parsed is None:
        return url.strip().lower()
    host = (parsed.hostname or "").lower().removeprefix("www.")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{parsed.path.rstrip('/')}{query}"


def _is_public_host(host: str) -> bool:
    """Отсекает localhost и IP-литералы из частных, loopback и link-local диапазонов."""
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return True  # не IP-литерал: обычное имя хоста


def is_fetch_allowed(url: str, *, allow_unknown: bool) -> bool:
    parsed = _parse(url)
    if parsed is None or parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not host or not _is_public_host(host):
        return False
    return allow_unknown or classify(url) is not Tier.UNKNOWN
