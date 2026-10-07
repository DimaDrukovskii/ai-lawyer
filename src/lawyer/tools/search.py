from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Protocol

import httpx
from defusedxml import ElementTree  # защита от XML-бомб в ответе внешнего сервиса

from ..schemas import Tier
from .domains import SCOPE_TIERS, SEARCH_HOSTS, classify

SEARCH_URL = "https://searchapi.api.cloud.yandex.net/v2/web/search"
MAX_QUERY_CHARS = 400  # лимит Yandex Search API на queryText
RUSSIA_REGION = "225"


class SearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str
    snippet: str
    tier: Tier
    modtime: str = ""


class SearchBackend(Protocol):
    async def search(
        self, query: str, *, scope: str = "primary", limit: int = 8
    ) -> list[SearchHit]: ...


def build_query(query: str, scope: str) -> str:
    """Дописывает site:-фильтр по официальным доменам, пока влезает в лимит запроса."""
    hosts = SEARCH_HOSTS.get(scope, ())
    base = query.strip()[:MAX_QUERY_CHARS]
    picked: list[str] = []
    for host in hosts:
        candidate = "(" + " | ".join(f"site:{h}" for h in [*picked, host]) + ") " + base
        if len(candidate) > MAX_QUERY_CHARS:
            break
        picked.append(host)
    if not picked:
        return base
    return "(" + " | ".join(f"site:{h}" for h in picked) + ") " + base


def parse_results(xml_bytes: bytes) -> list[SearchHit]:
    try:
        root = ElementTree.fromstring(xml_bytes)
    except (ElementTree.ParseError, ValueError) as exc:  # ValueError ⊇ DefusedXmlException
        raise SearchError(f"невалидный XML от Search API: {exc}") from exc
    error = root.find(".//error")
    if error is not None:
        raise SearchError(
            f"Search API: {error.get('code', '?')} {''.join(error.itertext()).strip()}"
        )
    hits: list[SearchHit] = []
    for doc in root.iter("doc"):
        url = (doc.findtext("url") or "").strip()
        if not url:
            continue
        title_el = doc.find("title")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        snippet = " … ".join("".join(p.itertext()).strip() for p in doc.iter("passage"))
        hits.append(
            SearchHit(
                url=url,
                title=title,
                snippet=snippet,
                tier=classify(url),
                modtime=(doc.findtext("modtime") or "").strip(),
            )
        )
    return hits


class YandexSearch:
    def __init__(
        self, *, api_key: str, folder_id: str, http_client: httpx.AsyncClient | None = None
    ) -> None:
        if not api_key or not folder_id:
            raise SearchError(
                "для поиска нужны YANDEX_SEARCH_API_KEY (или YANDEX_API_KEY) и YANDEX_FOLDER_ID"
            )
        self._api_key = api_key
        self._folder_id = folder_id
        self._http = http_client or httpx.AsyncClient(timeout=30.0)

    async def search(
        self, query: str, *, scope: str = "primary", limit: int = 8
    ) -> list[SearchHit]:
        body = {
            "query": {
                "searchType": "SEARCH_TYPE_RU",
                "queryText": build_query(query, scope),
                "familyMode": "FAMILY_MODE_NONE",
                "page": 0,
                "fixTypoMode": "FIX_TYPO_MODE_ON",
            },
            "sortSpec": {"sortMode": "SORT_MODE_BY_RELEVANCE"},
            "groupSpec": {
                "groupMode": "GROUP_MODE_FLAT",
                "groupsOnPage": max(1, min(limit * 2, 100)),
                "docsInGroup": 1,
            },
            "maxPassages": 3,
            "region": RUSSIA_REGION,
            "l10n": "LOCALIZATION_RU",
            "folderId": self._folder_id,
            "responseFormat": "FORMAT_XML",
        }
        try:
            resp = await self._http.post(
                SEARCH_URL, json=body, headers={"Authorization": f"Api-Key {self._api_key}"}
            )
            resp.raise_for_status()
            raw = base64.b64decode(resp.json()["rawData"])
        except (httpx.HTTPError, KeyError, binascii.Error, ValueError) as exc:
            raise SearchError(f"запрос к Search API не удался: {exc}") from exc
        allowed = SCOPE_TIERS.get(scope, SCOPE_TIERS["any"])
        # Пост-фильтр по tier — страховка, если site: оператор отработал не полностью
        return [h for h in parse_results(raw) if h.tier in allowed][:limit]
