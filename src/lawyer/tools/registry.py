from __future__ import annotations

from typing import Any

from ..llm.base import ToolSpec
from .fetch import Fetcher, FetchError, Page
from .search import SearchBackend, SearchError

SEARCH_SPEC = ToolSpec(
    name="web_search",
    description=(
        "Поиск по официальным источникам РФ. scope=primary: ФНС, Минфин, pravo.gov.ru, "
        "КонсультантПлюс/Гарант, суды. scope=marketplace: оферты и справка WB/Ozon/Яндекс Маркета. "
        "scope=any: любые сайты (результаты из блогов — только наводка, не источник)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Поисковый запрос на русском, до 300 символов",
            },
            "scope": {"type": "string", "enum": ["primary", "marketplace", "any"]},
        },
        "required": ["query"],
    },
)

FETCH_SPEC = ToolSpec(
    name="fetch_url",
    description=(
        "Загружает страницу или PDF и возвращает текст с датой получения. Длинные документы "
        "отдаются кусками: продолжай с offset из подсказки. Только домены из белого списка."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "offset": {"type": "integer", "description": "Смещение в символах, по умолчанию 0"},
        },
        "required": ["url"],
    },
)


class ToolRegistry:
    """Набор инструментов агента. call() никогда не бросает: ошибка — это тоже ответ модели."""

    def __init__(
        self,
        *,
        search: SearchBackend | None = None,
        fetcher: Fetcher | None = None,
        chunk_chars: int = 12_000,
    ) -> None:
        self._search = search
        self._fetcher = fetcher
        self._chunk = chunk_chars
        self._cache: dict[str, Page] = {}

    def specs(self) -> tuple[ToolSpec, ...]:
        specs: list[ToolSpec] = []
        if self._search is not None:
            specs.append(SEARCH_SPEC)
        if self._fetcher is not None:
            specs.append(FETCH_SPEC)
        return tuple(specs)

    async def call(self, name: str, args: dict[str, Any]) -> str:
        if "_error" in args:
            return f"ERROR: {args['_error']}"
        try:
            if name == "web_search" and self._search is not None:
                return await self._web_search(args)
            if name == "fetch_url" and self._fetcher is not None:
                return await self._fetch_url(args)
        except (SearchError, FetchError) as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # граница инструмента: аргументы пришли от модели, падать нельзя
            return f"ERROR: {type(exc).__name__}: {exc}"[:300]
        return f"ERROR: неизвестный инструмент {name!r}"

    async def _web_search(self, args: dict[str, Any]) -> str:
        assert self._search is not None
        query = str(args.get("query", "")).strip()
        if not query:
            return "ERROR: пустой запрос"
        hits = await self._search.search(query, scope=str(args.get("scope", "primary")))
        if not hits:
            return "Ничего не найдено. Переформулируй запрос или смени scope."
        return "\n".join(
            f"[{i}] ({h.tier}) {h.title} — {h.url}"
            + (f" [изменено {h.modtime}]" if h.modtime else "")
            + (f"\n    {h.snippet}" if h.snippet else "")
            for i, h in enumerate(hits, 1)
        )

    async def _fetch_url(self, args: dict[str, Any]) -> str:
        assert self._fetcher is not None
        url = str(args.get("url", "")).strip()
        try:
            offset = max(0, int(args.get("offset", 0) or 0))
        except (TypeError, ValueError):
            return "ERROR: offset должен быть целым числом"
        page = self._cache.get(url)
        if page is None:
            page = await self._fetcher.fetch(url)
            self._cache[url] = page
        end = min(len(page.text), offset + self._chunk)
        header = (
            f"URL: {page.final_url} | уровень: {page.tier} | получено: {page.fetched_at} | "
            f"заголовок: {page.title} | символы {offset}-{end} из {len(page.text)}"
        )
        tail = f"\n[продолжение: offset={end}]" if end < len(page.text) else ""
        return f"{header}\n{page.text[offset:end]}{tail}"
