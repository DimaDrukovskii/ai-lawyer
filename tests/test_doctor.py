from __future__ import annotations

import httpx

from lawyer.deps import Deps
from lawyer.doctor import run_doctor
from lawyer.llm.base import LLMError, LLMResponse, ToolCall
from lawyer.schemas import Tier
from lawyer.tools import Fetcher, ToolRegistry
from lawyer.tools.search import SearchHit


class FakeSearch:
    async def search(self, query, *, scope="primary", limit=8):
        return [
            SearchHit("https://www.consultant.ru/x", "НК РФ ст. 346.21", "…", Tier.OFFICIAL_TEXT)
        ]


class WorkingProvider:
    async def chat(self, *, tools=(), **_kw) -> LLMResponse:
        if tools:
            return LLMResponse(text="", tool_calls=(ToolCall("c", "get_number", {}),))
        return LLMResponse(text="готово")


class NoToolsProvider:
    """Модель, которая игнорирует инструменты: типичный отказ нового провайдера."""

    async def chat(self, **_kw) -> LLMResponse:
        return LLMResponse(text="Число — 42")


class BrokenProvider:
    async def chat(self, **_kw) -> LLMResponse:
        raise LLMError("401 Unauthorized")


def registry() -> ToolRegistry:
    page = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text="<html><title>ФНС</title><p>ok</p></html>")
        )
    )
    return ToolRegistry(search=FakeSearch(), fetcher=Fetcher(http_client=page))


async def test_all_green(settings):
    checks = await run_doctor(Deps(WorkingProvider(), settings, registry()))
    assert [c.ok for c in checks] == [True, True, True, True]


async def test_model_without_tool_calling_is_caught(settings):
    checks = {c.name: c for c in await run_doctor(Deps(NoToolsProvider(), settings, registry()))}
    tool_check = next(c for n, c in checks.items() if n.startswith("tool calling"))
    assert not tool_check.ok and "нет" in tool_check.detail


async def test_broken_provider_reports_not_raises(settings):
    checks = await run_doctor(Deps(BrokenProvider(), settings, registry()))
    assert not checks[0].ok and "401" in checks[0].detail


async def test_missing_search_is_reported_with_hint(settings):
    checks = await run_doctor(Deps(WorkingProvider(), settings, None))
    search = next(c for c in checks if c.name.startswith("поиск"))
    assert not search.ok and "YANDEX_FOLDER_ID" in search.detail
