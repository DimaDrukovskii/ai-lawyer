from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from lawyer.agent import run_agent
from lawyer.config import Settings
from lawyer.llm.base import LLMError, LLMResponse, ThrottledProvider, ToolCall, ToolSpec
from lawyer.llm.factory import build_provider
from lawyer.llm.gigachat import GigaChatProvider
from lawyer.llm.mock import MockProvider
from lawyer.llm.yandex import YandexProvider
from lawyer.tools import ToolRegistry


def _completion(content: str = "", tool_calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }  # fmt: skip


def _yandex_settings() -> Settings:
    return replace(Settings.from_env({}), yandex_api_key="key", yandex_folder_id="b1gfolder")


class TestYandexProvider:
    async def test_model_uri_auth_and_tool_parsing(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.read())
            call = {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "УСН"}'},
            }
            return httpx.Response(200, json=_completion("", [call]))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = YandexProvider(_yandex_settings(), http_client=client)
        tool = ToolSpec("web_search", "поиск", {"type": "object", "properties": {}})
        resp = await provider.chat(
            model="aliceai-llm",
            system="sys",
            messages=[{"role": "user", "content": "q"}],
            tools=[tool],
        )

        assert seen["url"].endswith("/chat/completions")
        assert (
            seen["auth"] == "Bearer key"
        )  # SDK кладёт ключ в Bearer; проверь в doctor на живом API
        assert seen["body"]["model"] == "gpt://b1gfolder/aliceai-llm"
        assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
        assert seen["body"]["tools"][0]["function"]["name"] == "web_search"
        assert resp.tool_calls == (ToolCall("call_1", "web_search", {"query": "УСН"}),)
        assert resp.usage.input_tokens == 11 and resp.usage.output_tokens == 7
        assert resp.assistant_message["tool_calls"][0]["id"] == "call_1"

    async def test_full_uri_is_not_double_prefixed(self):
        provider = YandexProvider(_yandex_settings())
        assert provider.model_id("gpt://other/yandexgpt/latest") == "gpt://other/yandexgpt/latest"

    async def test_bad_tool_arguments_do_not_crash(self):
        def handler(request: httpx.Request) -> httpx.Response:
            call = {
                "id": "c",
                "type": "function",
                "function": {"name": "fetch_url", "arguments": "{oops"},
            }
            return httpx.Response(200, json=_completion("", [call]))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        resp = await YandexProvider(_yandex_settings(), http_client=client).chat(
            model="m", system="s", messages=[{"role": "user", "content": "q"}]
        )
        assert "_error" in resp.tool_calls[0].arguments

    async def test_http_error_becomes_llm_error(self):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(400, json={"error": {"message": "bad"}})
            )
        )
        provider = YandexProvider(_yandex_settings(), http_client=client)
        with pytest.raises(LLMError):
            await provider.chat(model="m", system="s", messages=[{"role": "user", "content": "q"}])

    def test_requires_credentials(self):
        with pytest.raises(LLMError, match="YANDEX_API_KEY"):
            YandexProvider(Settings.from_env({}))


class TestGigaChatProvider:
    async def test_token_is_fetched_cached_and_refreshed(self):
        oauth_calls: list[httpx.Request] = []
        api_auth: list[str] = []
        now = [1_000.0]

        def handler(request: httpx.Request) -> httpx.Response:
            if "oauth" in request.url.path:
                oauth_calls.append(request)
                expires_ms = int((now[0] + 1800) * 1000)
                return httpx.Response(
                    200, json={"access_token": f"tok{len(oauth_calls)}", "expires_at": expires_ms}
                )
            api_auth.append(request.headers["Authorization"])
            return httpx.Response(200, json=_completion("ok"))

        settings = replace(Settings.from_env({}), gigachat_auth_key="BASICKEY")
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = GigaChatProvider(settings, http_client=client, clock=lambda: now[0])
        msgs = [{"role": "user", "content": "q"}]

        await provider.chat(model="GigaChat-2-Max", system="s", messages=msgs)
        await provider.chat(model="GigaChat-2-Max", system="s", messages=msgs)
        assert len(oauth_calls) == 1  # токен переиспользован
        assert oauth_calls[0].headers["Authorization"] == "Basic BASICKEY"
        assert "RqUID" in oauth_calls[0].headers

        now[0] += 1800  # токен истёк
        await provider.chat(model="GigaChat-2-Max", system="s", messages=msgs)
        assert len(oauth_calls) == 2
        assert api_auth == ["Bearer tok1", "Bearer tok1", "Bearer tok2"]

    async def test_oauth_failure_is_llm_error(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
        settings = replace(Settings.from_env({}), gigachat_auth_key="K")
        with pytest.raises(LLMError, match="401"):
            await GigaChatProvider(settings, http_client=client).chat(
                model="m", system="s", messages=[{"role": "user", "content": "q"}]
            )


class TestFactoryAndThrottle:
    def test_factory_unknown_provider(self):
        with pytest.raises(LLMError, match="неизвестный"):
            build_provider(replace(Settings.from_env({}), provider="openai"))

    async def test_factory_mock_works(self):
        provider = build_provider(replace(Settings.from_env({}), provider="mock"))
        r = await provider.chat(model="m", system="<!-- role: intake -->", messages=[])
        assert json.loads(r.text) == {"missing": []}

    async def test_throttle_limits_concurrency(self):
        import asyncio

        active = peak = 0

        class Slow:
            async def chat(self, **_kw) -> LLMResponse:
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                return LLMResponse(text="x")

        throttled = ThrottledProvider(Slow(), max_parallel=2)
        await asyncio.gather(
            *(throttled.chat(model="m", system="s", messages=[]) for _ in range(8))
        )
        assert peak == 2


class ScriptedProvider:
    """Отдаёт заранее заданные ответы по порядку; запоминает, что ей прислали."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    async def chat(self, **kwargs) -> LLMResponse:
        self.requests.append(kwargs)
        return self._responses.pop(0)


class TestAgentLoop:
    async def test_tool_roundtrip_then_final(self):
        call = ToolCall("c1", "fetch_url", {"url": "https://nalog.gov.ru/"})
        provider = ScriptedProvider(
            [
                LLMResponse(
                    text="",
                    tool_calls=(call,),
                    assistant_message={"role": "assistant", "content": "", "tool_calls": []},
                ),
                LLMResponse(text="итог"),
            ]
        )

        class FakeRegistry(ToolRegistry):
            def specs(self):
                return (ToolSpec("fetch_url", "d", {"type": "object", "properties": {}}),)

            async def call(self, name, args):
                return "СТРАНИЦА"

        result = await run_agent(provider, model="m", system="s", user="u", registry=FakeRegistry())
        assert result.text == "итог" and result.steps == 2
        assert [t.name for t in result.tool_log] == ["fetch_url"]
        second_request_messages = provider.requests[1]["messages"]
        assert second_request_messages[-1] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "СТРАНИЦА",
        }

    async def test_step_limit_forces_final_answer_without_tools(self):
        call = ToolCall("c", "fetch_url", {})
        looping = LLMResponse(
            text="", tool_calls=(call,), assistant_message={"role": "assistant", "content": ""}
        )
        provider = ScriptedProvider([looping, looping, LLMResponse(text="принудительный итог")])

        class FakeRegistry(ToolRegistry):
            def specs(self):
                return (ToolSpec("fetch_url", "d", {"type": "object", "properties": {}}),)

            async def call(self, name, args):
                return "ERROR: нет"

        result = await run_agent(
            provider, model="m", system="s", user="u", registry=FakeRegistry(), max_steps=2
        )
        assert result.hit_step_limit and result.text == "принудительный итог"
        assert provider.requests[-1]["tools"] == ()
        assert not result.tool_log[0].ok


async def test_mock_provider_requires_known_role():
    with pytest.raises(KeyError):
        await MockProvider().chat(model="m", system="без маркера роли", messages=[])
