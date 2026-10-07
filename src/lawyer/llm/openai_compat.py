from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from .base import LLMError, LLMResponse, ToolCall, ToolSpec, Usage


def _loads_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_error": f"невалидный JSON в аргументах инструмента: {raw[:200]}"}
    return value if isinstance(value, dict) else {"_error": "аргументы инструмента не объект"}


def _tool_payload(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


class OpenAICompatProvider:
    """Любой бэкенд с OpenAI-совместимым /chat/completions + function calling."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        project: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 4,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._project = project
        self._http_client = http_client
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: AsyncOpenAI | None = None

    # --- точки расширения для конкретных провайдеров

    def model_id(self, model: str) -> str:
        return model

    async def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = self._build_client(self._api_key)
        return self._client

    def _build_client(self, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=self._base_url,
            project=self._project,
            http_client=self._http_client,
            max_retries=self._max_retries,
            timeout=self._timeout,
        )

    # --- контракт LLMProvider

    async def chat(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> LLMResponse:
        client = await self._get_client()
        extra: dict[str, Any] = {"tools": _tool_payload(tools)} if tools else {}
        try:
            resp = await client.chat.completions.create(
                model=self.model_id(model),
                messages=[{"role": "system", "content": system}, *messages],
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        if not resp.choices:
            raise LLMError("провайдер вернул пустой список choices")
        choice = resp.choices[0]
        msg = choice.message
        calls = tuple(
            ToolCall(id=tc.id, name=tc.function.name, arguments=_loads_args(tc.function.arguments))
            for tc in (msg.tool_calls or [])
            if getattr(tc, "type", None) in (None, "function")  # часть бэкендов не шлёт type
        )
        if choice.finish_reason == "length" and not calls:
            # Обрезанный JSON хуже явной ошибки: он молча превращается в «ничего не найдено».
            raise LLMError(
                f"ответ обрезан по max_tokens={max_tokens}: увеличь MAX_OUTPUT_TOKENS или сузь задачу"
            )
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in calls
            ]
        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return LLMResponse(
            text=msg.content or "", tool_calls=calls, usage=usage, assistant_message=assistant
        )
