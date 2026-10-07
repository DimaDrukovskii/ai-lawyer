"""Контракт провайдера LLM. Всё остальное ядро знает только этот интерфейс.

Сообщения — в формате OpenAI chat (dict с role/content/tool_calls): оба российских
провайдера (Yandex AI Studio, GigaChat) говорят на нём или близком диалекте.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    assistant_message: dict[str, Any] = field(default_factory=dict)  # для истории диалога


class LLMProvider(Protocol):
    async def chat(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Провайдер вернул ошибку, которую ретраи не лечат."""


class ThrottledProvider:
    """Ограничивает число параллельных запросов к провайдеру (общий семафор на процесс)."""

    def __init__(self, inner: LLMProvider, max_parallel: int) -> None:
        self._inner = inner
        self._sem = asyncio.Semaphore(max(1, max_parallel))

    async def chat(self, **kwargs: Any) -> LLMResponse:
        async with self._sem:
            return await self._inner.chat(**kwargs)
