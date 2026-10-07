"""Один агент = цикл «модель → вызовы инструментов → модель», пока не будет финального текста."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm.base import LLMProvider, Usage
from .tools.registry import ToolRegistry

MAX_TOOL_OUTPUT_CHARS = 14_000
_FORCE_FINAL = (
    "Лимит шагов исчерпан. Больше инструментов не будет. Верни итог СЕЙЧАС в требуемом "
    "формате по тому, что уже собрано; чего не успел подтвердить — перечисли как неподтверждённое."
)


@dataclass(frozen=True)
class ToolLogEntry:
    name: str
    arguments: dict[str, Any]
    ok: bool


@dataclass(frozen=True)
class AgentResult:
    text: str
    steps: int
    tool_log: tuple[ToolLogEntry, ...]
    usage: Usage
    hit_step_limit: bool = False


async def run_agent(
    provider: LLMProvider,
    *,
    model: str,
    system: str,
    user: str,
    registry: ToolRegistry | None = None,
    max_steps: int = 8,
    max_tokens: int = 4000,
) -> AgentResult:
    tools = registry.specs() if registry is not None else ()
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    log: list[ToolLogEntry] = []
    usage = Usage()

    for step in range(1, max_steps + 1):
        resp = await provider.chat(
            model=model, system=system, messages=messages, tools=tools, max_tokens=max_tokens
        )
        usage = usage + resp.usage
        if not resp.tool_calls or registry is None:
            return AgentResult(resp.text, step, tuple(log), usage)

        messages = [*messages, resp.assistant_message]
        for call in resp.tool_calls:
            output = await registry.call(call.name, call.arguments)
            log.append(ToolLogEntry(call.name, call.arguments, ok=not output.startswith("ERROR:")))
            messages = [
                *messages,
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output[:MAX_TOOL_OUTPUT_CHARS],
                },
            ]

    final = await provider.chat(
        model=model,
        system=system,
        messages=[*messages, {"role": "user", "content": _FORCE_FINAL}],
        tools=(),
        max_tokens=max_tokens,
    )
    return AgentResult(final.text, max_steps + 1, tuple(log), usage + final.usage, True)
