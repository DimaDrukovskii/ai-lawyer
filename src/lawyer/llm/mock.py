"""MockProvider: прогон всего пайплайна без ключей и сети.

Роль определяется маркером <!-- role: ... --> в системном промпте. Ответы шаблонные:
цель — проверить структуру, контракты и запись результатов, а не качество анализа.
Не использовать как источник фактов: ссылки в моках фиктивные.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .base import LLMResponse, ToolCall, ToolSpec

Handler = Callable[[str, Sequence[dict[str, Any]]], str]

_ROLE = re.compile(r"<!--\s*role:\s*(\w+)\s*-->")


def _last_user(messages: Sequence[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _field(text: str, name: str, default: str = "unknown") -> str:
    m = re.search(rf"^{name}:\s*(\S+)", text, re.M)
    return m.group(1) if m else default


def _researcher(_system: str, messages: Sequence[dict[str, Any]]) -> str:
    zone_id = _field(_last_user(messages), "zone_id")
    return json.dumps(
        {
            "zone_id": zone_id,
            "summary": f"[MOCK] сводка по зоне {zone_id}",
            "claims": [
                {
                    "id": "c1",
                    "text": "[MOCK] норма, действующая безусловно",
                    "kind": "law_norm",
                    "numbers": [],
                    "sources": [
                        {
                            # домен primary, чтобы пройти правило доверия; путь явно фиктивный
                            "url": "https://www.nalog.gov.ru/mock-do-not-use",
                            "title": "mock",
                            "doc_date": "2026-01-01",
                            "quote": "mock quote",
                            "tier": "primary",
                        }
                    ],
                },
                {
                    "id": "c2",
                    "text": "[MOCK] норма, зависящая от неизвестного факта",
                    "kind": "conditional",
                    "condition": "[MOCK] если у клиента есть сотрудники",
                    "sources": [],
                },
            ],
            "confidence": "low",
            "unverified": ["[MOCK] ничего не проверялось по-настоящему"],
        },
        ensure_ascii=False,
    )


def _verifier(_system: str, messages: Sequence[dict[str, Any]]) -> str:
    text = _last_user(messages)
    ids = re.findall(r'"id":\s*"([^"]+)"', text)
    return json.dumps(
        {
            "zone_id": _field(text, "zone_id"),
            "checks": [{"claim_id": cid, "verdict": "confirmed", "note": "[MOCK]"} for cid in ids],
        },
        ensure_ascii=False,
    )


def _critic(_system: str, messages: Sequence[dict[str, Any]]) -> str:
    return json.dumps(
        {"lens": _field(_last_user(messages), "lens"), "complete": True, "gaps": []},
        ensure_ascii=False,
    )


# Согласовано с cases/demo/docs/*_synthetic.txt (там заведомо заложены ошибки)
_EXTRACT = {
    "declaration": {
        "periods": {
            "q1": {
                "income": 1000000,
                "rate_percent": 6,
                "tax_calculated": 60000,
                "contributions_deducted": 30000,
            },
            "h1": {
                "income": 2500000,
                "rate_percent": 6,
                "tax_calculated": 125000,
                "contributions_deducted": 55000,
            },
            "m9": {
                "income": 3800000,
                "rate_percent": 6,
                "tax_calculated": 228000,
                "contributions_deducted": 75000,
            },
            "year": {
                "income": 5000000,
                "rate_percent": 6,
                "tax_calculated": 300000,
                "contributions_deducted": 100658,
            },
        },
        "tax_for_year": 199342,
        "filed_on": "2026-04-28",
        "form_version": "КНД 1152017 [MOCK]",
    },
    "kudir": {
        "income": {"q1": 1000000, "h1": 2500000, "m9": 3800000, "year": 5200000},
        "expenses": {},
    },
    "evidence": [],
    "unreadable": [],
}


def _extract(_system: str, _messages: Sequence[dict[str, Any]]) -> str:
    return json.dumps(_EXTRACT, ensure_ascii=False)


_DEFAULTS: dict[str, Handler] = {
    "intake": lambda s, m: json.dumps({"missing": []}),
    "researcher": _researcher,
    "verifier": _verifier,
    "critic": _critic,
    "extract": _extract,
    "judge": lambda s, m: json.dumps({"findings": []}),
    "report": lambda s, m: json.dumps(
        {
            "summary": "[MOCK] краткое резюме",
            "questions_for_accountant": ["[MOCK] вопрос бухгалтеру"],
            "next_steps": ["[MOCK] следующий шаг"],
        },
        ensure_ascii=False,
    ),
    "repair": lambda s, m: "{}",
}


class MockProvider:
    def __init__(self, overrides: Mapping[str, Handler] | None = None) -> None:
        self._handlers = {**_DEFAULTS, **(overrides or {})}
        self.calls: list[str] = []  # роли в порядке вызова — удобно для тестов

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
        m = _ROLE.search(system)
        role = m.group(1) if m else "unknown"
        self.calls.append(role)
        if role == "verifier" and tools and not any(m.get("role") == "tool" for m in messages):
            # как настоящий верификатор: сначала открыть все процитированные источники
            urls = list(dict.fromkeys(re.findall(r'"url":\s*"([^"]+)"', _last_user(messages))))
            calls = tuple(ToolCall(f"mock{i}", "fetch_url", {"url": u}) for i, u in enumerate(urls))
            if calls:
                return LLMResponse(
                    text="",
                    tool_calls=calls,
                    assistant_message={"role": "assistant", "content": ""},
                )
        handler = self._handlers.get(role)
        if handler is None:
            raise KeyError(f"MockProvider: нет обработчика для роли {role!r}")
        text = handler(system, messages)
        return LLMResponse(text=text, assistant_message={"role": "assistant", "content": text})
