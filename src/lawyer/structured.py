"""Надёжный разбор JSON из ответов моделей: парсинг → валидация → один проход ремонта."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .jsonutil import JsonParseError, extract_json
from .llm.base import LLMProvider

T = TypeVar("T", bound=BaseModel)

REPAIR_SYSTEM = (
    "<!-- role: repair -->\n"
    "Ты чинишь JSON. Верни ТОЛЬКО валидный JSON по схеме, без пояснений и без markdown. "
    "Ничего не выдумывай: используй только то, что есть во входном тексте."
)


def _validate(model_cls: type[T], text: str) -> T:
    return model_cls.model_validate(extract_json(text))


async def parse_structured(
    provider: LLMProvider, *, model: str, model_cls: type[T], text: str
) -> T:
    """Парсит ответ в модель. Если не вышло — один раз просит быструю модель починить JSON."""
    try:
        return _validate(model_cls, text)
    except (JsonParseError, ValidationError, ArithmeticError) as first:
        schema = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
        resp = await provider.chat(
            model=model,
            system=REPAIR_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Схема:\n{schema}\n\nОшибка разбора: {first}\n\nТекст:\n{text}",
                }
            ],
            max_tokens=6000,
            temperature=0.0,
        )
        return _validate(model_cls, resp.text)
