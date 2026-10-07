from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_LINE_START_OBJECT = re.compile(r"^\{", re.M)


class JsonParseError(ValueError):
    """В ответе модели нет валидного JSON."""


def _starts(chunk: str, opener: str) -> list[int]:
    """Где пробовать начать разбор: первая скобка и скобки в начале строки.

    Во вложенные объекты НЕ проваливаемся: у усечённого ответа (обрыв по max_tokens) внутри
    почти всегда есть валидный кусок, и принять его за весь ответ значит молча потерять данные.
    """
    first = chunk.find(opener)
    if first == -1:
        return []
    extra = [m.start() for m in _LINE_START_OBJECT.finditer(chunk)] if opener == "{" else []
    return sorted({first, *extra})


def extract_json(text: str) -> Any:
    """Достаёт JSON-объект (иначе массив) из ответа модели.

    Объекты приоритетнее массивов: фраза вроде «[1]» в прозе не должна приниматься за ответ.
    Code-fence и обрамляющая проза допускаются.
    """
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for opener in ("{", "["):
        for chunk in candidates:
            for start in _starts(chunk, opener):
                try:
                    obj, _ = decoder.raw_decode(chunk[start:])
                except json.JSONDecodeError:
                    continue
                return obj
    raise JsonParseError("в ответе нет валидного JSON (возможно, ответ обрезан)")
