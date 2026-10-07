from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


class JsonParseError(ValueError):
    """В ответе модели нет валидного JSON."""


def extract_json(text: str) -> Any:
    """Достаёт первый валидный JSON-объект (или массив) из ответа модели.

    Объекты приоритетнее массивов: фраза вроде «[1]» в прозе не должна
    приниматься за ответ. Code-fence и обрамляющая проза допускаются.
    """
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for opener in ("{", "["):
        for chunk in candidates:
            for m in re.finditer(re.escape(opener), chunk):
                try:
                    obj, _ = decoder.raw_decode(chunk[m.start() :])
                except json.JSONDecodeError:
                    continue
                return obj
    raise JsonParseError("в ответе нет валидного JSON")
