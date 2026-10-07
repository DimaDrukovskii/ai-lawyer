"""`lawyer doctor`: первая проверка после получения ключей.

Главные риски нового провайдера — не качество текста, а то, что tool calling не работает
на выбранной модели, а поиск не отдаёт результатов. Ловим это за минуту, до дорогого прогона.
"""

from __future__ import annotations

from dataclasses import dataclass

from .deps import Deps
from .llm.base import ToolSpec


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


_PING_TOOL = ToolSpec(
    name="get_number",
    description="Возвращает число. Всегда вызывай этот инструмент, когда просят число.",
    parameters={"type": "object", "properties": {}, "required": []},
)


async def _guard(name: str, coro) -> Check:
    # Диагностика: здесь нужен широкий except, чтобы показать любую причину отказа
    try:
        return Check(name, *await coro)
    except Exception as exc:
        return Check(name, False, f"{type(exc).__name__}: {exc}"[:300])


async def _chat(deps: Deps) -> tuple[bool, str]:
    r = await deps.provider.chat(
        model=deps.settings.model_fast,
        system="Отвечай одним словом.",
        messages=[{"role": "user", "content": "Скажи: готово"}],
        max_tokens=20,
    )
    return bool(r.text.strip()), r.text.strip()[:80]


async def _tool_calling(deps: Deps) -> tuple[bool, str]:
    r = await deps.provider.chat(
        model=deps.settings.model_fast,
        system="Ты должен вызвать инструмент.",
        messages=[{"role": "user", "content": "Дай мне число через инструмент."}],
        tools=(_PING_TOOL,),
        max_tokens=100,
    )
    called = [c.name for c in r.tool_calls]
    return "get_number" in called, f"вызовы: {called or 'нет'}"


async def _search(deps: Deps) -> tuple[bool, str]:
    if deps.registry is None or not any(s.name == "web_search" for s in deps.registry.specs()):
        return (
            False,
            "поиск не настроен (нужны YANDEX_SEARCH_API_KEY/YANDEX_API_KEY и YANDEX_FOLDER_ID)",
        )
    out = await deps.registry.call(
        "web_search", {"query": "статья 346.21 НК РФ", "scope": "primary"}
    )
    return not out.startswith(("ERROR", "Ничего")), out.splitlines()[0][:160]


async def _fetch(deps: Deps) -> tuple[bool, str]:
    if deps.registry is None or not any(s.name == "fetch_url" for s in deps.registry.specs()):
        return False, "fetch_url недоступен"
    out = await deps.registry.call("fetch_url", {"url": "https://www.nalog.gov.ru/"})
    return not out.startswith("ERROR"), out.splitlines()[0][:160]


async def run_doctor(deps: Deps) -> list[Check]:
    return [
        await _guard("чат: провайдер отвечает", _chat(deps)),
        await _guard("tool calling: модель вызывает инструменты", _tool_calling(deps)),
        await _guard("поиск: Yandex Search API", _search(deps)),
        await _guard("fetch: загрузка nalog.gov.ru", _fetch(deps)),
    ]
