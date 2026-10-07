from __future__ import annotations

from ..config import Settings
from .base import LLMError, LLMProvider, ThrottledProvider
from .gigachat import GigaChatProvider
from .mock import MockProvider
from .yandex import YandexProvider


def build_provider(settings: Settings) -> LLMProvider:
    """Собирает провайдера по LLM_PROVIDER и оборачивает в общий лимит параллелизма."""
    match settings.provider:
        case "yandex":
            inner: LLMProvider = YandexProvider(settings)
        case "gigachat":
            inner = GigaChatProvider(settings)
        case "mock":
            inner = MockProvider()
        case other:
            raise LLMError(f"неизвестный LLM_PROVIDER={other!r} (yandex | gigachat | mock)")
    return ThrottledProvider(inner, settings.max_parallel)
