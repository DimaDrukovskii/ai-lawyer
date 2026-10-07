"""Сборка зависимостей из настроек. Единая точка для CLI, API и тестов."""

from __future__ import annotations

from dataclasses import replace

from .config import Settings
from .deps import Deps
from .llm import build_provider
from .tools import Fetcher, ToolRegistry, YandexSearch


def build_registry(settings: Settings) -> ToolRegistry | None:
    """Инструменты исследователей. В режиме mock сети нет вовсе."""
    if settings.provider == "mock":
        return None
    search = None
    if settings.search_key and settings.yandex_folder_id:
        search = YandexSearch(api_key=settings.search_key, folder_id=settings.yandex_folder_id)
    return ToolRegistry(
        search=search, fetcher=Fetcher(allow_unknown=settings.allow_unknown_domains)
    )


def build_deps(settings: Settings) -> Deps:
    return Deps(
        provider=build_provider(settings), settings=settings, registry=build_registry(settings)
    )


def with_provider(settings: Settings, provider: str) -> Settings:
    return replace(settings, provider=provider)
