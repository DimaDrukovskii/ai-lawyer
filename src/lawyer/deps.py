from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .llm.base import LLMProvider
from .prompts import load_prompt
from .tools.registry import ToolRegistry


@dataclass(frozen=True)
class Deps:
    """Всё, что нужно агентам: провайдер, настройки, инструменты. Передаётся явно, без глобалов."""

    provider: LLMProvider
    settings: Settings
    registry: ToolRegistry | None = None

    def prompt(self, name: str, **variables: object) -> str:
        return load_prompt(self.settings.prompts_dir, name, **variables)
