from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class Rule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    value: Any = None
    unit: str = ""
    source: str = ""
    verified: bool = False
    verified_on: str = ""
    note: str = ""


class RuleBook:
    """Числа и сроки для детерминированных проверок. Источник правды — rules/usn.yaml."""

    def __init__(self, rules: dict[str, Rule]) -> None:
        self._rules = dict(rules)

    @classmethod
    def from_yaml(cls, path: Path) -> RuleBook:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls({k: Rule.model_validate(v) for k, v in (raw.get("rules") or {}).items()})

    def get(self, key: str) -> Rule | None:
        """Правило, если у него есть значение. Нет значения — проверка не может работать."""
        rule = self._rules.get(key)
        return rule if rule is not None and rule.value is not None else None

    def caveat(self, key: str) -> str:
        rule = self._rules.get(key)
        if rule is None or rule.verified:
            return ""
        return " (правило не подтверждено research-прогоном, сверь с базой знаний)"

    def unverified_keys(self) -> list[str]:
        return sorted(k for k, r in self._rules.items() if r.value is not None and not r.verified)
