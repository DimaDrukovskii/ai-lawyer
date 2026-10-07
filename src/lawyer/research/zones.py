from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class Zone(BaseModel):
    """Зона исследования: одно направление, один исследователь, один верификатор."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    title: str
    question: str
    must_cover: list[str] = []
    scope: Literal["primary", "marketplace", "any"] = "primary"
    tags: list[str] = []  # зоны, которые review подтягивает в judge


@dataclass(frozen=True)
class ResearchContext:
    tax_year: int
    as_of: date
    audience: str = "селлеры маркетплейсов (WB, Ozon, Яндекс Маркет) на УСН: ИП и ООО"


def load_zones(path: Path) -> list[Zone]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Zone.model_validate(item) for item in raw["zones"]]


def select_zones(zones: list[Zone], ids: list[str] | None) -> list[Zone]:
    if not ids:
        return zones
    by_id = {z.id: z for z in zones}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise KeyError(f"неизвестные зоны: {', '.join(unknown)}")
    return [by_id[i] for i in ids]


def safe_id(raw: str) -> str:
    """Идентификатор, безопасный для имени файла (id пришёл от LLM)."""
    return re.sub(r"[^a-z0-9_-]+", "-", raw.lower()).strip("-") or "x"
