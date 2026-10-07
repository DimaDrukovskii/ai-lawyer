from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=64)
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render(template: str, **variables: object) -> str:
    """Подстановка {{name}}. Не str.format: в промптах полно JSON-фигурных скобок."""
    out = template
    for key, value in variables.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def load_prompt(prompts_dir: Path, name: str, **variables: object) -> str:
    """base.md + <name>.md. Роль задаёт маркер <!-- role: ... --> в начале role-файла."""
    base = _read(prompts_dir / "base.md")
    role = _read(prompts_dir / f"{name}.md")
    return render(f"{base}\n\n{role}", **variables)
