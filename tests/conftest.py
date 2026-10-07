from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from lawyer.config import Settings
from lawyer.deps import Deps
from lawyer.llm.mock import MockProvider

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings() -> Settings:
    return replace(Settings.from_env({}), root=ROOT, provider="mock")


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def deps(settings: Settings, mock_provider: MockProvider) -> Deps:
    return Deps(provider=mock_provider, settings=settings, registry=None)


@pytest.fixture
def demo_case(tmp_path: Path) -> Path:
    """Копия демо-кейса: пайплайн дописывает файлы в папку кейса, репозиторий не трогаем."""
    dst = tmp_path / "demo"
    shutil.copytree(ROOT / "cases" / "demo", dst)
    return dst
