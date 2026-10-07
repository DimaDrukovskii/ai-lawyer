from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lawyer.api import create_app
from lawyer.cli import main
from lawyer.config import Settings
from lawyer.deps import Deps
from lawyer.llm.mock import MockProvider
from lawyer.schemas import CriticReport
from lawyer.structured import parse_structured

ROOT = Path(__file__).resolve().parents[1]


class TestSettings:
    def test_defaults_and_overrides(self):
        s = Settings.from_env(
            {"LLM_PROVIDER": "GigaChat", "MAX_PARALLEL": "3", "ALLOW_UNKNOWN_DOMAINS": "1"}
        )
        assert s.provider == "gigachat" and s.max_parallel == 3 and s.allow_unknown_domains
        assert Settings.from_env({}).provider == "yandex"

    def test_search_key_falls_back_to_llm_key(self):
        assert Settings.from_env({"YANDEX_API_KEY": "a"}).search_key == "a"
        assert (
            Settings.from_env({"YANDEX_API_KEY": "a", "YANDEX_SEARCH_API_KEY": "b"}).search_key
            == "b"
        )


class TestStructured:
    async def test_invalid_json_is_repaired_once(self):
        provider = MockProvider({"repair": lambda s, m: '{"lens": "calendar", "complete": true}'})
        out = await parse_structured(provider, model="m", model_cls=CriticReport, text="мусор")
        assert out.complete and provider.calls == ["repair"]

    async def test_valid_json_makes_no_extra_call(self):
        provider = MockProvider()
        await parse_structured(provider, model="m", model_cls=CriticReport, text='{"lens": "x"}')
        assert provider.calls == []


class TestCli:
    def test_zones_lists_all(self, capsys):
        assert main(["zones"]) == 0
        out = capsys.readouterr().out
        assert out.count("\n") == 12 and "usn-income-recognition-marketplaces" in out

    def test_mock_research_then_review_use_repo_dirs(self, capsys, monkeypatch, tmp_path):
        # изолируем корень: прогон пишет в out/ и knowledge/, а не в рабочее дерево
        root = tmp_path / "repo"
        for name in ("prompts", "research", "rules", "skills", "cases"):
            shutil.copytree(ROOT / name, root / name)
        (root / "knowledge").mkdir()
        monkeypatch.setenv("LAWYER_HOME", str(root))

        assert (
            main(
                [
                    "--mock",
                    "research",
                    "--zones",
                    "usn-regime-and-limits",
                    "--tax-year",
                    "2026",
                    "--promote",
                ]
            )
            == 0
        )
        # фейковая база знаний изолирована от настоящей
        assert (root / "out" / "mock-knowledge" / "LATEST").exists()
        assert not (root / "knowledge" / "LATEST").exists()

        assert main(["--mock", "review", str(root / "cases" / "demo")]) == 0
        assert "chk-arith-h1" in (root / "out" / "review" / "demo_findings.json").read_text()
        # review в mock-режиме действительно прочитал мок-базу
        assert "База знаний: прогон" in (root / "out" / "review" / "demo_report.md").read_text()

    def test_unknown_zone_is_clean_error(self, capsys):
        assert main(["--mock", "research", "--zones", "nope"]) == 2
        assert "nope" in capsys.readouterr().err

    def test_missing_provider_keys_is_clean_error(self, capsys, monkeypatch):
        monkeypatch.delenv("YANDEX_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "yandex")
        assert main(["doctor"]) == 2
        assert "YANDEX_API_KEY" in capsys.readouterr().err


class TestApi:
    @pytest.fixture
    def client(self, tmp_path):
        s = replace(Settings.from_env({}), root=tmp_path, provider="mock")
        for name in ("prompts", "rules", "skills"):
            shutil.copytree(ROOT / name, tmp_path / name)
        return TestClient(create_app(Deps(provider=MockProvider(), settings=s)))

    PROFILE = json.dumps(
        {"entity_type": "ip", "usn_object": "income", "region": "Москва", "tax_year": 2025}
    )

    def test_health(self, client):
        assert client.get("/healthz").json() == {"status": "ok"}

    def test_review_roundtrip(self, client):
        files = [("files", ("declaration.txt", b"demo", "text/plain"))]
        r = client.post("/v1/reviews", data={"profile": self.PROFILE}, files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["id"] and "report_md" in body and isinstance(body["findings"], list)

    def test_rejects_bad_profile(self, client):
        files = [("files", ("a.txt", b"x", "text/plain"))]
        assert client.post("/v1/reviews", data={"profile": "{}"}, files=files).status_code == 422

    def test_rejects_unsupported_extension(self, client):
        files = [("files", ("evil.exe", b"MZ", "application/octet-stream"))]
        assert (
            client.post("/v1/reviews", data={"profile": self.PROFILE}, files=files).status_code
            == 415
        )

    def test_path_traversal_in_filename_is_neutralized(self, client, tmp_path):
        files = [("files", ("../../escape.txt", b"x", "text/plain"))]
        assert (
            client.post("/v1/reviews", data={"profile": self.PROFILE}, files=files).status_code
            == 200
        )
        assert not (tmp_path / "escape.txt").exists()
        # документы клиента и результаты не остаются на диске после ответа
        assert not any((tmp_path / "out" / "api_cases").glob("*"))
        assert not any((tmp_path / "out" / "review").glob("*"))
