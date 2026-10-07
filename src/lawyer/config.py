from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _default_root() -> Path:
    home = os.getenv("LAWYER_HOME")
    return Path(home) if home else Path(__file__).resolve().parents[2]


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    return int(raw) if raw else default


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Все настройки процесса. Иммутабельны: меняем через dataclasses.replace."""

    root: Path
    provider: str = "yandex"
    model_strong: str = "aliceai-llm"
    model_fast: str = "aliceai-llm"

    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_base_url: str = "https://ai.api.cloud.yandex.net/v1"
    yandex_search_api_key: str = ""

    gigachat_auth_key: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_ca_bundle: str = ""

    max_parallel: int = 6
    max_tool_steps: int = 8
    max_claims_per_zone: int = 12
    allow_unknown_domains: bool = False

    @property
    def search_key(self) -> str:
        return self.yandex_search_api_key or self.yandex_api_key

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        return cls(
            root=_default_root(),
            provider=e.get("LLM_PROVIDER", "yandex").strip().lower() or "yandex",
            model_strong=e.get("MODEL_STRONG", "").strip() or "aliceai-llm",
            model_fast=e.get("MODEL_FAST", "").strip() or "aliceai-llm",
            yandex_api_key=e.get("YANDEX_API_KEY", "").strip(),
            yandex_folder_id=e.get("YANDEX_FOLDER_ID", "").strip(),
            yandex_base_url=e.get("YANDEX_BASE_URL", "").strip()
            or "https://ai.api.cloud.yandex.net/v1",
            yandex_search_api_key=e.get("YANDEX_SEARCH_API_KEY", "").strip(),
            gigachat_auth_key=e.get("GIGACHAT_AUTH_KEY", "").strip(),
            gigachat_scope=e.get("GIGACHAT_SCOPE", "").strip() or "GIGACHAT_API_PERS",
            gigachat_ca_bundle=e.get("GIGACHAT_CA_BUNDLE", "").strip(),
            max_parallel=_int(e, "MAX_PARALLEL", 6),
            max_tool_steps=_int(e, "MAX_TOOL_STEPS", 8),
            max_claims_per_zone=_int(e, "MAX_CLAIMS_PER_ZONE", 12),
            allow_unknown_domains=_flag(e, "ALLOW_UNKNOWN_DOMAINS"),
        )
