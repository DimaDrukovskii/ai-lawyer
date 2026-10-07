"""GigaChat. ЭКСПЕРИМЕНТАЛЬНО: схема OAuth и базовые URL взяты из документации Сбера,
но адаптер не прогонялся на живом API. Формат tools у GigaChat совместим с OpenAI
только частично — первым делом запусти `python -m lawyer doctor`.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import httpx
from openai import AsyncOpenAI

from ..config import Settings
from .base import LLMError
from .openai_compat import OpenAICompatProvider

_REFRESH_MARGIN_S = 60.0


class GigaChatProvider(OpenAICompatProvider):
    """Токен доступа живёт ~30 минут, поэтому клиент пересобирается при истечении."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not settings.gigachat_auth_key:
            raise LLMError("для провайдера gigachat нужен GIGACHAT_AUTH_KEY")
        # Минцифры-сертификат нужен для ngw.devices.sberbank.ru; путь задаётся в .env
        verify: bool | str = settings.gigachat_ca_bundle or True
        shared = http_client or httpx.AsyncClient(verify=verify, timeout=120.0)
        super().__init__(api_key="", base_url=settings.gigachat_base_url, http_client=shared)
        self._s = settings
        self._shared_http = shared
        self._clock = clock
        self._token = ""
        self._expires_at = 0.0

    async def _refresh_token(self) -> None:
        resp = await self._shared_http.post(
            self._s.gigachat_oauth_url,
            headers={
                "Authorization": f"Basic {self._s.gigachat_auth_key}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"scope": self._s.gigachat_scope},
        )
        if resp.status_code != 200:
            raise LLMError(f"GigaChat OAuth: HTTP {resp.status_code}")
        body = resp.json()
        self._token = body["access_token"]
        # expires_at приходит в миллисекундах epoch
        self._expires_at = float(body["expires_at"]) / 1000.0

    async def _get_client(self) -> AsyncOpenAI:
        if not self._token or self._clock() >= self._expires_at - _REFRESH_MARGIN_S:
            await self._refresh_token()
            self._client = self._build_client(self._token)
        assert self._client is not None
        return self._client
