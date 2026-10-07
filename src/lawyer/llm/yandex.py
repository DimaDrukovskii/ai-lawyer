from __future__ import annotations

import httpx

from ..config import Settings
from .base import LLMError
from .openai_compat import OpenAICompatProvider


class YandexProvider(OpenAICompatProvider):
    """Yandex AI Studio через OpenAI-совместимый API.

    Авторизация: Api-Key; каталог (folder) передаётся как OpenAI `project`.
    Модель задаётся URI вида gpt://<folder_id>/<model>; если в конфиге указан
    голый идентификатор (aliceai-llm), префикс добавляем сами.
    """

    def __init__(self, settings: Settings, *, http_client: httpx.AsyncClient | None = None) -> None:
        if not settings.yandex_api_key or not settings.yandex_folder_id:
            raise LLMError("для провайдера yandex нужны YANDEX_API_KEY и YANDEX_FOLDER_ID")
        super().__init__(
            api_key=settings.yandex_api_key,
            base_url=settings.yandex_base_url,
            project=settings.yandex_folder_id,
            http_client=http_client,
        )
        self._folder = settings.yandex_folder_id

    def model_id(self, model: str) -> str:
        return model if "://" in model else f"gpt://{self._folder}/{model}"
