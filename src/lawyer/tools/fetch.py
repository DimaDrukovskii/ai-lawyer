from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from ..schemas import Tier
from .domains import classify, is_fetch_allowed

MAX_BYTES = 8_000_000
USER_AGENT = "ai-lawyer/0.1 (+research; tax second-opinion)"
_NOISE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Page:
    url: str
    final_url: str
    title: str
    text: str
    tier: Tier
    fetched_at: str  # ISO-8601 UTC: «дата актуальности» для каждого утверждения
    content_hash: str


def html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    return title, soup.get_text("\n", strip=True)


def pdf_to_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pypdf бросает разнородные исключения на битых файлах
        raise FetchError(f"не удалось прочитать PDF: {exc}") from exc


class Fetcher:
    def __init__(
        self, *, allow_unknown: bool = False, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self._allow_unknown = allow_unknown
        self._http = http_client or httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )

    async def fetch(self, url: str) -> Page:
        if not is_fetch_allowed(url, allow_unknown=self._allow_unknown):
            raise FetchError(
                f"домен не в белом списке источников: {url}. Ищи на первоисточнике "
                "(nalog.gov.ru, minfin.gov.ru, pravo.gov.ru, consultant.ru, garant.ru и т.п.)"
            )
        try:
            resp = await self._http.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise FetchError(f"не удалось загрузить {url}: {exc}") from exc

        final_url = str(resp.url)
        # редирект мог увести на домен вне белого списка
        if not is_fetch_allowed(final_url, allow_unknown=self._allow_unknown):
            raise FetchError(f"редирект на домен вне белого списка: {final_url}")
        if len(resp.content) > MAX_BYTES:
            raise FetchError(f"ответ больше {MAX_BYTES // 1_000_000} МБ: {final_url}")

        ctype = resp.headers.get("content-type", "").lower()
        if "pdf" in ctype or final_url.lower().endswith(".pdf"):
            title, text = "", pdf_to_text(resp.content)
        else:
            title, text = html_to_text(resp.text)
        return Page(
            url=url,
            final_url=final_url,
            title=title,
            text=text,
            tier=classify(final_url),
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        )
