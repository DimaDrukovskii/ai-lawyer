from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from ..schemas import Tier
from .domains import classify, is_fetch_allowed

MAX_BYTES = 8_000_000
MAX_REDIRECTS = 5
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
        # редиректы ведём сами, чтобы проверять белый список ДО каждого запроса
        self._http = http_client or httpx.AsyncClient(
            timeout=30.0, headers={"User-Agent": USER_AGENT}
        )

    async def _get(self, url: str) -> tuple[str, str, bytes, str]:
        """GET с ручным обходом редиректов и потоковым лимитом размера.

        Каждый хоп (включая первый) проверяется по белому списку до отправки запроса:
        открытый редирект на разрешённом хосте не должен привести к запросу на чужой адрес.
        Лимит считается по распакованным данным, поэтому gzip-бомба упирается в него.
        """
        current = url
        for hop in range(MAX_REDIRECTS + 1):
            if not is_fetch_allowed(current, allow_unknown=self._allow_unknown):
                what = "редирект на домен вне белого списка" if hop else "домен не в белом списке"
                raise FetchError(
                    f"{what}: {current}. Ищи на первоисточнике "
                    "(nalog.gov.ru, minfin.gov.ru, pravo.gov.ru, consultant.ru и т.п.)"
                )
            try:
                async with self._http.stream("GET", current, follow_redirects=False) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location", "")
                        if not location:
                            raise FetchError(f"редирект без Location: {current}")
                        current = urljoin(current, location)
                        continue
                    resp.raise_for_status()
                    body = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body += chunk
                        if len(body) > MAX_BYTES:
                            raise FetchError(f"ответ больше {MAX_BYTES // 1_000_000} МБ: {current}")
                    ctype = resp.headers.get("content-type", "").lower()
                    return current, ctype, bytes(body), resp.encoding or "utf-8"
            except (httpx.HTTPError, httpx.InvalidURL) as exc:
                raise FetchError(f"не удалось загрузить {current}: {exc}") from exc
        raise FetchError(f"больше {MAX_REDIRECTS} редиректов подряд: {url}")

    async def fetch(self, url: str) -> Page:
        final_url, ctype, body, encoding = await self._get(url)
        if "pdf" in ctype or final_url.lower().endswith(".pdf"):
            title, text = "", pdf_to_text(body)
        else:
            title, text = html_to_text(body.decode(encoding, errors="replace"))
        return Page(
            url=url,
            final_url=final_url,
            title=title,
            text=text,
            tier=classify(final_url),
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        )
