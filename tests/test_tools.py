from __future__ import annotations

import base64

import httpx
import pytest

from lawyer.schemas import Tier
from lawyer.tools import Fetcher, SearchError, ToolRegistry, YandexSearch
from lawyer.tools.search import MAX_QUERY_CHARS, build_query, parse_results

# Сконструирован по описанию формата XML-ответа Yandex Search API (не снимок реального ответа)
XML_OK = """<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0"><response><results><grouping>
<group><doc>
  <url>https://www.nalog.gov.ru/rn77/taxation/usn/</url>
  <title>Упрощённая <hlword>система</hlword> налогообложения</title>
  <modtime>20260110T120000</modtime>
  <passages><passage>Налогоплательщики <hlword>УСН</hlword></passage><passage>вторая выдержка</passage></passages>
</doc></group>
<group><doc>
  <url>https://random-blog.example/usn</url>
  <title>Блог</title>
</doc></group>
</grouping></results></response></yandexsearch>"""

XML_ERR = """<?xml version="1.0"?><yandexsearch><response><error code="15">Искомая комбинация не найдена</error></response></yandexsearch>"""


class TestSearch:
    def test_parse_results_strips_markup(self):
        hits = parse_results(XML_OK.encode())
        assert [h.url for h in hits][0].startswith("https://www.nalog.gov.ru")
        assert hits[0].title == "Упрощённая система налогообложения"
        assert hits[0].snippet == "Налогоплательщики УСН … вторая выдержка"
        assert hits[0].tier is Tier.PRIMARY
        assert hits[0].modtime == "20260110T120000"

    def test_parse_error_element(self):
        with pytest.raises(SearchError, match="15"):
            parse_results(XML_ERR.encode())

    def test_parse_rejects_entity_bomb(self):
        bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>'
        with pytest.raises(SearchError):
            parse_results(bomb)

    def test_build_query_respects_limit_and_keeps_sites(self):
        q = build_query("УСН доходы селлер маркетплейс", "primary")
        assert q.startswith("(site:nalog.gov.ru")
        assert len(q) <= MAX_QUERY_CHARS

    def test_build_query_long_text_still_within_limit(self):
        q = build_query("слово " * 200, "primary")
        assert len(q) <= MAX_QUERY_CHARS

    def test_build_query_any_scope_has_no_site_filter(self):
        assert build_query("ставка УСН", "any") == "ставка УСН"

    async def test_search_posts_and_postfilters_by_scope(self):
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = request.read().decode()
            return httpx.Response(200, json={"rawData": base64.b64encode(XML_OK.encode()).decode()})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        search = YandexSearch(api_key="k", folder_id="f", http_client=client)
        hits = await search.search("УСН", scope="primary")

        assert seen["auth"] == "Api-Key k"
        assert '"folderId":"f"' in str(seen["body"]).replace(" ", "")
        # блог отфильтрован пост-фильтром по tier, даже если поиск его вернул
        assert [h.tier for h in hits] == [Tier.PRIMARY]

    async def test_search_http_error_becomes_search_error(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
        search = YandexSearch(api_key="k", folder_id="f", http_client=client)
        with pytest.raises(SearchError):
            await search.search("УСН")


def _fetcher(handler, *, allow_unknown: bool = False) -> Fetcher:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    return Fetcher(allow_unknown=allow_unknown, http_client=client)


HTML = "<html><head><title>НК РФ</title></head><body><nav>меню</nav><script>x()</script><p>Статья 346.21</p></body></html>"


class TestFetch:
    async def test_html_to_text_strips_noise(self):
        f = _fetcher(
            lambda r: httpx.Response(200, text=HTML, headers={"content-type": "text/html"})
        )
        page = await f.fetch("https://www.consultant.ru/doc")
        assert page.title == "НК РФ"
        assert "Статья 346.21" in page.text
        assert "меню" not in page.text and "x()" not in page.text
        assert page.tier is Tier.OFFICIAL_TEXT
        assert page.fetched_at.endswith("+00:00")

    async def test_blocked_domain_never_requested(self):
        called = []
        f = _fetcher(lambda r: called.append(r) or httpx.Response(200, text="x"))
        from lawyer.tools import FetchError

        with pytest.raises(FetchError, match="белом списке"):
            await f.fetch("https://blog.example/usn")
        assert called == []

    async def test_redirect_to_unlisted_domain_is_rejected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "www.nalog.gov.ru":
                return httpx.Response(302, headers={"location": "https://evil.example/x"})
            return httpx.Response(200, text="evil")

        from lawyer.tools import FetchError

        with pytest.raises(FetchError, match="редирект"):
            await _fetcher(handler).fetch("https://www.nalog.gov.ru/go")


class TestRegistry:
    async def test_fetch_is_chunked_with_continuation_hint(self):
        text = "<html><body><p>" + "а" * 50 + "</p></body></html>"
        f = _fetcher(lambda r: httpx.Response(200, text=text))
        reg = ToolRegistry(fetcher=f, chunk_chars=20)

        first = await reg.call("fetch_url", {"url": "https://www.nalog.gov.ru/x"})
        assert "символы 0-20 из 50" in first and "offset=20" in first
        last = await reg.call("fetch_url", {"url": "https://www.nalog.gov.ru/x", "offset": 40})
        assert "символы 40-50 из 50" in last and "продолжение" not in last

    async def test_errors_are_returned_not_raised(self):
        f = _fetcher(lambda r: httpx.Response(404))
        reg = ToolRegistry(fetcher=f)
        out = await reg.call("fetch_url", {"url": "https://www.nalog.gov.ru/missing"})
        assert out.startswith("ERROR:")
        assert (await reg.call("no_such_tool", {})).startswith("ERROR:")
        assert (await reg.call("fetch_url", {"_error": "bad json"})).startswith("ERROR:")

    async def test_specs_reflect_available_tools(self):
        assert ToolRegistry().specs() == ()
        assert [
            s.name for s in ToolRegistry(fetcher=_fetcher(lambda r: httpx.Response(200))).specs()
        ] == ["fetch_url"]
