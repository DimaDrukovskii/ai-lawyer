from __future__ import annotations

import pytest

from lawyer.jsonutil import JsonParseError, extract_json
from lawyer.schemas import Tier
from lawyer.tools.domains import classify, is_fetch_allowed


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_code_fence_and_prose(self):
        text = 'Вот результат:\n```json\n{"complete": true, "gaps": []}\n```\nГотово.'
        assert extract_json(text) == {"complete": True, "gaps": []}

    def test_object_preferred_over_bracket_in_prose(self):
        assert extract_json('см. [1] и {"ok": true}') == {"ok": True}

    def test_array_when_no_object(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_nested_braces_inside_strings(self):
        assert extract_json('{"t": "a {b} c"}') == {"t": "a {b} c"}

    def test_raises_without_json(self):
        with pytest.raises(JsonParseError):
            extract_json("никакого json здесь нет")


class TestClassify:
    @pytest.mark.parametrize(
        ("url", "tier"),
        [
            ("https://www.nalog.gov.ru/rn77/", Tier.PRIMARY),
            ("https://minfin.gov.ru/ru/document/?id=1", Tier.PRIMARY),
            ("https://kad.arbitr.ru/Card/1", Tier.PRIMARY),
            ("https://www.consultant.ru/document/cons_doc_LAW_28165/", Tier.OFFICIAL_TEXT),
            ("https://base.garant.ru/10900200/", Tier.OFFICIAL_TEXT),
            ("https://seller.ozon.ru/docs", Tier.MARKETPLACE),
            ("https://yandex.ru/legal/marketplace_oferta/", Tier.MARKETPLACE),
            ("https://www.klerk.ru/blogs/x", Tier.LEAD),
            ("https://random-blog.example/usn", Tier.UNKNOWN),
        ],
    )
    def test_tiers(self, url, tier):
        assert classify(url) is tier

    def test_lookalike_host_is_not_trusted(self):
        # nalog.gov.ru.evil.com и evilnalog.gov.ru не должны проходить как первоисточник
        assert classify("https://nalog.gov.ru.evil.com/x") is Tier.UNKNOWN
        assert classify("https://evilnalog.gov.ru/x") is Tier.UNKNOWN

    def test_yandex_non_legal_path_is_unknown(self):
        assert classify("https://yandex.ru/search/?text=usn") is Tier.UNKNOWN

    def test_fetch_policy(self):
        assert is_fetch_allowed("https://nalog.gov.ru/", allow_unknown=False)
        assert not is_fetch_allowed("https://blog.example/", allow_unknown=False)
        assert is_fetch_allowed("https://blog.example/", allow_unknown=True)
        assert not is_fetch_allowed("file:///etc/passwd", allow_unknown=True)
        assert not is_fetch_allowed("ftp://nalog.gov.ru/", allow_unknown=True)
