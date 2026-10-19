"""配置接线：section_to_config_dict / apply_web_search_config / WebTools 注入。

背景：主 Agent 的 WebTools 此前不带任何配置（runtime.py 的 WebTools()），
[web_search].engines 改了也不生效；同时 [web_search.api] 需要一次进程级应用。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from neobot_app.agent_tools.web import WebTools
from neobot_app.config.schemas.bot import WebSearchApiConfig, WebSearchConfig
from neobot_app.web_search import deepseek_api
from neobot_app.web_search.config_section import section_to_config_dict, session_kwargs
from neobot_app.web_search.models import SearchResponse, SearchResult


def test_section_defaults_include_deepseek() -> None:
    result = section_to_config_dict(WebSearchConfig())
    assert result["engines"][0] == "deepseek"
    assert result["browser_fallback"] is True
    assert result["max_rounds"] == 5


def test_section_to_config_dict_handles_missing_section() -> None:
    assert section_to_config_dict(None) == {}


def test_session_kwargs_filters_unknown_keys() -> None:
    kwargs = session_kwargs({"engines": ["bing"], "preview_pages_limit": 12, "max_rounds": 3})
    assert kwargs == {"engines": ["bing"], "max_rounds": 3}


def test_apply_web_search_config_sets_process_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from neobot_app.bootstrap import apply_web_search_config

    monkeypatch.setenv("DeepSeekSearch_APIKey", "sk-from-env")
    config = SimpleNamespace(web_search=WebSearchConfig(api=WebSearchApiConfig(qps_limit=3.0)))

    apply_web_search_config(config)

    applied = deepseek_api.get_config()
    assert applied.api_key == "sk-from-env"
    assert applied.qps_limit == 3.0
    assert applied.enabled is True
    deepseek_api.reset_runtime()


class _CapturingSession:
    """记录构造参数，并返回一条结果，供 WebTools 走完 _search 主路径。"""

    instances: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).instances.append(kwargs)

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        return SearchResponse(
            query=query,
            results=[SearchResult(index=0, title="t", url="https://example.com/a", snippet="s", engine="fake")],
            engine="fake",
        )

    async def research(self, *args: Any, **kwargs: Any) -> SearchResponse:
        return await self.search(str(args[0]) if args else "")


async def test_webtools_passes_web_search_config_to_session() -> None:
    _CapturingSession.instances = []
    tools = WebTools(
        search_factory=_CapturingSession,
        web_search_config={
            "engines": ["duckduckgo"],
            "max_rounds": 2,
            "preview_pages_limit": 12,
            "browser_fallback": False,
            "engine_budgets": {"duckduckgo": 15},
        },
    )

    result = await tools._search({"queries": ["q1"]})

    assert result["sources"][0]["url"] == "https://example.com/a"
    assert _CapturingSession.instances[0]["engines"] == ["duckduckgo"]
    assert _CapturingSession.instances[0]["max_rounds"] == 2
    assert _CapturingSession.instances[0]["browser_fallback"] is False
    assert _CapturingSession.instances[0]["engine_budgets"] == {"duckduckgo": 15}
    assert tools.max_sources == 12
