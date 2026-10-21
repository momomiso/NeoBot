"""研究模式 operators 与工具参数（sites/lang/after/before/need_content）。

核心不变量：
1. operators 策略下，一次 research 只发 1 条查询（DeepSeek 按请求计费，变体 = 4 倍成本）；
2. variants 策略保留旧行为（1 主查询 + N 变体），供排障/对照；
3. 结构化参数拼进查询时**运算符优先、主题可截断**，且总长不超过 128 码点；
4. need_content 只转发给签名支持的引擎，bing/DDG 不受影响。
"""

from __future__ import annotations

from typing import Any

import pytest

from neobot_app.web_search.manager import SearchManager
from neobot_app.web_search.models import SearchResponse, SearchResult
from neobot_app.web_search.session import SearchSession
from neobot_app.web_search_package import WebSearchExecutor


def _ok(engine: str = "fake") -> SearchResponse:
    return SearchResponse(
        query="q",
        results=[SearchResult(index=0, title="t", url="https://example.com/a", snippet="s", engine=engine)],
        engine=engine,
    )


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def search_with_fallback(
        self, query: str, num_results: int | None = None, *, need_content: bool | None = None
    ) -> SearchResponse:
        self.calls.append({"query": query, "n": num_results, "need_content": need_content})
        response = _ok()
        response.query = query
        return response


def _session(strategy: str = "operators") -> tuple[SearchSession, FakeManager]:
    session = SearchSession(browser_fallback=False, research_strategy=strategy)
    manager = FakeManager()
    session._manager = manager  # 注入计数器，绕开真实网络
    return session, manager


# ── 研究策略 ────────────────────────────────────────────────────────


async def test_operators_strategy_sends_one_query() -> None:
    session, manager = _session("operators")
    response = await session.research("量子计算", modes="encyclopedia")

    assert response.success
    assert len(manager.calls) == 1
    query = manager.calls[0]["query"]
    assert "site:baike.baidu.com" in query
    assert "量子计算" in query


async def test_variants_strategy_keeps_old_cost_profile() -> None:
    session, manager = _session("variants")
    await session.research("量子计算", modes="encyclopedia")

    # 1 条主查询 + 3 条 encyclopedia 变体
    assert len(manager.calls) == 4
    assert manager.calls[0]["query"] == "量子计算"


async def test_operators_merges_structured_filters() -> None:
    session, manager = _session("operators")
    await session.research(
        "DeepSeek API", modes="news", sites=["github.com"], lang="zh", after="2026-01-01"
    )

    query = manager.calls[0]["query"]
    assert "site:github.com" in query
    assert "lang:zh" in query
    # 显式 after 覆盖 news 的默认 30 天窗口，且 lang 不重复拼接
    assert "after:2026-01-01" in query
    assert query.count("lang:zh") == 1
    assert len(query) <= 128


async def test_two_modes_send_two_queries_at_most() -> None:
    session, manager = _session("operators")
    await session.research("量子计算", modes=["encyclopedia", "academic"])
    assert len(manager.calls) == 2


# ── 查询组装 ────────────────────────────────────────────────────────


def test_compose_query_trims_topic_not_operators() -> None:
    query = SearchSession.compose_query(
        "主" * 300, sites=["github.com", "arxiv.org"], lang="en", after="2026-01-01"
    )
    assert len(query) <= 128
    assert "site:github.com OR site:arxiv.org" in query
    assert query.endswith("after:2026-01-01")


def test_compose_query_without_filters_is_plain() -> None:
    assert SearchSession.compose_query("  量子计算  ") == "量子计算"


# ── need_content 转发 ───────────────────────────────────────────────


class _NeedContentEngine:
    name = "bing"

    def __init__(self) -> None:
        self.seen: list[Any] = []

    async def search(self, query: str, num_results: int = 10, *, need_content: Any = None) -> SearchResponse:
        self.seen.append(need_content)
        return _ok("bing")


class _PlainEngine:
    name = "duckduckgo"

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        self.calls += 1
        return _ok("duckduckgo")


async def test_manager_forwards_need_content_only_when_supported() -> None:
    plain = _PlainEngine()
    manager = SearchManager(engines=["duckduckgo"], min_delay=0.0, max_retries=0, browser_fallback=False)
    manager._engines["duckduckgo"] = plain

    response = await manager.search("q", 5, engine="duckduckgo", need_content=True)

    assert response.success and plain.calls == 1  # 没 kwargs 也不该抛 TypeError


async def test_manager_passes_need_content_to_capable_engine() -> None:
    engine = _NeedContentEngine()
    manager = SearchManager(engines=["bing"], min_delay=0.0, max_retries=0, browser_fallback=False)
    manager._engines["bing"] = engine

    await manager.search("q", 5, engine="bing", need_content=True)

    assert engine.seen == [True]


# ── 工具面透传 ──────────────────────────────────────────────────────


class _CapturingSession:
    instances: list[dict[str, Any]] = []
    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).instances.append(kwargs)

    async def search(self, query: str, num_results: int = 10, **kwargs: Any) -> SearchResponse:
        type(self).last_kwargs = kwargs
        return _ok()


async def test_webtools_search_passes_operator_args() -> None:
    from neobot_app.agent_tools.web import WebTools

    _CapturingSession.instances = []
    tools = WebTools(search_factory=_CapturingSession)
    result = await tools._search(
        {"queries": ["q"], "sites": ["github.com"], "lang": "zh", "need_content": True}
    )

    assert result["sources"]
    assert _CapturingSession.last_kwargs == {
        "sites": ["github.com"],
        "lang": "zh",
        "need_content": True,
    }


async def test_webtools_rejects_bad_operator_args() -> None:
    from neobot_app.agent_tools.contracts import AgentToolError
    from neobot_app.agent_tools.web import WebTools

    tools = WebTools(search_factory=_CapturingSession)
    with pytest.raises(AgentToolError):
        await tools._search({"queries": ["q"], "sites": []})
    with pytest.raises(AgentToolError):
        await tools._search({"queries": ["q"], "need_content": "yes"})


class _FakeExecutorSession:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def search(self, query: str, num_results: int = 10, **kwargs: Any) -> SearchResponse:
        self.kwargs = kwargs
        return _ok()

    async def research(self, topic: str, **kwargs: Any) -> SearchResponse:
        self.kwargs = kwargs
        return _ok()

    def reset(self) -> None:
        return None

    def get_summary_for_agent(self) -> str:
        return ""


async def test_executor_search_passes_need_content() -> None:
    executor = WebSearchExecutor(engines=["bing"])
    fake = _FakeExecutorSession()
    executor._get_session = lambda: fake  # type: ignore[method-assign]

    await executor._handle_search({"query": "q", "need_content": True, "sites": ["a.com"]})

    assert fake.kwargs["need_content"] is True
    assert fake.kwargs["sites"] == ["a.com"]
