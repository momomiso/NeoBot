"""真机冒烟：DeepSeek 搜索 / 抓取 API（会花真钱，默认跳过）。

跑法：
    NEOBOT_LIVE_SEARCH=1 uv run pytest app/tests/manual/test_live_deepseek_search.py -q -s

成本：固定 5 次请求 ≈ ¥0.025（官方 $0.75 / 1000 成功请求，按 ¥5/1000 折算）。
断言只钉「形状 / 比例 / 上限」，不钉具体内容——真实结果天天变，钉内容等于自造 flaky。
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
import pytest_asyncio

from neobot_app.config.loader.env import load_env
from neobot_app.config.schemas.bot import WebSearchApiConfig
from neobot_app.web_search.deepseek_api import ApiConfig, DeepSeekSearchEngine, resolve_api_key

# Windows 控制台默认 GBK：检索结果里带 emoji 时 print 会抛 UnicodeEncodeError，
# 那是输出编码问题，不是搜索失败——重定向为标准输出并放宽解码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

pytestmark = [
    pytest.mark.network,
    pytest.mark.timeout(180),
    pytest.mark.skipif(
        os.environ.get("NEOBOT_LIVE_SEARCH") != "1",
        reason="真机冒烟：需 NEOBOT_LIVE_SEARCH=1 且 app/.env 里配好 DeepSeekSearch_APIKey",
    ),
]


@pytest_asyncio.fixture()
async def engine() -> Any:
    load_env()
    key = resolve_api_key("DeepSeekSearch_APIKey")
    if not key:
        pytest.skip("未配置 DeepSeekSearch_APIKey / DeepSeek_APIKey")
    config = ApiConfig.from_section(WebSearchApiConfig(), api_key=key)
    search_engine = DeepSeekSearchEngine(config)
    try:
        yield search_engine
    finally:
        await search_engine.aclose()


def _report(label: str, response: Any) -> None:
    print(f"\n[{label}] engine={response.engine} results={len(response.results)} "
          f"degraded={response.degraded} error={response.error}")
    for result in response.results:
        print(f"  - {result.rerank_score:>4} a={result.authority:>2} "
              f"{result.published_at or '-':<25} {result.url[:90]}")
        safe_title = result.title[:80].encode("utf-8", "replace").decode("utf-8", "replace")
        print(f"    title={safe_title} summary_len={len(result.summary)}")


async def test_01_chinese_query_returns_structured_results(engine: DeepSeekSearchEngine) -> None:
    response = await engine.search("DeepSeek 搜索 API 上线", num_results=5)
    _report("中文查询", response)

    assert response.error is None, response.error
    assert len(response.results) >= 3
    for result in response.results:
        assert result.url.startswith("http")
        assert result.title
        assert result.summary or result.snippet
    assert response.signals["api"]["billed"] == 1


async def test_02_site_operator_is_honoured(engine: DeepSeekSearchEngine) -> None:
    response = await engine.search("DeepSeek site:github.com", num_results=5)
    _report("site 运算符", response)

    assert response.results, response.error
    hosts = [result.url.split("/")[2].lower() for result in response.results]
    github = [host for host in hosts if host.endswith("github.com")]
    assert github, f"site:github.com 未生效，实际 host={hosts}"
    assert len(github) / len(hosts) >= 0.6, hosts


async def test_03_after_operator_keeps_dates(engine: DeepSeekSearchEngine) -> None:
    response = await engine.search("DeepSeek API after:2026-09-01", num_results=5)
    _report("after 运算符", response)

    assert response.results, response.error
    dated = [result for result in response.results if result.published_at]
    assert dated, "after: 查询没带回任何 published_at，运算符可能未生效"
    assert len(dated) / len(response.results) >= 0.3


async def test_04_need_content_returns_markdown(engine: DeepSeekSearchEngine) -> None:
    response = await engine.search("DeepSeek API 文档", num_results=3, need_content=True)
    _report("need_content", response)

    assert response.results, response.error
    fetched = [result for result in response.results if result.content_fetched and result.content]
    assert fetched, "need_content=true 却没带回任何正文"
    assert max(len(result.content or "") for result in response.results) > 200
    first = fetched[0]
    assert first.content is not None and not first.content.startswith("[获取失败]")


async def test_05_fetch_returns_markdown(engine: DeepSeekSearchEngine) -> None:
    outcome = await engine.fetch("https://api-docs.deepseek.com/guides/search")
    print(f"\n[Fetch] ok={outcome.ok} status={outcome.status} "
          f"title={outcome.title!r} len={len(outcome.content)} error={outcome.error}")

    assert outcome.ok is True, outcome.error
    assert len(outcome.content) > 200
    assert "search" in outcome.content.lower()


async def test_07_session_read_uses_fetch_api() -> None:
    """M3 接线：SearchSession.read() 走 Fetch API，正文标 markdown 并跳过 HTML 抽取器。"""
    from neobot_app.web_search import deepseek_api
    from neobot_app.web_search.session import SearchSession

    load_env()  # 没走 engine fixture，这里要自己把 app/.env 读进环境
    key = resolve_api_key("DeepSeekSearch_APIKey")
    if not key:
        pytest.skip("未配置 DeepSeekSearch_APIKey / DeepSeek_APIKey")
    deepseek_api.configure(WebSearchApiConfig(), api_key=key)  # 等价于启动期 apply_web_search_config

    session = SearchSession(engines=["deepseek"], fetch_mode="api_first")
    try:
        response = await session.search("DeepSeek API 文档", num_results=3)
        assert response.results, response.error
        first = response.results[0]
        print(f"\n[read] target={first.url[:100]}")

        returned = await session.read([first.index])
        read = returned[0]
        print(f"[read] fetched={read.content_fetched} format={read.content_format} "
              f"len={len(read.content or '')}")
        assert read.content_fetched is True
        assert read.content_format == "markdown"
        assert len(read.content or "") > 200
    finally:
        await session.aclose()
        deepseek_api.reset_runtime()


async def test_08_research_operators_send_one_query() -> None:
    """M4：research(mode=...) 在 operators 策略下只发 1 条查询（真机 1 次请求 ≈ ¥0.005）。"""
    from neobot_app.web_search import deepseek_api
    from neobot_app.web_search.session import SearchSession

    load_env()
    key = resolve_api_key("DeepSeekSearch_APIKey")
    if not key:
        pytest.skip("未配置 DeepSeekSearch_APIKey / DeepSeek_APIKey")
    deepseek_api.configure(WebSearchApiConfig(), api_key=key)

    session = SearchSession(engines=["deepseek"], research_strategy="operators")
    try:
        response = await session.research("DeepSeek API 文档", modes="official", num_results=5)
        assert response.success, response.error
        round_label = session.rounds[-1].query
        print(f"\n[research] label={round_label} results={len(response.results)}")
        assert "operators×1" in round_label
        assert response.results
    finally:
        await session.aclose()
        deepseek_api.reset_runtime()


async def test_06_cost_and_budget_accounting(engine: DeepSeekSearchEngine) -> None:
    """顺带确认两次成功请求后：预算计数 +1、单价换算正确。"""
    before = engine.calls
    await engine.search("DeepSeek V4", num_results=3)
    after = engine.calls
    assert after == before + 1
    assert engine.config.cost_per_request == pytest.approx(0.005)
