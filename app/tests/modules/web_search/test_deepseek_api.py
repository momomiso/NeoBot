"""DeepSeek 搜索 / 抓取 API 通道（deepseek_api.py）的契约与边界测试。

全部走 httpx.MockTransport：不出网、不花钱、不依赖时间（QPS 用假时钟）。
覆盖：请求形状 / 响应映射 / 空结果语义 / 质量门 / 状态码分级 / 熔断 / 预算 /
QPS 间隔 / Fetch 前置拦截 / 成本记账。
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from neobot_app.web_search import deepseek_api
from neobot_app.web_search.deepseek_api import (
    ApiConfig,
    DeepSeekSearchEngine,
    SearchApiRuntime,
    search_scope,
)
from neobot_app.web_search.models import SearchResponse


class FakeClock:
    """sleep 推进虚拟时间：QPS 与退避断言不依赖真实等待。"""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 4))
        self.now += seconds


ROW = {
    "title": "DeepSeek V4 released",
    "url": "https://example.com/deepseek-v4",
    "snippet": "short",
    "summary": "long summary for the model",
    "content": "",
    "site_name": "Example News",
    "site_icon": "https://example.com/favicon.ico",
    "date": "2026-06-01T00:00:00+00:00",
    "authority": 3,
    "rerank_score": 2.4,
}


def rows(*overrides: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**ROW, **item} for item in (overrides or ({},))]


def make_engine(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    config: ApiConfig | None = None,
    clock: FakeClock | None = None,
) -> tuple[DeepSeekSearchEngine, list[httpx.Request], FakeClock]:
    seen: list[httpx.Request] = []
    clock = clock or FakeClock()

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    runtime = SearchApiRuntime(
        config or ApiConfig(api_key="sk-test"),
        transport=httpx.MockTransport(_record),
        clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    return DeepSeekSearchEngine(runtime=runtime), seen, clock


def ok(payload_rows: list[dict[str, Any]] | None = None) -> httpx.Response:
    return httpx.Response(200, json={"results": payload_rows if payload_rows is not None else rows()})


# ── 请求形状 ────────────────────────────────────────────────────────


async def test_request_shape_auth_and_clipping() -> None:
    engine, seen, _ = make_engine(lambda r: ok())
    await engine.search("x" * 300, num_results=99)

    request = seen[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/tools/search"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert len(body["query"]) == 128
    assert body["count"] == 50
    assert body["need_content"] is False
    assert "ori_question" not in body


async def test_ori_question_and_need_content() -> None:
    engine, seen, _ = make_engine(lambda r: ok())
    await engine.search("q", ori_question="原" * 200, need_content=True)
    body = json.loads(seen[0].content)
    assert len(body["ori_question"]) == 128
    assert body["need_content"] is True


# ── 响应映射 ────────────────────────────────────────────────────────


async def test_mapping_keeps_structured_fields() -> None:
    engine, _, _ = make_engine(lambda r: ok())
    resp = await engine.search("DeepSeek V4")

    assert resp.error is None and resp.degraded is False
    result = resp.results[0]
    assert result.engine == "deepseek" and result.channel == "api"
    assert result.summary == "long summary for the model"
    assert result.authority == 3
    assert result.rerank_score == 2.4
    assert result.published_at == "2026-06-01T00:00:00+00:00"
    assert result.site_name == "Example News"
    assert result.content_fetched is False
    assert resp.signals["api"]["billed"] == 1


async def test_content_marks_fetched_only_when_non_empty() -> None:
    engine, _, _ = make_engine(lambda r: ok(rows({"content": "# body"})))
    resp = await engine.search("q")
    assert resp.results[0].content == "# body"
    assert resp.results[0].content_fetched is True


# ── 语义边界 ────────────────────────────────────────────────────────


async def test_empty_results_is_not_failure() -> None:
    engine, seen, _ = make_engine(lambda r: ok([]))
    resp = await engine.search("q")
    assert resp.error is None and resp.results == []
    assert resp.signals["empty"] is True
    assert len(seen) == 1
    assert resp.signals["api"]["billed"] == 0


async def test_low_rerank_keeps_results_but_flags_degraded() -> None:
    engine, _, _ = make_engine(lambda r: ok(rows({"rerank_score": 0.4})))
    resp = await engine.search("q")
    assert len(resp.results) == 1
    assert resp.error is None
    assert resp.degraded is True
    assert resp.signals["validation"]["low_rerank"] == 1


async def test_dirty_urls_fail_the_gate() -> None:
    engine, _, _ = make_engine(lambda r: ok(rows({"url": "javascript:alert(1)"})))
    resp = await engine.search("q")
    assert resp.error is not None
    assert resp.signals["failure"] == "structure"
    assert resp.signals["api"]["billed"] == 0


# ── 状态码分级 ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status,permanent,calls",
    [
        (400, True, 1),
        (401, True, 1),
        (402, True, 1),
        (403, True, 1),
        (404, True, 1),
        (413, True, 1),
        (408, False, 3),
        (429, False, 3),
        (500, False, 3),
        (504, False, 3),
    ],
)
async def test_status_mapping(status: int, permanent: bool, calls: int) -> None:
    engine, seen, _ = make_engine(lambda r: httpx.Response(status, json={"error": {"message": "boom"}}))
    resp = await engine.search("q")
    assert resp.results == [] and resp.error
    assert (resp.signals.get("permanent") is True) is permanent
    assert len(seen) == calls


async def test_retry_after_is_respected() -> None:
    engine, _, clock = make_engine(
        lambda r: httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}})
    )
    await engine.search("q")
    assert clock.sleeps[0] == 2.0


async def test_402_trips_breaker_and_notifies(monkeypatch: pytest.MonkeyPatch) -> None:
    notified: list[str] = []
    monkeypatch.setattr(deepseek_api, "_notifier", notified.append)
    engine, seen, _ = make_engine(lambda r: httpx.Response(402, json={"error": {"message": "balance"}}))

    resp = await engine.search("q")
    assert resp.error and "402" in resp.error
    assert engine.available() is False
    assert notified and "402" in notified[0]

    await engine.search("q again")
    assert len(seen) == 1


async def test_repeated_429_opens_breaker() -> None:
    engine, seen, _ = make_engine(lambda r: httpx.Response(429, json={"error": {"message": "qps"}}))
    await engine.search("q")
    assert engine.available() is False
    assert len(seen) == 3
    await engine.search("q2")
    assert len(seen) == 3


# ── 预算 / 限流 / 记账 ──────────────────────────────────────────────


async def test_daily_budget_stops_requests() -> None:
    engine, seen, _ = make_engine(lambda r: ok(), config=ApiConfig(api_key="sk-test", daily_request_limit=1))
    assert (await engine.search("a")).error is None
    second = await engine.search("b")
    assert second.signals["budget_exceeded"] is True
    assert len(seen) == 1


async def test_conversation_budget_is_scoped() -> None:
    engine, seen, _ = make_engine(
        lambda r: ok(), config=ApiConfig(api_key="sk-test", conversation_request_limit=1)
    )
    with search_scope("chat-1"):
        assert (await engine.search("a")).error is None
        blocked = await engine.search("b")
    assert blocked.signals["budget_exceeded"] is True
    with search_scope("chat-2"):
        assert (await engine.search("c")).error is None
    assert len(seen) == 2


async def test_qps_spacing_uses_injected_clock() -> None:
    engine, _, clock = make_engine(lambda r: ok(), config=ApiConfig(api_key="sk-test", qps_limit=4))
    for query in ("a", "b", "c"):
        await engine.search(query)
    assert clock.sleeps == [0.25, 0.25]


async def test_cost_recorded_once_per_success(monkeypatch: pytest.MonkeyPatch) -> None:
    charged: list[float] = []
    monkeypatch.setattr(deepseek_api, "_cost_recorder", charged.append)

    engine, _, _ = make_engine(lambda r: ok())
    await engine.search("ok")
    assert charged == [pytest.approx(0.005)]

    failing, _, _ = make_engine(lambda r: httpx.Response(500, json={}))
    await failing.search("boom")
    assert charged == [pytest.approx(0.005)]


async def test_keyless_engine_makes_no_request() -> None:
    engine, seen, _ = make_engine(lambda r: ok(), config=ApiConfig(api_key=""))
    resp = await engine.search("q")
    assert resp.error and "Key" in resp.error
    assert seen == []
    assert engine.available() is False


# ── Fetch ───────────────────────────────────────────────────────────


async def test_fetch_returns_markdown() -> None:
    engine, seen, _ = make_engine(lambda r: httpx.Response(200, json={"title": "T", "content": "# body"}))
    outcome = await engine.fetch("https://example.com/a")
    assert outcome.ok is True
    assert outcome.content == "# body" and outcome.title == "T"
    assert seen[0].url.path == "/v1/tools/fetch"
    assert json.loads(seen[0].content) == {"url": "https://example.com/a"}


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:9981/panel", "http://[::1]/x", "http://user:pass@example.com/", "file:///etc/passwd"],
)
async def test_fetch_blocks_private_and_credential_urls(url: str) -> None:
    engine, seen, _ = make_engine(lambda r: httpx.Response(200, json={"content": "x"}))
    outcome = await engine.fetch(url)
    assert outcome.ok is False
    assert seen == []


async def test_fetch_failure_reports_error_without_raising() -> None:
    engine, _, _ = make_engine(lambda r: httpx.Response(404, json={"error": {"message": "no content"}}))
    outcome = await engine.fetch("https://example.com/gone")
    assert outcome.ok is False and outcome.status == 404 and "404" in outcome.error


# ── 与回退链的衔接 ──────────────────────────────────────────────────


class _OkEngine:
    """最小可用引擎桩：只为验证「上一级不可用时下一级照常接住」。"""

    name = "bing"

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, num_results: int = 10, **kwargs: Any) -> SearchResponse:
        from neobot_app.web_search.models import SearchResult

        self.calls += 1
        return SearchResponse(
            query=query,
            results=[SearchResult(index=0, title="t", url="https://example.com/b", snippet="s", engine="bing")],
            engine="bing",
        )


async def test_manager_skips_unavailable_engine_without_request() -> None:
    from neobot_app.web_search.manager import SearchManager

    engine, seen, _ = make_engine(lambda r: ok(), config=ApiConfig(api_key=""))
    manager = SearchManager(engines=["deepseek", "bing"], min_delay=0.0, max_retries=0, browser_fallback=False)
    manager._engines["deepseek"] = engine
    manager._engines["bing"] = _OkEngine()

    resp = await manager.search_with_fallback("q")

    assert resp.engine == "bing"
    assert seen == []
    assert resp.signals["stages"][0]["kind"] == "unavailable"
    assert resp.degraded is False


async def test_manager_does_not_retry_permanent_failure() -> None:
    from neobot_app.web_search.manager import SearchManager

    engine, seen, _ = make_engine(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    manager = SearchManager(engines=["deepseek"], min_delay=0.0, max_retries=3, browser_fallback=False)
    manager._engines["deepseek"] = engine

    resp = await manager.search_with_fallback("q")

    assert len(seen) == 1
    assert resp.error


def test_get_engine_registers_deepseek_lazily() -> None:
    from neobot_app.web_search.engine import ENGINE_REGISTRY, get_engine

    engine = get_engine("deepseek")
    assert isinstance(engine, DeepSeekSearchEngine)
    assert ENGINE_REGISTRY["deepseek"] is DeepSeekSearchEngine
