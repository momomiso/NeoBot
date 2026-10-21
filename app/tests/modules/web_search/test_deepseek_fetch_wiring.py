"""Fetch API 接线：SearchSession.read 与 WebTools.web_fetch 的 api_first / 回落 / 闸门。

不变量（改这里前先读一遍）：
1. api_first 成功 → 正文标记 content_format=markdown，**不再**走本地 httpx；
2. API 失败/空正文 → 回落本地抓取，content_format 仍是 html；
3. 私网 / 带账号密码 URL 在**发出前**就被拒，绝不给第三方；
4. fetch_mode=off / http_first → 一次 API 都不试。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from neobot_app.agent_tools.contracts import AgentToolError, ToolContext
from neobot_app.agent_tools.web import WebTools
from neobot_app.web_search import session as session_module
from neobot_app.web_search.deepseek_api import FetchOutcome
from neobot_app.web_search.models import SearchResult
from neobot_app.web_search.session import SearchSession

CTX = ToolContext("group:1:main", "group:1")

#: 原始类先抓住：monkeypatch 会把 httpx.AsyncClient 换掉，工厂里必须用这份引用。
_REAL_CLIENT = httpx.AsyncClient


def _mock_client(**kwargs: Any) -> httpx.AsyncClient:
    """按 read() 的调用形状返回 MockTransport 客户端（丢弃 headers 以免重复传参）。"""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=GOOD_HTML, headers={"content-type": "text/html"})
    )
    safe = {k: v for k, v in kwargs.items() if k not in {"headers", "transport"}}
    return _REAL_CLIENT(transport=transport, **safe)
GOOD_HTML = "<html><body>" + ("这是一段足够长的正常正文内容。" * 40) + "</body></html>"


def _session(result: SearchResult, *, handler: Any, mode: str = "api_first") -> SearchSession:
    session = SearchSession.__new__(SearchSession)
    session._results_index = {result.index: result}
    session._read_urls = set()
    session._rounds = []
    session._read_timeout = 5.0
    session._fetch_mode = mode
    session._api_fetch = handler
    return session


def _result(url: str = "https://example.com/page") -> SearchResult:
    return SearchResult(index=0, title="t", url=url, snippet="s", engine="fake")


def _allow_public(monkeypatch: pytest.MonkeyPatch) -> None:
    async def allow(_url: str) -> bool:
        return True

    monkeypatch.setattr(session_module, "validate_public_url_async", allow)


# ── SearchSession.read ──────────────────────────────────────────────


async def test_read_prefers_api_and_marks_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_public(monkeypatch)
    calls: list[str] = []

    async def handler(url: str) -> FetchOutcome:
        calls.append(url)
        return FetchOutcome(ok=True, url=url, title="API 标题", content="# 正文 Markdown")

    def _no_http(**kwargs: Any) -> Any:
        raise AssertionError("API 成功时不应构造本地 httpx 客户端")

    monkeypatch.setattr(session_module.httpx, "AsyncClient", _no_http)
    result = _result()
    session = _session(result, handler=handler)

    returned = await session.read([0])

    assert calls == ["https://example.com/page"]
    assert returned[0].content_fetched is True
    assert returned[0].content_format == "markdown"
    assert returned[0].content == "# 正文 Markdown"
    # 已有标题不被 API 标题覆盖（引擎标题通常更贴合查询）
    assert returned[0].title == "t"
    assert session._read_urls == {"https://example.com/page"}


async def test_read_falls_back_to_local_when_api_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_public(monkeypatch)

    async def handler(url: str) -> FetchOutcome:
        return FetchOutcome(ok=False, url=url, error="HTTP 404")

    monkeypatch.setattr(session_module.httpx, "AsyncClient", _mock_client)
    result = _result()
    session = _session(result, handler=handler)

    returned = await session.read([0])

    assert returned[0].content_fetched is True
    assert returned[0].content_format == "html"
    assert "正常正文内容" in (returned[0].content or "")


async def test_read_blocks_private_url_before_api(monkeypatch: pytest.MonkeyPatch) -> None:
    async def deny(_url: str) -> bool:
        return False

    monkeypatch.setattr(session_module, "validate_public_url_async", deny)
    calls: list[str] = []

    async def handler(url: str) -> FetchOutcome:
        calls.append(url)
        return FetchOutcome(ok=True, content="不该发生")

    result = _result("http://127.0.0.1:9981/panel")
    session = _session(result, handler=handler)

    returned = await session.read([0])

    assert calls == []
    assert returned[0].content_fetched is False
    assert "已阻止" in (returned[0].content or "")


@pytest.mark.parametrize("mode", ["off", "http_first"])
async def test_read_skips_api_when_mode_disallows(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    _allow_public(monkeypatch)
    calls: list[str] = []

    async def handler(url: str) -> FetchOutcome:
        calls.append(url)
        return FetchOutcome(ok=True, content="# x")

    monkeypatch.setattr(session_module.httpx, "AsyncClient", _mock_client)
    result = _result()
    session = _session(result, handler=handler, mode=mode)

    returned = await session.read([0])

    assert calls == []
    assert returned[0].content_format == "html"


# ── WebTools.web_fetch ──────────────────────────────────────────────


def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def allow(_url: str) -> bool:
        return True

    async def resolve(*args: Any, **kwargs: Any) -> list[Any]:
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("neobot_app.agent_tools.web.validate_public_url_async", allow)
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)


async def test_web_fetch_uses_api_first(monkeypatch: pytest.MonkeyPatch) -> None:
    _public_dns(monkeypatch)
    seen: list[str] = []

    async def handler(url: str) -> FetchOutcome:
        seen.append(url)
        return FetchOutcome(ok=True, url=url, title="T", content="# Markdown 正文" * 20)

    tools = WebTools(api_fetch=handler, client_factory=lambda **kw: (_ for _ in ()).throw(AssertionError("不应走本地")))
    result = await tools.execute("web_fetch", {"url": "https://example.com/a"}, CTX)

    assert seen == ["https://example.com/a"]
    assert result["statusCode"] == 200
    assert result["untrusted"] is True
    assert result["body"]["content"].startswith("# Markdown 正文")


async def test_web_fetch_falls_back_when_api_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _public_dns(monkeypatch)

    async def handler(url: str) -> FetchOutcome:
        return FetchOutcome(ok=False, url=url, error="HTTP 403")

    seen: list[httpx.Request] = []

    def response(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="local body")

    tools = WebTools(api_fetch=handler,
                     client_factory=lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(response), **kw))
    result = await tools.execute("web_fetch", {"url": "https://example.com/a"}, CTX)

    assert result["body"]["content"] == "local body"
    assert len(seen) == 1


async def test_web_fetch_api_path_still_rejects_private_url(monkeypatch: pytest.MonkeyPatch) -> None:
    async def deny(_url: str) -> bool:
        return False

    monkeypatch.setattr("neobot_app.agent_tools.web.validate_public_url_async", deny)
    calls: list[str] = []

    async def handler(url: str) -> FetchOutcome:
        calls.append(url)
        return FetchOutcome(ok=True, content="x")

    tools = WebTools(api_fetch=handler)
    with pytest.raises(AgentToolError):
        await tools.execute("web_fetch", {"url": "http://127.0.0.1:9981/panel"}, CTX)
    assert calls == []


async def test_web_fetch_http_first_skips_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _public_dns(monkeypatch)
    calls: list[str] = []

    async def handler(url: str) -> FetchOutcome:
        calls.append(url)
        return FetchOutcome(ok=True, content="x")

    def response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="local")

    tools = WebTools(fetch_mode="http_first", api_fetch=handler,
                     client_factory=lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(response), **kw))
    result = await tools.execute("web_fetch", {"url": "https://example.com/a"}, CTX)

    assert calls == []
    assert result["body"]["content"] == "local"
