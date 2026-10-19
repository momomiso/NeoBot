"""DeepSeek 搜索 / 抓取 API 通道（/v1/tools/search、/v1/tools/fetch）。

为什么单独成文件、为什么带进程级状态
------------------------------------
get_engine(name) 不带参数（engine.py:437），凭证只能来自进程级配置；更要命的是
调用链每次查询都会新建 SearchSession -> 新建 SearchManager -> 新建引擎实例
（session.py:117、web_tools 的 search_factory），所以**限流、熔断、日预算、
HTTP 连接都必须放在模块级 SearchApiRuntime 里**，否则每次查询都会重置计数：5 QPS 形同
不存在，日上限永远打不到。

与 HTML 通道的刻意差异（不要「统一」掉）
----------------------------------------
1. 出口判据用 validate_api_results：不套壳页/验证码/li.b_algo；C3 只记 warning；
   低 rerank_score / 未知 authority 只标 degraded，不清空结果。
2. 空 results 是 EMPTY（不是失败、不重试、不回退），与 manager.py 既有语义一致。
3. 4xx 分「永久失败」与「可重试」，避免重试把账单放大（官方只对 408/429/5xx 承诺可重试）。
4. 计费口径：只对「HTTP 200 且至少一条结果」记账，失败/空结果不记。

配置来源
--------
bootstrap 调 configure(section) 一次（section 是 [web_search.api] 的 dataclass）；
测试直接构造 ApiConfig + SearchApiRuntime 注入 transport/clock，不碰全局。
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx

from neobot_app.web_search.engine import ENGINE_REGISTRY, BaseSearchEngine
from neobot_app.web_search.models import SearchResponse, SearchResult
from neobot_app.web_search.urls import host_of, normalize_result_url
from neobot_app.web_search.validate import FailureKind, validate_api_results

logger = logging.getLogger(__name__)

SEARCH_PATH = "/v1/tools/search"
FETCH_PATH = "/v1/tools/fetch"

#: 官方约束：query 1–128 个 Unicode 码点；fetch url 最多 8192 字符。
MAX_QUERY_CODEPOINTS = 128
MAX_FETCH_URL_CHARS = 8192

#: 官方口径：只有 408 / 429 / 5xx 建议重试；其余 4xx 改请求前重试无意义且会放大账单。
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: 连续多少次 429 触发熔断（单次 429 只退避，不熔断：官方 QPS 抖动很常见）。
BREAKER_429_THRESHOLD = 3

#: 会话级预算的作用域；调用方（WebTools / SearchSession）进入一次会话时 set。
CURRENT_SEARCH_SCOPE: ContextVar[str] = ContextVar("web_search_scope", default="")


@contextlib.contextmanager
def search_scope(scope: str) -> Iterator[None]:
    """把一段调用标记成同一个会话，用于 conversation_request_limit 计数。"""
    token = CURRENT_SEARCH_SCOPE.set(str(scope or ""))
    try:
        yield
    finally:
        CURRENT_SEARCH_SCOPE.reset(token)


# ── 配置 ────────────────────────────────────────────────────────────


@dataclass
class ApiConfig:
    """[web_search.api] 的运行时视图（与 config/schemas/bot.py 同名字段）。"""

    enabled: bool = True
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    api_key_env: str = "DeepSeekSearch_APIKey"
    timeout_seconds: float = 8.0
    qps_limit: float = 4.0
    default_count: int = 10
    max_count: int = 50
    min_rerank_score: float = 1.5
    content_on_search: bool = False
    fetch_mode: str = "api_first"
    price_per_1000_cny: float = 5.0
    daily_request_limit: int = 1000
    conversation_request_limit: int = 50
    cooldown_on_402_seconds: int = 3600
    cooldown_on_429_seconds: int = 60
    max_retries: int = 2
    research_strategy: str = "operators"
    allow_external_query: bool = True
    use_system_proxy: bool = False

    @property
    def cost_per_request(self) -> float:
        return max(0.0, float(self.price_per_1000_cny)) / 1000.0

    @classmethod
    def from_section(cls, section: Any, *, api_key: str | None = None) -> ApiConfig:
        """从配置段（dataclass）构造；section 为 None 时返回带 Key 探测的默认值。"""
        fields = [name for name in cls.__dataclass_fields__ if name != "api_key"]
        values: dict[str, Any] = {}
        for name in fields:
            if section is not None and hasattr(section, name):
                raw = getattr(section, name)
                if raw is not None:
                    values[name] = raw
        resolved = api_key if api_key is not None else resolve_api_key(
            str(values.get("api_key_env", cls.api_key_env))
        )
        values["api_key"] = resolved
        return cls(**values)


def _env_value(name: str) -> str:
    """大小写不敏感地读环境变量（Windows 上常见大小写不一致）。"""
    target = str(name or "").strip().casefold()
    if not target:
        return ""
    for key, value in os.environ.items():
        if key.casefold() == target and value:
            return str(value).strip()
    return ""


def resolve_api_key(api_key_env: str = "") -> str:
    """按 配置项 -> DeepSeekSearch_APIKey -> DeepSeek_APIKey 顺序找 Key。

    搜索专用 Key 优先于聊天 Key：两者共用一个 Key 时，402（余额不足）会同时
    打断聊天，而搜索本身只需降级即可。
    """
    candidates = [str(api_key_env or "").strip(), "DeepSeekSearch_APIKey", "DeepSeek_APIKey"]
    seen: set[str] = set()
    for name in candidates:
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        value = _env_value(name)
        if value:
            return value
    return ""


# ── 限流 / 熔断 / 预算 ──────────────────────────────────────────────


class TokenBucket:
    """单进程匀速令牌桶：next_at 记「下一次最早可发时刻」。

    注入 clock/sleeper 是为了测试：真实 sleep 会让 QPS 用例变成毫秒级 flaky。
    """

    def __init__(
        self,
        qps: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.qps = max(0.1, float(qps or 0.1))
        self._clock = clock
        self._sleep = sleeper
        self._next_at = 0.0
        self.waits: list[float] = []

    async def acquire(self) -> None:
        now = self._clock()
        if now < self._next_at:
            wait = self._next_at - now
            self.waits.append(round(wait, 4))
            await self._sleep(wait)
            now = self._clock()
        self._next_at = max(now, self._next_at) + 1.0 / self.qps


@dataclass
class Breaker:
    """熔断状态：402 直接长熔断，连续 429 短熔断。"""

    open_until: float = 0.0
    reason: str = ""
    consecutive_429: int = 0

    def is_open(self, now: float) -> bool:
        return now < self.open_until

    def trip(self, now: float, seconds: float, reason: str) -> None:
        self.open_until = max(self.open_until, now + max(0.0, float(seconds)))
        self.reason = reason

    def record_success(self) -> None:
        self.consecutive_429 = 0

    def record_throttle(self, now: float, cooldown: float) -> bool:
        """记一次 429；返回是否刚触发熔断。"""
        self.consecutive_429 += 1
        if self.consecutive_429 >= BREAKER_429_THRESHOLD:
            self.trip(now, cooldown, "连续 429")
            return True
        return False


@dataclass
class Budget:
    """进程级日预算 + 会话级预算（会话作用域由 CURRENT_SEARCH_SCOPE 提供）。"""

    day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    used_today: int = 0
    per_scope: dict[str, int] = field(default_factory=dict)

    def reset_if_new_day(self, today: date) -> None:
        if today != self.day:
            self.day = today
            self.used_today = 0
            self.per_scope.clear()

    def try_acquire(self, *, today: date, daily_limit: int, scope: str, scope_limit: int) -> bool:
        self.reset_if_new_day(today)
        if daily_limit > 0 and self.used_today >= daily_limit:
            return False
        if scope and scope_limit > 0 and self.per_scope.get(scope, 0) >= scope_limit:
            return False
        self.used_today += 1
        if scope:
            self.per_scope[scope] = self.per_scope.get(scope, 0) + 1
        return True


# ── 运行时（进程级）──────────────────────────────────────────────────


class SearchApiRuntime:
    """共享客户端 / 限流 / 熔断 / 预算 / 回调（进程级，测试可独立构造）。"""

    def __init__(
        self,
        config: ApiConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self._transport = transport
        self.clock = clock
        self.sleeper = sleeper
        self.bucket = TokenBucket(config.qps_limit, clock=clock, sleeper=sleeper)
        self.breaker = Breaker()
        self.budget = Budget()
        self._client: httpx.AsyncClient | None = None

    def client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                timeout=self.config.timeout_seconds,
                headers={
                    "Authorization": "Bearer " + self.config.api_key,
                    "Content-Type": "application/json",
                },
                transport=self._transport,
                trust_env=bool(self.config.use_system_proxy),
            )
            self._client = client
        return client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and not client.is_closed:
            with contextlib.suppress(Exception):
                await client.aclose()


_CONFIG: ApiConfig = ApiConfig()
_RUNTIME: SearchApiRuntime | None = None

#: 用量记账回调：bootstrap 注入（保持 web_search 不依赖 statistics 模块）。
_cost_recorder: Callable[[float], Awaitable[Any] | Any] | None = None
#: 熔断告警回调：bootstrap 注入（复用 notification_hub）。
_notifier: Callable[[str], Any] | None = None


def set_cost_recorder(recorder: Callable[[float], Awaitable[Any] | Any] | None) -> None:
    global _cost_recorder
    _cost_recorder = recorder


def set_notifier(notifier: Callable[[str], Any] | None) -> None:
    global _notifier
    _notifier = notifier


def configure(section: Any = None, *, api_key: str | None = None) -> ApiConfig:
    """bootstrap 期调用一次：把 [web_search.api] 段变成进程级配置。"""
    global _CONFIG, _RUNTIME
    _CONFIG = ApiConfig.from_section(section, api_key=api_key)
    _RUNTIME = None
    return _CONFIG


def get_config() -> ApiConfig:
    return _CONFIG


def reset_runtime() -> None:
    """测试用：丢掉进程级运行时（含共享客户端）。"""
    global _RUNTIME
    _RUNTIME = None


def get_runtime() -> SearchApiRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = SearchApiRuntime(_CONFIG)
    return _RUNTIME


# ── 抓取结果 ────────────────────────────────────────────────────────


@dataclass
class FetchOutcome:
    ok: bool
    url: str = ""
    title: str = ""
    content: str = ""
    error: str = ""
    status: int = 0


# ── 引擎 ────────────────────────────────────────────────────────────


def _clip_query(query: str) -> str:
    """按 Unicode 码点截断（不是字节，也不是 UTF-16 单位）。"""
    text = str(query or "").strip()
    return text[:MAX_QUERY_CODEPOINTS]


def _clip_url_chars(url: str) -> str:
    return str(url or "").strip()[:MAX_FETCH_URL_CHARS]


def _local_url_blocker(url: str) -> str:
    """发给第三方前的第一道闸：只放行「无凭据的公网 http(s) URL」。

    DNS 级私网判定由调用方（agent_tools/web.py 的 _pinned_url、session.read 的
    validate_public_url_async）负责；这里挡的是字面量，避免把 127.0.0.1 / [::1] /
    user:pass@host 这类明确不该外发的东西交给第三方。
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "URL 解析失败"
    if parsed.scheme not in {"http", "https"}:
        return "仅支持 http/https"
    if not parsed.hostname:
        return "缺少主机名"
    if parsed.username is not None or parsed.password is not None:
        return "URL 不得包含账号密码"
    host = parsed.hostname.strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local", ".internal")):
        return "拒绝内网主机名"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return ""
    if not address.is_global:
        return "拒绝非公网地址"
    return ""


def _authority(row: dict[str, Any]) -> int:
    try:
        return int(row.get("authority", -1))
    except (TypeError, ValueError):
        return -1


def _rerank(row: dict[str, Any]) -> float:
    try:
        return float(row.get("rerank_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _to_result(index: int, row: dict[str, Any], *, base: str) -> SearchResult | None:
    """把一条 API 结果映射成 SearchResult；URL 仍过统一归一化，脏行直接丢。"""
    title = str(row.get("title") or "").strip()
    raw_url = str(row.get("url") or "").strip()
    if not raw_url:
        return None
    display, raw_href = normalize_result_url(base, raw_url)
    if not display:
        return None
    content = str(row.get("content") or "")
    result = SearchResult(
        index=index,
        title=title or display,
        url=display,
        snippet=str(row.get("snippet") or ""),
        engine="deepseek",
        raw_url=raw_href,
        channel="api",
        site_name=str(row.get("site_name") or "") or host_of(display),
        published_at=(str(row.get("date")) if row.get("date") else None),
        summary=str(row.get("summary") or ""),
        authority=_authority(row),
        rerank_score=_rerank(row),
        site_icon=str(row.get("site_icon") or ""),
        content=content or None,
        content_fetched=bool(content),
    )
    return result


def _classify_status(status: int) -> tuple[FailureKind, bool]:
    """状态码 -> (失败分级, 是否永久失败)。"""
    if status in RETRYABLE_STATUS:
        return FailureKind.NETWORK, False
    return FailureKind.NETWORK, True


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None
    detail = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or "")
    detail = detail or (response.text or "")[:200]
    return f"HTTP {response.status_code}" + (f": {detail}" if detail else "")


async def _record_cost(cost: float) -> None:
    recorder = _cost_recorder
    if recorder is None or cost <= 0:
        return
    try:
        outcome = recorder(cost)
        if asyncio.iscoroutine(outcome) or isinstance(outcome, Awaitable):
            await outcome
    except Exception:
        logger.debug("搜索 API 用量记账失败（不影响搜索）", exc_info=True)


def _notify(message: str) -> None:
    notifier = _notifier
    if notifier is None:
        logger.warning(message)
        return
    try:
        notifier(message)
    except Exception:
        logger.debug("搜索 API 告警发送失败", exc_info=True)


class DeepSeekSearchEngine(BaseSearchEngine):
    """DeepSeek 官方搜索 / 抓取通道。"""

    name = "deepseek"

    def __init__(
        self,
        config: ApiConfig | None = None,
        *,
        runtime: SearchApiRuntime | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if runtime is not None:
            self._runtime = runtime
        else:
            self._runtime = SearchApiRuntime(config or _CONFIG, transport=transport, clock=clock, sleeper=sleeper)

    @property
    def config(self) -> ApiConfig:
        return self._runtime.config

    @property
    def calls(self) -> int:
        """已发出的 HTTP 请求数（测试与观测用）。"""
        return self._runtime.budget.used_today

    def available(self) -> bool:
        cfg = self.config
        if not cfg.enabled or not cfg.allow_external_query:
            return False
        if not cfg.api_key:
            return False
        return not self._runtime.breaker.is_open(self._runtime.clock())

    def unavailable_reason(self) -> str:
        cfg = self.config
        if not cfg.enabled:
            return "通道已关闭（[web_search.api].enabled=false）"
        if not cfg.allow_external_query:
            return "未允许外发查询（allow_external_query=false）"
        if not cfg.api_key:
            return "未配置 API Key（DeepSeekSearch_APIKey / DeepSeek_APIKey）"
        if self._runtime.breaker.is_open(self._runtime.clock()):
            return "熔断中：" + (self._runtime.breaker.reason or "未知原因")
        return ""

    async def aclose(self) -> None:
        await self._runtime.aclose()

    # ── 搜索 ──────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        num_results: int = 10,
        *,
        ori_question: str | None = None,
        need_content: bool | None = None,
    ) -> SearchResponse:
        runtime = self._runtime
        cfg = self.config
        started = time.perf_counter()

        if not self.available():
            return self._failure(query, self.unavailable_reason(), started, kind=FailureKind.NETWORK,
                                 signals={"available": False})

        clip = _clip_query(query)
        if not clip:
            return self._failure(query, "查询为空", started, kind=FailureKind.NETWORK, permanent=True)

        if not runtime.budget.try_acquire(
            today=datetime.now(timezone.utc).date(),
            daily_limit=cfg.daily_request_limit,
            scope=CURRENT_SEARCH_SCOPE.get(),
            scope_limit=cfg.conversation_request_limit,
        ):
            return self._failure(
                query,
                "达到 API 请求预算（日/会话上限），已降级到免费通道",
                started,
                kind=FailureKind.NETWORK,
                permanent=True,
                signals={"budget_exceeded": True},
            )

        body: dict[str, Any] = {
            "query": clip,
            "count": max(1, min(int(num_results or cfg.default_count), cfg.max_count)),
            "need_content": bool(cfg.content_on_search if need_content is None else need_content),
        }
        ori = _clip_query(ori_question or "")
        if ori:
            body["ori_question"] = ori

        attempts = 0
        last_error = ""
        for attempt in range(max(1, cfg.max_retries + 1)):
            attempts = attempt + 1
            await runtime.bucket.acquire()
            try:
                resp = await runtime.client().post(SEARCH_PATH, json=body)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempts > cfg.max_retries:
                    break
                await runtime.sleeper(2.0 ** (attempts - 1))
                continue

            if resp.status_code == 200:
                return await self._success(query, resp, started, attempts)

            kind, permanent = _classify_status(resp.status_code)
            last_error = _error_message(resp)
            now = runtime.clock()
            if resp.status_code == 402:
                runtime.breaker.trip(now, cfg.cooldown_on_402_seconds, "余额不足（402）")
                _notify("DeepSeek 搜索 API 余额不足（402），已熔断 " + str(cfg.cooldown_on_402_seconds) + " 秒")
                return self._failure(query, last_error, started, kind=kind, permanent=True,
                                     signals={"status": 402})
            if resp.status_code == 401:
                runtime.breaker.trip(now, cfg.cooldown_on_402_seconds, "Key 无效（401）")
                _notify("DeepSeek 搜索 API Key 无效（401），已熔断")
                return self._failure(query, last_error, started, kind=kind, permanent=True,
                                     signals={"status": 401})
            if resp.status_code == 429:
                if runtime.breaker.record_throttle(now, cfg.cooldown_on_429_seconds):
                    _notify("DeepSeek 搜索 API 连续 429，已熔断 " + str(cfg.cooldown_on_429_seconds) + " 秒")
                    return self._failure(query, last_error, started, kind=kind,
                                         signals={"status": 429, "breaker": True})
            if permanent or attempts > cfg.max_retries:
                return self._failure(query, last_error, started, kind=kind, permanent=permanent,
                                     signals={"status": resp.status_code})
            await runtime.sleeper(self._retry_delay(resp, attempts))

        return self._failure(query, last_error or "请求失败", started, kind=FailureKind.NETWORK)

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        raw = response.headers.get("retry-after", "")
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 2.0 ** (attempt - 1)

    async def _success(self, query: str, response: httpx.Response, started: float, attempts: int) -> SearchResponse:
        runtime = self._runtime
        cfg = self.config
        try:
            payload = response.json()
        except Exception as exc:
            return self._failure(query, f"响应不是 JSON: {exc}", started, kind=FailureKind.STRUCTURE,
                                 permanent=True)
        rows = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return self._failure(query, "响应缺少 results 数组", started, kind=FailureKind.STRUCTURE,
                                 permanent=True)

        raw_urls = [str(row.get("url") or "") for row in rows if isinstance(row, dict)]
        base = cfg.base_url
        results: list[SearchResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                mapped = _to_result(len(results), row, base=base)
            except Exception:  # 脏 URL/异常字段不得让整次搜索抛错
                logger.debug("DeepSeek 结果行映射失败，已丢弃", exc_info=True)
                continue
            if mapped is not None:
                results.append(mapped)
        runtime.breaker.record_success()

        if not results:
            return SearchResponse(
                query=query,
                results=[],
                engine=self.name,
                total_estimated=0,
                error=None,
                search_time_ms=(time.perf_counter() - started) * 1000,
                degraded=True,
                attempts=attempts,
                signals={"empty": True, "api": {"requests": attempts, "billed": 0, "dropped": len(raw_urls)}},
            )

        report = validate_api_results(query, results, min_rerank_score=cfg.min_rerank_score)
        cost = cfg.cost_per_request
        if report.valid:
            await _record_cost(cost)
        api_signals = {
            "requests": attempts,
            "billed": 1 if report.valid else 0,
            "cost_cny": round(cost, 6) if report.valid else 0.0,
            "dropped": max(0, len(raw_urls) - len(results)),
            "low_rerank": report.signals.get("low_rerank", 0),
            "high_authority": report.signals.get("high_authority", 0),
        }
        signals: dict[str, Any] = {"validation": report.signals, "api": api_signals}
        if not report.valid:
            codes = report.signals.get("codes") or []
            signals["failure"] = (
                FailureKind.STRUCTURE.value if "url_ok_ratio" in codes else FailureKind.CONTENT.value
            )
        return SearchResponse(
            query=query,
            results=results,
            engine=self.name,
            total_estimated=len(results),
            error=None if report.valid else report.reason,
            search_time_ms=(time.perf_counter() - started) * 1000,
            degraded=bool(report.signals.get("degraded")) or not report.valid,
            attempts=attempts,
            signals=signals,
        )

    def _failure(
        self,
        query: str,
        error: str,
        started: float,
        *,
        kind: FailureKind,
        permanent: bool = False,
        signals: dict[str, Any] | None = None,
    ) -> SearchResponse:
        payload: dict[str, Any] = {"failure": kind.value}
        if permanent:
            payload["permanent"] = True
        if signals:
            payload.update(signals)
        payload.setdefault("api", {"requests": self._runtime.budget.used_today, "billed": 0})
        return SearchResponse(
            query=query,
            results=[],
            engine=self.name,
            error=error,
            search_time_ms=(time.perf_counter() - started) * 1000,
            degraded=True,
            signals=payload,
        )

    # ── 抓取 ──────────────────────────────────────────────────────

    async def fetch(self, url: str, *, timeout: float | None = None) -> FetchOutcome:
        """单页抓取；失败只返回 ok=False，不抛错（调用方据此回落本地抓取）。"""
        runtime = self._runtime
        cfg = self.config
        if not self.available():
            return FetchOutcome(ok=False, url=url, error=self.unavailable_reason())
        target = _clip_url_chars(url)
        blocked = _local_url_blocker(target)
        if blocked:
            return FetchOutcome(ok=False, url=target, error=blocked)
        if not runtime.budget.try_acquire(
            today=datetime.now(timezone.utc).date(),
            daily_limit=cfg.daily_request_limit,
            scope=CURRENT_SEARCH_SCOPE.get(),
            scope_limit=cfg.conversation_request_limit,
        ):
            return FetchOutcome(ok=False, url=target, error="达到 API 请求预算")
        await runtime.bucket.acquire()
        try:
            resp = await runtime.client().post(FETCH_PATH, json={"url": target})
        except httpx.HTTPError as exc:
            return FetchOutcome(ok=False, url=target, error=f"{type(exc).__name__}: {exc}")
        if resp.status_code != 200:
            if resp.status_code == 402:
                runtime.breaker.trip(runtime.clock(), cfg.cooldown_on_402_seconds, "余额不足（402）")
            return FetchOutcome(ok=False, url=target, error=_error_message(resp), status=resp.status_code)
        try:
            payload = resp.json()
        except Exception as exc:
            return FetchOutcome(ok=False, url=target, error=f"响应不是 JSON: {exc}", status=200)
        content = str((payload or {}).get("content") or "")
        if not content:
            return FetchOutcome(ok=False, url=target, error="正文为空", status=200)
        runtime.breaker.record_success()
        await _record_cost(cfg.cost_per_request)
        return FetchOutcome(ok=True, url=target, title=str((payload or {}).get("title") or ""),
                            content=content, status=200)


#: 注册进统一引擎表；engine.get_engine 对未知名字会惰性 import 本模块。
ENGINE_REGISTRY["deepseek"] = DeepSeekSearchEngine
