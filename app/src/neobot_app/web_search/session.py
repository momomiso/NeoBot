"""SearchSession — 多轮检索-阅读工作流，支持多样性排序与去重。"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

from neobot_app.utils.ssrf import validate_public_url_async
from neobot_app.web_search.manager import SearchManager
from neobot_app.web_search.models import SearchResponse, SearchResult
from neobot_app.web_search.urls import strip_tracking_params, unwrap_tracking
from neobot_app.web_search.validate import (
    is_js_redirect_shell,
    looks_garbled,
    looks_like_empty_body,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchRound:
    """会话中的单轮搜索。"""

    round_num: int
    query: str
    response: SearchResponse
    read_indices: list[int] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _is_text_content(content_type: str) -> bool:
    """content-type 是否为可读文本（html/xml/json/plain 等）。"""
    ctype = (content_type or "").lower()
    if not ctype:
        return True  # 缺 content-type 时不武断拒绝，交给正文判据
    if ctype.startswith("text/"):
        return True
    return any(marker in ctype for marker in ("html", "xml", "json"))


class SearchSession:
    """管理多轮搜索与阅读对话。"""

    MAX_PAGE_SIZE = 500_000       # 500KB limit per page
    MAX_RESULTS_PER_SEARCH = 20   # cap results per search round

    READ_HEADERS: dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (compatible; NeoBot/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
    }

    RESEARCH_MODES: dict[str, list[str]] = {
        "encyclopedia": [
            "{topic} 百度百科",
            "{topic} 百科",
            "{topic} wiki",
        ],
        "community": [
            "{topic} 知乎",
            "{topic} 贴吧",
            "{topic} 论坛",
            "{topic} NGA",
        ],
        "news": [
            "{topic} 新闻",
            "{topic} 资讯",
            "{topic} 最新消息",
            "{topic} 情报",
        ],
        "official": [
            "{topic} 官网",
            "{topic} 官方网站",
        ],
        "video": [
            "{topic} bilibili",
            "{topic} 视频",
            "{topic} 评测",
        ],
        "academic": [
            "{topic} 论文",
            "{topic} 研究",
            "{topic} 文献",
        ],
        "raw": [],
    }

    #: operators 策略：一个 mode 一条查询（靠 site:/lang:/after: 表达原本要 3-4 条变体
    #: 才能覆盖的意图）。DeepSeek 按「请求」计费，4 次变体 = 4 倍成本，而运算符是免费的。
    #: 注意：官方只承诺 site: 子句之间用空格 + OR 连接，**不要发明括号嵌套**。
    #: 每项声明「这个角度默认靠哪些运算符表达」，由 _operator_queries 统一拼装：
    #: sites 是默认站点集（调用方显式传 sites 时**整体替换**，避免两串 OR 互相干扰）；
    #: extra 是自然语言补充（如「官网」）；after_days/lang 是时效与语种默认值
    #: （调用方显式传值优先）。
    RESEARCH_OPERATORS: dict[str, dict[str, Any]] = {
        "encyclopedia": {
            "sites": ["baike.baidu.com", "zh.wikipedia.org", "baike.sogou.com"]
        },
        "community": {
            "sites": ["zhihu.com", "tieba.baidu.com", "nga.cn", "v2ex.com"]
        },
        "news": {"after_days": 30, "lang": "zh"},
        "official": {"extra": "官网"},
        "video": {"sites": ["bilibili.com", "youtube.com"]},
        "academic": {"sites": ["arxiv.org", "scholar.google.com", "cnki.net"]},
    }

    #: news 模式默认时间窗（天）；调用方显式传 after 时以调用方为准。
    NEWS_WINDOW_DAYS = 30

    _DEFAULT_MODES = ("encyclopedia", "community")

    @staticmethod
    def _build_browser_channel(timeout_seconds: float):
        """按预算构造浏览器通道（进程级单例；不可用时通道内自行降级）。"""
        from neobot_app.web_search.browser_channel import get_browser_channel

        return get_browser_channel(timeout_seconds=timeout_seconds)

    def __init__(
        self,
        engines: Optional[list[str]] = None,
        max_rounds: int = 5,
        read_timeout: float = 30.0,
        *,
        browser_fallback: bool = True,
        browser_timeout_seconds: float = 8.0,
        engine_budgets: Optional[dict[str, float]] = None,
        fetch_mode: str = "api_first",
        api_fetch: Optional[Callable[[str], Awaitable[Any]]] = None,
        research_strategy: str = "operators",
    ) -> None:
        budgets = dict(engine_budgets or {})
        budgets.setdefault("duckduckgo", 15.0)
        self._manager = SearchManager(
            engines=engines,
            browser_fallback=browser_fallback,
            engine_budgets=budgets,
            browser_channel=self._build_browser_channel(browser_timeout_seconds)
            if browser_fallback
            else None,
        )
        self._max_rounds = max_rounds
        self._read_timeout = read_timeout
        #: 读页路径：api_first（先 Fetch API，失败回落本地）/ http_first / off
        self._fetch_mode = str(fetch_mode or "api_first").strip().lower()
        self._api_fetch = api_fetch
        #: 研究模式策略：operators（1 条查询）/ variants（旧的多查询，成本 ×N）
        self._research_strategy = str(research_strategy or "operators").strip().lower()
        self._rounds: list[SearchRound] = []
        self._results_index: dict[int, SearchResult] = {}
        self._read_urls: set[str] = set()

    async def aclose(self) -> None:
        """释放底层引擎的长连接资源。"""
        await self._manager.aclose()

    @property
    def rounds(self) -> list[SearchRound]:
        return list(self._rounds)

    @property
    def current_round(self) -> int:
        return len(self._rounds)

    @property
    def all_results(self) -> list[SearchResult]:
        """所有轮次的全部搜索结果，全局统一编号。"""
        return sorted(self._results_index.values(), key=lambda r: r.index)

    # ── 查询组装（运算符）─────────────────────────────────────────

    @staticmethod
    def compose_query(
        query: str,
        *,
        sites: Optional[list[str]] = None,
        lang: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        max_codepoints: int = 128,
    ) -> str:
        """把结构化参数拼成 DeepSeek 能识别的运算符查询。

        截断顺序很重要：**先保证运算符完整，再截主题**——运算符被砍掉等于参数静默失效，
        而主题被砍短只是模糊一点。
        """
        clauses: list[str] = []
        site_values = [str(s).strip() for s in (sites or ()) if str(s).strip()]
        if site_values:
            clauses.append(" OR ".join(f"site:{value}" for value in site_values))
        if lang:
            clauses.append(f"lang:{str(lang).strip()}")
        if after:
            clauses.append(f"after:{str(after).strip()}")
        if before:
            clauses.append(f"before:{str(before).strip()}")
        base = str(query or "").strip()
        if not clauses:
            return base[:max_codepoints]
        suffix = " " + " ".join(clauses)
        room = max(1, max_codepoints - len(suffix))
        return (base[:room] + suffix)[:max_codepoints]

    @staticmethod
    def _operator_queries(
        topic: str,
        modes: list[str],
        *,
        sites: Optional[list[str]] = None,
        lang: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> list[str]:
        """把 mode 展开成一条（或少数几条）运算符查询，而不是 3-4 条自然语言变体。"""
        from datetime import date, timedelta

        queries: list[str] = []
        for mode in modes:
            spec = SearchSession.RESEARCH_OPERATORS.get(str(mode).strip())
            if not spec:
                continue
            parts = [topic]
            if spec.get("extra"):
                parts.append(str(spec["extra"]))
            effective_sites = list(sites) if sites else list(spec.get("sites") or ())
            effective_after = after
            if not effective_after and spec.get("after_days"):
                days = int(spec["after_days"])
                effective_after = (date.today() - timedelta(days=days)).isoformat()
            effective_lang = lang or spec.get("lang")
            queries.append(
                SearchSession.compose_query(
                    " ".join(part for part in parts if part),
                    sites=effective_sites,
                    lang=effective_lang,
                    after=effective_after,
                    before=before,
                )
            )
        if not queries:
            queries.append(
                SearchSession.compose_query(topic, sites=sites, lang=lang, after=after, before=before)
            )
        return queries

    async def search(
        self,
        query: str,
        num_results: int = 10,
        *,
        sites: Optional[list[str]] = None,
        lang: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        need_content: Optional[bool] = None,
    ) -> SearchResponse:
        """执行新一轮搜索（可带 site:/lang:/after:/before: 运算符）。"""
        composed = self.compose_query(
            query, sites=sites, lang=lang, after=after, before=before
        )
        resp = await self._execute_search(composed, num_results, need_content=need_content)
        if resp.success:
            self._rounds.append(
                SearchRound(round_num=self.current_round + 1, query=query, response=resp)
            )
        return resp

    async def read(self, indices: list[int]) -> list[SearchResult]:
        """获取指定结果索引对应的完整页面内容。"""
        results_to_fetch = []
        for idx in indices:
            r = self._results_index.get(idx)
            if r is None:
                continue
            if not r.content_fetched:
                results_to_fetch.append(r)

        if not results_to_fetch:
            return [self._results_index[idx] for idx in indices if idx in self._results_index]

        # ⓪ Fetch API 优先：能拿到 Markdown 的页面就不再走本地 httpx（省一次反爬对抗）。
        # 失败/被拒/格式不对的条目留在 remaining 里，继续走原有本地抓取，行为不变。
        # 注意：requested 必须保留全量——_read_urls 要按「服务成功」登记，
        # 否则 API 已读的页面不会进已读集合，下一轮搜索又会把它们当新结果。
        requested = list(results_to_fetch)
        remaining = requested
        if self._fetch_mode == "api_first":
            remaining = await self._fetch_via_api(requested)

        if remaining:
            async with httpx.AsyncClient(
                timeout=self._read_timeout,
                follow_redirects=True,
                headers=dict(self.READ_HEADERS),
            ) as client:
                await self._fetch_all(client, remaining)

        for r in requested:
            if r.content_fetched:
                self._read_urls.add(r.url)

        if self._rounds:
            self._rounds[-1].read_indices.extend(indices)

        return [self._results_index[idx] for idx in indices if idx in self._results_index]

    async def _fetch_via_api(self, results: list[SearchResult]) -> list[SearchResult]:
        """尝试用 Fetch API 读页；返回仍需本地抓取的条目。

        与本地路径共用同一道 SSRF 闸门：**先把 URL 归一化并校验公网可达**，
        再决定要不要把它交给第三方。私网地址一律不发（含内网面板、文件服务器）。
        """
        handler = self._api_fetch
        if handler is None:
            from neobot_app.web_search import deepseek_api

            handler = deepseek_api.fetch_page

        remaining: list[SearchResult] = []
        for r in results:
            target = self._normalize_fetch_url(r.url)
            if target != r.url and not r.raw_url:
                r.raw_url = r.url
            if not await validate_public_url_async(target):
                r.content = "[已阻止] 该地址不是公网地址，未抓取"
                r.content_fetched = False
                continue
            try:
                outcome = await handler(target)
            except Exception:
                logger.debug("Fetch API 读页失败，回落本地抓取: %s", target, exc_info=True)
                remaining.append(r)
                continue
            content = getattr(outcome, "content", "") or ""
            if not getattr(outcome, "ok", False) or not content:
                remaining.append(r)
                continue
            r.content = content[: self.MAX_PAGE_SIZE]
            r.content_fetched = True
            r.content_format = "markdown"
            title = getattr(outcome, "title", "") or ""
            if title and not r.title:
                r.title = title
        return remaining

    async def read_single(self, index: int) -> Optional[SearchResult]:
        """按索引读取单条结果。"""
        results = await self.read([index])
        return results[0] if results else None

    def get_summary_for_agent(self) -> str:
        """为智能体生成会话状态摘要。"""
        lines = [f"===== 搜索会话 (共 {self.current_round} 轮) ====="]
        unread = [r for r in self._results_index.values() if not r.content_fetched]
        read_count = len(self._results_index) - len(unread)
        lines.append(f"结果: {len(self._results_index)} 条 (已读 {read_count}, 未读 {len(unread)})")
        lines.append("")

        for rd in self._rounds:
            lines.append(f"--- 第 {rd.round_num} 轮: '{rd.query}' ---")
            lines.append(f"引擎: {rd.response.engine}, 结果: {len(rd.response.results)} 条")
            for r in rd.response.results:
                status = "[R]" if r.content_fetched else "[ ]"
                lines.append(f"  [{r.index}] {status} {r.title}")
                lines.append(f"      {r.snippet[:100]}...")
            if rd.read_indices:
                lines.append(f"  已读取: {rd.read_indices}")
            lines.append("")
        return "\n".join(lines)

    async def research(
        self,
        topic: str,
        *,
        modes: list[str] | str | None = None,
        variants: list[str] | None = None,
        num_results: int = 10,
        variant_result_limit: int = 6,
        total_result_limit: int = 30,
        sites: Optional[list[str]] = None,
        lang: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        need_content: Optional[bool] = None,
    ) -> SearchResponse:
        """智能研究模式：从多个角度自动生成查询变体，合并去重，仅计 1 轮。

        Args:
            topic: 研究主题
            modes: 查询模式
            variants: 手动覆盖变体列表
            num_results: 主查询返回结果数
            variant_result_limit: 每个变体查询的最大返回结果数
            total_result_limit: 合并后总结果数上限
        """
        import asyncio

        if isinstance(modes, str):
            mode_list = [modes]
        elif modes is None:
            mode_list = list(self._DEFAULT_MODES)
        else:
            mode_list = list(modes)

        use_operators = (
            variants is None
            and self._research_strategy != "variants"
            and "raw" not in mode_list
        )
        if use_operators:
            # 运算符策略：每个 mode 一条查询，合并去重后当一轮。DeepSeek 按请求计费，
            # 这等于把原来 4-8 次请求压到 1-2 次，语义靠 site:/lang:/after: 表达。
            query_list = self._operator_queries(
                topic, mode_list,
                sites=sites, lang=lang, after=after, before=before,
            )
            variants = []
        else:
            if variants is None:
                resolved: list[str] = []
                for m in mode_list:
                    tmpls = self.RESEARCH_MODES.get(m)
                    if tmpls is None:
                        continue
                    for t in tmpls:
                        resolved.append(t.format(topic=topic))
                variants = resolved
            query_list = [topic] + [
                self.compose_query(v, sites=sites, lang=lang, after=after, before=before)
                for v in variants
            ]

        async def _query(q: str, n: int) -> SearchResponse:
            return await self._execute_search(q, n, need_content=need_content)

        variant_n = max(1, min(variant_result_limit, num_results))
        if use_operators:
            primary = await _query(query_list[0], num_results)
            variant_responses = [await _query(q, variant_n) for q in query_list[1:]]
        else:
            # 旧行为：主查询与变体并发（运营商通道下会放大 QPS，故默认不启用）
            primary = await _query(query_list[0], num_results)
            variant_responses = list(
                await asyncio.gather(*(_query(q, variant_n) for q in query_list[1:]))
            )

        if not primary.success:
            return primary

        seen_urls: set[str] = {r.url for r in primary.results} | self._read_urls

        for vp in variant_responses:
            if not vp.success:
                continue
            for r in vp.results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    primary.results.append(r)

        primary.results = self._rerank_by_diversity(primary.results)
        self._reindex_results(primary.results)

        # 总结果数受 total_result_limit 限制
        cap = max(1, total_result_limit)
        if len(primary.results) > cap:
            primary.results = primary.results[:cap]

        primary.total_estimated = len(primary.results)
        primary.query = topic

        mode_labels = mode_list if variants is not None else ["custom"]
        strategy_label = (
            f"operators×{len(query_list)}" if use_operators else f"+{len(variants)} variants"
        )
        self._rounds.append(
            SearchRound(
                round_num=self.current_round + 1,
                query=f"{topic} [modes: {','.join(mode_labels)}, {strategy_label}]",
                response=primary,
            )
        )
        return primary

    async def _execute_search(
        self, query: str, num_results: int, *, need_content: Optional[bool] = None
    ) -> SearchResponse:
        """核心搜索流水线：抓取 → 去重 → 多样性排序 → 编号。"""
        if self.current_round >= self._max_rounds:
            return SearchResponse(
                query=query,
                results=[],
                engine="session",
                error=f"已达到最大搜索轮次 ({self._max_rounds})",
                degraded=True,
            )

        fetch_count = min(num_results * 2, 30)
        resp = await self._manager.search_with_fallback(query, fetch_count, need_content=need_content)

        if not resp.success:
            return resp

        # Filter out already-read URLs
        unfiltered = len(resp.results)
        resp.results = [r for r in resp.results if r.url not in self._read_urls]
        filtered = unfiltered - len(resp.results)

        # Domain diversity reranking
        resp.results = self._rerank_by_diversity(resp.results)

        # Cap
        if len(resp.results) > self.MAX_RESULTS_PER_SEARCH:
            resp.results = resp.results[: self.MAX_RESULTS_PER_SEARCH]

        # Assign global indices
        self._reindex_results(resp.results)

        resp.total_estimated = len(resp.results)
        if filtered:
            resp.engine += f" (dedup: -{filtered})"
        return resp

    def _reindex_results(self, results: list[SearchResult]) -> None:
        start_idx = max(self._results_index.keys(), default=-1) + 1
        for i, r in enumerate(results):
            r.index = start_idx + i
            self._results_index[r.index] = r

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            netloc = urlparse(url).netloc
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc
        except Exception:
            return url

    @staticmethod
    def _rerank_by_diversity(results: list[SearchResult]) -> list[SearchResult]:
        """通过轮询交错重排结果，提升域名多样性。"""
        if len(results) <= 1:
            return list(results)

        domain_buckets: dict[str, list[SearchResult]] = defaultdict(list)
        domain_order: list[str] = []
        for r in results:
            domain = SearchSession._extract_domain(r.url)
            if domain not in domain_buckets:
                domain_order.append(domain)
            domain_buckets[domain].append(r)

        reranked: list[SearchResult] = []
        while len(reranked) < len(results):
            added = False
            for domain in domain_order:
                bucket = domain_buckets[domain]
                if bucket:
                    reranked.append(bucket.pop(0))
                    added = True
            if not added:
                break

        return reranked

    @staticmethod
    def _normalize_fetch_url(url: str) -> str:
        """抓取前先归一化：解追踪壳 + 去跟踪参数；失败保留原值。"""
        raw = str(url or "")
        if not raw:
            return raw
        try:
            return strip_tracking_params(unwrap_tracking(raw)) or raw
        except Exception:
            return raw

    async def _fetch_all(
        self,
        client: httpx.AsyncClient,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        import asyncio

        async def _fetch_one(r: SearchResult) -> SearchResult:
            target = self._normalize_fetch_url(r.url)
            if target != r.url and not r.raw_url:
                r.raw_url = r.url
            # 搜索结果里的 URL 来自第三方页面（可被 SEO/恶意内容影响），
            # 必须按 SSRF 处理：否则可被引导去探测内网、本机面板(9981)
            # 或文件服务器(8765)。校验同时覆盖 IP 字面量与 DNS 解析结果。
            if not await validate_public_url_async(target):
                r.content = "[已阻止] 该地址不是公网地址，未抓取"
                r.content_fetched = False
                return r
            try:
                resp = await client.get(target, headers=dict(self.READ_HEADERS))
                resp.raise_for_status()
                if not _is_text_content(resp.headers.get("content-type", "")):
                    r.content = f"[非文本内容] content-type={resp.headers.get('content-type')}"
                    r.content_fetched = False
                    return r
                body = resp.text or ""
                # 壳页 / 空正文 / 乱码都不算抓取成功（fix(9) D3 read 加固）。
                if is_js_redirect_shell(body):
                    r.content = "[跳过] 跳转壳页面，正文需浏览器渲染"
                    r.content_fetched = False
                    return r
                if looks_like_empty_body(body):
                    r.content = "[空内容] 页面正文为空"
                    r.content_fetched = False
                    return r
                if looks_garbled(body):
                    r.content = "[乱码] 正文不是可读文本"
                    r.content_fetched = False
                    return r
                r.content = body[: SearchSession.MAX_PAGE_SIZE]
                r.content_fetched = True
            except Exception as e:
                r.content = f"[获取失败] {e}"
                r.content_fetched = False
            return r

        return list(await asyncio.gather(*(_fetch_one(r) for r in results)))
