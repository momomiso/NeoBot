"""联网搜索（[web_search] / [web_search.api] / 环境变量 Key）的热重载消费者。

为什么需要它：搜索通道的配置分散在两个进程级位置——
1. `deepseek_api.configure(...)`：Key、base_url、QPS、预算、熔断时长、重试次数；
2. `config_section.set_live_config(...)`：engines 顺序、fetch_mode、research_strategy、
   轮次/预算类参数（工具面每次调用现取）。
两者都在启动期写一次。没有消费者时，改 Key 或改 fetch_mode 必须重启，
而面板却可能显示「已生效」——这是本次实现里最后一个已知缺口。

接入方式（装配处一行）：`hot_reload_registry.register(WebSearchReloadConsumer(logger=...))`。
"""

from __future__ import annotations

from typing import Any

from neobot_contracts.ports.logging import Logger, NullLogger
from neobot_app.runtime.hot_reload_registry import ConfigConsumer


class WebSearchReloadConsumer(ConfigConsumer):
    """把 [web_search] 段的改动应用到进程级搜索配置与工具面活配置。"""

    #: 段本身 + env：Key 在环境变量里（换 Key / 充值后换账号都走这条）。
    config_paths: tuple[str, ...] = ("web_search", "env")

    def __init__(self, *, logger: Logger | None = None) -> None:
        self._logger = logger or NullLogger()

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def hot_reload_policies(self) -> tuple[Any, ...]:
        from neobot_app.config.hot_reload import HotReloadRule

        return (
            HotReloadRule(
                "web_search",
                True,
                "搜索通道的 Key/QPS/预算/fetch_mode 由进程级配置现取，重载后立即生效",
            ),
        )

    async def apply_config(self, config: Any) -> None:
        from neobot_app.web_search import deepseek_api
        from neobot_app.web_search.config_section import section_to_config_dict, set_live_config

        section = getattr(config, "web_search", None)
        deepseek_api.configure(getattr(section, "api", None))
        set_live_config(section_to_config_dict(section))

        applied = deepseek_api.get_config()
        # 只记「有没有 Key / 关键参数」，绝不打印 Key 本身。
        self._logger.info(
            "联网搜索配置已生效",
            key_configured=bool(applied.api_key),
            qps_limit=applied.qps_limit,
            fetch_mode=applied.fetch_mode,
            research_strategy=applied.research_strategy,
            engines=",".join(
                str(name) for name in (getattr(section, "engines", None) or ())
            ),
        )
