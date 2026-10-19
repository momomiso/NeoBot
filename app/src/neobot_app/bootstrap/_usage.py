"""用量追踪与报告"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from neobot_app.statistics.billing import BillingService
from neobot_app.statistics.tracker import UsageTracker, initialize_usage_tracker
from neobot_app.statistics.reporter import UsageReportService


def build_usage_components(
    *,
    _engine: Any,
    logger_factory: Any,
    config: Any = None,
) -> dict[str, Any]:
    usage_session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    # 计价脚本服务（spec(4) Part A）：只在 [billing].enabled=true 时才会被真正求值。
    # 传入 ConfigProxy，改动 [billing] 段无需重启即可生效（Q13）。
    billing = BillingService(config=config, logger=logger_factory.get_logger("app.billing"))
    usage_tracker = UsageTracker(
        usage_session_factory,
        logger=logger_factory.get_logger("app.usage"),
        billing=billing,
    )
    initialize_usage_tracker(usage_tracker)

    # 搜索/抓取 API 的按次记账：web_search 不依赖 statistics，回调在装配期注入。
    # 未装配（纯单元测试）时 _cost_recorder 为 None，引擎照常工作、只是不落库。
    from neobot_app.web_search import deepseek_api

    async def _record_search_cost(cost: float) -> None:
        await usage_tracker.record_api_call(cost_cny=cost)

    deepseek_api.set_cost_recorder(_record_search_cost)

    report_service = UsageReportService(
        usage_session_factory,
        logger=logger_factory.get_logger("app.usage_report"),
    )
    return {
        "tracker": usage_tracker,
        "report_service": report_service,
        "billing": billing,
    }
