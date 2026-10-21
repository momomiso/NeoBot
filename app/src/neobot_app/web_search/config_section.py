"""[web_search] 配置段 -> 工具/会话参数的单一转换点。

为什么单独一个模块：这段逻辑原先只存在于 bootstrap 的 _web_search_config_dict，
而主 Agent 的工具面（agent_tools/runtime.py 里的 WebTools）从来没有拿到过配置，
于是 [web_search].engines 对主链是死配置（fix(9) F5 的同类问题）。现在由
bootstrap 与 skills 两处共用本函数，避免再出现「两套默认值」。
"""

from __future__ import annotations

from typing import Any

#: 出厂默认顺序：有 Key 走 DeepSeek，没有就自动跳到免费通道。
DEFAULT_ENGINES: tuple[str, ...] = ("deepseek", "bing", "duckduckgo")

#: 最近一次生效的 [web_search] 配置（由装配期与热重载消费者写入）。
#: 为什么要这个「活配置」：WebTools 在启动时被构造一次，若只持有构造期快照，
#: 改 engines / fetch_mode / research_strategy 都必须重启才生效 —— 热重载消费者
#: 只需刷新这里，工具面每次调用现取即可。
_LIVE_CONFIG: dict[str, Any] = {}


def set_live_config(config_dict: dict[str, Any]) -> None:
    """写入当前生效的 [web_search] 配置（装配期一次，热重载每次）。"""
    global _LIVE_CONFIG
    _LIVE_CONFIG = dict(config_dict or {})


def current_config() -> dict[str, Any]:
    """读取当前生效的配置；未写入时返回空 dict（调用方按默认值兜底）。"""
    return dict(_LIVE_CONFIG)


def section_to_config_dict(section: Any) -> dict[str, Any]:
    """把 [web_search] 段转成 WebSearchExecutor / SearchSession 认识的 dict。

    缺失段返回空 dict（老配置、测试替身）；字段缺省时按实测默认值补齐：
    duckduckgo 作为最后兜底给 15s（部署实测它慢），浏览器通道给 8s。
    """
    if section is None:
        return {}
    engines = getattr(section, "engines", None)
    result: dict[str, Any] = {
        "engines": [str(name) for name in engines] if engines else list(DEFAULT_ENGINES),
        "max_rounds": int(getattr(section, "max_search_rounds", None) or 5),
        "preview_pages_limit": int(getattr(section, "preview_pages_limit", None) or 30),
        "variant_result_limit": int(getattr(section, "variant_result_limit", None) or 6),
        "browser_fallback": bool(getattr(section, "browser_fallback", True)),
        "browser_timeout_seconds": float(
            getattr(section, "browser_timeout_seconds", None) or 8.0
        ),
    }
    budgets = getattr(section, "engine_timeout_seconds", None)
    if isinstance(budgets, dict) and budgets:
        result["engine_budgets"] = {str(key): float(value) for key, value in budgets.items()}
    # 读页路径来自 [web_search.api].fetch_mode（api_first / http_first / off）
    api_section = getattr(section, "api", None)
    fetch_mode = getattr(api_section, "fetch_mode", None)
    if fetch_mode:
        result["fetch_mode"] = str(fetch_mode)
    research_strategy = getattr(api_section, "research_strategy", None)
    if research_strategy:
        result["research_strategy"] = str(research_strategy)
    return result


def session_kwargs(config_dict: dict[str, Any]) -> dict[str, Any]:
    """只挑 SearchSession 认识的键（它不接受 preview_pages_limit 之类）。"""
    allowed = (
        "engines",
        "max_rounds",
        "browser_fallback",
        "browser_timeout_seconds",
        "engine_budgets",
        "fetch_mode",
        "research_strategy",
    )
    return {key: config_dict[key] for key in allowed if key in config_dict}
