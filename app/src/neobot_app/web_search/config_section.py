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
    return result


def session_kwargs(config_dict: dict[str, Any]) -> dict[str, Any]:
    """只挑 SearchSession 认识的键（它不接受 preview_pages_limit 之类）。"""
    allowed = (
        "engines",
        "max_rounds",
        "browser_fallback",
        "browser_timeout_seconds",
        "engine_budgets",
    )
    return {key: config_dict[key] for key in allowed if key in config_dict}
