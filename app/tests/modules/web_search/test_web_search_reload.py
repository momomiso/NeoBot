"""联网搜索的热重载：消费者 + 活配置（改 Key / fetch_mode 不重启即生效）。

不变量：
1. 应用新配置后，`deepseek_api.get_config()` 立即反映新 Key / QPS / 预算；
2. `config_section.current_config()` 立即反映 engines / fetch_mode / research_strategy；
3. 工具面（WebTools）在**下一次调用**就用上新配置（构造期快照不再生效）；
4. 消费者声明 `web_search` 为可热重载，注册表能按路径分发给它。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from neobot_app.config.hot_reload import classify, unregister_rule
from neobot_app.config.schemas.bot import WebSearchApiConfig, WebSearchConfig
from neobot_app.runtime.hot_reload_registry import HotReloadRegistry
from neobot_app.runtime.web_search_reload import WebSearchReloadConsumer
from neobot_app.web_search import config_section, deepseek_api


@pytest.fixture(autouse=True)
def _clean_runtime() -> Any:
    deepseek_api.reset_runtime()
    config_section.set_live_config({})
    yield
    deepseek_api.reset_runtime()
    config_section.set_live_config({})
    unregister_rule("web_search")


def _config(**api_kw: Any) -> SimpleNamespace:
    return SimpleNamespace(
        web_search=WebSearchConfig(
            engines=["bing"],
            api=WebSearchApiConfig(**api_kw),
        )
    )


async def test_apply_config_updates_key_qps_and_live_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DeepSeekSearch_APIKey", "sk-live-key")
    consumer = WebSearchReloadConsumer()

    await consumer.apply_config(
        _config(qps_limit=2.0, fetch_mode="off", research_strategy="variants", daily_request_limit=7)
    )

    applied = deepseek_api.get_config()
    assert applied.api_key == "sk-live-key"
    assert applied.qps_limit == 2.0
    assert applied.daily_request_limit == 7

    live = config_section.current_config()
    assert live["engines"] == ["bing"]
    assert live["fetch_mode"] == "off"
    assert live["research_strategy"] == "variants"


async def test_key_rotation_takes_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer = WebSearchReloadConsumer()
    monkeypatch.setenv("DeepSeekSearch_APIKey", "sk-first")
    await consumer.apply_config(_config())
    assert deepseek_api.get_config().api_key == "sk-first"

    monkeypatch.setenv("DeepSeekSearch_APIKey", "sk-second")
    await consumer.apply_config(_config())
    assert deepseek_api.get_config().api_key == "sk-second"


async def test_consumer_declares_web_search_hot_reloadable() -> None:
    registry = HotReloadRegistry()
    registry.register(WebSearchReloadConsumer())

    hot, reason = classify(("web_search", "api", "fetch_mode"))
    assert hot is True, reason
    assert registry.consumers_for(["web_search.api.fetch_mode"])


async def test_registry_apply_dispatches_to_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DeepSeekSearch_APIKey", "sk-registry")
    registry = HotReloadRegistry()
    registry.register(WebSearchReloadConsumer())

    report = await registry.apply(_config(fetch_mode="http_first"), ["web_search.api.fetch_mode"])

    assert "web_search" in report.applied
    assert config_section.current_config()["fetch_mode"] == "http_first"


async def test_tools_use_live_config_on_next_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from neobot_app.agent_tools.web import WebTools

    captured: list[dict[str, Any]] = []

    class Session:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)
            self.engines = kwargs.get("engines")

        async def search(self, query: str, num_results: int = 10, **kwargs: Any) -> Any:
            from neobot_app.web_search.models import SearchResponse, SearchResult

            return SearchResponse(
                query=query,
                results=[SearchResult(index=0, title="t", url="https://example.com/a", snippet="s", engine="x")],
                engine="x",
            )

    config_section.set_live_config({"engines": ["bing"], "fetch_mode": "api_first"})
    tools = WebTools(web_search_config=config_section.current_config, search_factory=Session)
    await tools._search({"queries": ["a"]})
    assert captured[-1]["engines"] == ["bing"]
    assert tools.fetch_mode == "api_first"

    # 热重载：改配置后不重建 WebTools
    config_section.set_live_config({"engines": ["duckduckgo"], "fetch_mode": "off"})
    await tools._search({"queries": ["b"]})
    assert captured[-1]["engines"] == ["duckduckgo"]
    assert tools.fetch_mode == "off"


async def test_reconfigure_closes_previous_client() -> None:
    """反复重载不能把旧 client 攒着：configure 会排一个关闭任务。"""
    import asyncio

    old_client = deepseek_api.get_runtime().client()

    deepseek_api.configure(WebSearchApiConfig(), api_key="sk-next")
    await asyncio.sleep(0)

    assert old_client.is_closed
    assert deepseek_api.get_runtime().config.api_key == "sk-next"


async def test_missing_section_falls_back_to_defaults() -> None:
    consumer = WebSearchReloadConsumer()
    await consumer.apply_config(SimpleNamespace())

    applied = deepseek_api.get_config()
    assert applied.enabled is True
    assert applied.fetch_mode == "api_first"
    assert config_section.current_config() == {}
