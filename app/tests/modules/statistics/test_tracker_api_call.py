"""按次用量记账（web_search_api）：成功才写、金额精确、token 恒为 0。"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from neobot_app.statistics.tracker import UsageTracker
from neobot_storage.engine import create_engine, sqlite_url
from neobot_storage.models import Base, ModelUsageRecord


@pytest_asyncio.fixture
async def session_factory(tmp_path):
    engine = create_engine(sqlite_url(tmp_path / "usage.db"))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def _fetch(session_factory) -> list[ModelUsageRecord]:
    async with session_factory() as session:
        result = await session.execute(select(ModelUsageRecord).order_by(ModelUsageRecord.id))
        return list(result.scalars().all())


async def test_record_api_call_writes_one_row(session_factory) -> None:
    tracker = UsageTracker(session_factory)
    await tracker.record_api_call(cost_cny=0.005)

    rows = await _fetch(session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.module_name == "web_search_api"
    assert row.model_name == "deepseek-search"
    assert row.provider_name == "DeepSeek"
    assert row.cost_cny == pytest.approx(0.005)
    assert row.cost_source == "builtin"
    assert (row.input_tokens, row.output_tokens) == (0, 0)


async def test_record_api_call_keeps_prices_independent(session_factory) -> None:
    tracker = UsageTracker(session_factory)
    await tracker.record_api_call(cost_cny=0.005)
    await tracker.record_api_call(cost_cny=0.0025)

    rows = await _fetch(session_factory)
    assert [round(row.cost_cny, 6) for row in rows] == [0.005, 0.0025]


async def test_unknown_module_is_skipped(session_factory) -> None:
    tracker = UsageTracker(session_factory)
    await tracker.record_api_call(module="not_a_module", cost_cny=9.0)
    assert await _fetch(session_factory) == []
