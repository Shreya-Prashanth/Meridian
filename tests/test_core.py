import asyncio
import pytest

from agent.clock import VirtualClock
from agent.cancellation import CancellationManager
from agent.idempotency import IdempotencyManager
from agent.models import ToolCall
from agent.tools import MockToolEnvironment, build_default_registry, ToolExecutor


@pytest.mark.asyncio
async def test_idempotency_prevents_duplicate_state_change():
    clock = VirtualClock()
    env = MockToolEnvironment(clock)
    env.configure("book_flight", latency_ms=1)
    registry = build_default_registry(env)
    idem = IdempotencyManager()
    executor = ToolExecutor(registry, idem)
    cm = CancellationManager()
    token = await cm.register("a")
    call1 = ToolCall("a", "book_flight", {"flight_id": "AI101"}, 0, 0, "session:key", "fp")
    call2 = ToolCall("b", "book_flight", {"flight_id": "AI101"}, 0, 0, "session:key", "fp")
    task1 = asyncio.create_task(executor.execute(call1, token))
    await asyncio.sleep(0)
    await clock.set(1)
    result1 = await task1
    result2 = await executor.execute(call2, token)
    assert len(env.bookings) == 1
    assert result1.success and result2.success
    assert result2.output["idempotent_replay"] is True


@pytest.mark.asyncio
async def test_epoch_invalidation():
    cm = CancellationManager()
    e0 = await cm.current_epoch()
    await cm.advance_epoch()
    e1 = await cm.current_epoch()
    assert e1 == e0 + 1
    assert not await cm.is_current(e0)
    assert await cm.is_current(e1)


@pytest.mark.asyncio
async def test_state_version_is_monotonic():
    from agent.state import StateManager
    s = StateManager()
    a = await s.snapshot()
    b = await s.update_slot("destination", "Delhi")
    c = await s.update_slot("passengers", 3)
    assert a.version < b.version < c.version
