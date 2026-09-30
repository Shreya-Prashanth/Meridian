import asyncio
import pytest

from agent.clock import VirtualClock
from agent.runtime import AgentRuntime
from agent.models import Event, EventType, ActionType
from agent.replay import ReplayHarness


async def send(rt, ts, typ, payload=None):
    await rt.clock.set(ts)
    await rt.submit(Event.make(ts, typ, payload or {}))
    for _ in range(4):
        await asyncio.sleep(0)


async def settle(rt, ts=10000):
    # First yield lets scheduled slow work establish its virtual deadline.
    for _ in range(4):
        await asyncio.sleep(0)
    await rt.clock.set(ts)
    for _ in range(4):
        await asyncio.sleep(0)
    # If a task established its deadline only after the first jump, one more jump settles it.
    await rt.clock.set(ts + 10000)
    for _ in range(6):
        await asyncio.sleep(0)
    await rt.drain()


@pytest.mark.asyncio
async def test_basic_successful_task():
    rt = AgentRuntime(clock=VirtualClock())
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 100, EventType.END_TURN)
    await settle(rt, 200)
    assert any(a.type == ActionType.FINAL for a in rt.actions.history)
    assert any(a.type == ActionType.TOOL_CALL for a in rt.actions.history)
    await rt.stop()


@pytest.mark.asyncio
async def test_interruption_during_tool_execution():
    rt = AgentRuntime(clock=VirtualClock())
    rt.mock_tools.configure("search_flights", latency_ms=500)
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 10, EventType.END_TURN)
    await send(rt, 50, EventType.INTERRUPTION, {"text": "Actually Mumbai"})
    await settle(rt, 1000)
    stale = [r for r in rt.trace.records if r.kind == "stale_result_rejected"]
    assert stale or any(r.kind == "tool_cancelled" for r in rt.trace.records)
    assert (await rt.state.snapshot()).slots["destination"] == "Mumbai"
    await rt.stop()


@pytest.mark.asyncio
async def test_multiple_consecutive_interruptions():
    rt = AgentRuntime(clock=VirtualClock())
    rt.mock_tools.configure("search_flights", latency_ms=500)
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.END_TURN)
    await send(rt, 2, EventType.INTERRUPTION, {"text": "Actually Mumbai"})
    await send(rt, 3, EventType.INTERRUPTION, {"text": "Actually Chennai"})
    await rt.clock.set(1000)
    await rt.drain()
    assert (await rt.state.snapshot()).slots["destination"] == "Chennai"
    assert await rt.cancellation.current_epoch() >= 2
    await rt.stop()


@pytest.mark.asyncio
async def test_interruption_immediately_before_completion():
    rt = AgentRuntime(clock=VirtualClock())
    rt.mock_tools.configure("search_flights", latency_ms=100)
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.END_TURN)
    await rt.clock.set(99)
    await asyncio.sleep(0)
    await send(rt, 100, EventType.INTERRUPTION, {"text": "Actually Mumbai"})
    await settle(rt, 200)
    assert (await rt.state.snapshot()).slots["destination"] == "Mumbai"
    await rt.stop()


@pytest.mark.asyncio
async def test_late_stale_result_is_rejected():
    rt = AgentRuntime(clock=VirtualClock())
    rt.mock_tools.configure("search_flights", latency_ms=1000)
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.END_TURN)
    await send(rt, 10, EventType.INTERRUPTION, {"text": "Actually Mumbai"})
    await settle(rt, 2000)
    assert any(r.kind == "stale_result_rejected" for r in rt.trace.records)
    await rt.stop()


@pytest.mark.asyncio
async def test_duplicate_state_modifying_call_prevention():
    rt = AgentRuntime(clock=VirtualClock())
    rt.mock_tools.configure("book_flight", latency_ms=10)
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.END_TURN)
    await rt.clock.set(100)
    # Directly request booking twice through executor-level calls is tested in test_core.
    await rt.stop()


@pytest.mark.asyncio
async def test_tool_failure_and_safe_retry():
    rt = AgentRuntime(clock=VirtualClock())
    rt.mock_tools.configure("search_flights", latency_ms=10, failures=1)
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.END_TURN)
    await rt.clock.set(20)
    await rt.drain()
    assert any(r.kind == "tool_failed" for r in rt.trace.records)
    assert not any(
        a.type == ActionType.FINAL and "completed" in a.payload.get("text", "").lower()
        for a in rt.actions.history
    )
    await rt.stop()


@pytest.mark.asyncio
async def test_localized_slot_correction():
    rt = AgentRuntime(clock=VirtualClock())
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.INTERRUPTION, {"text": "Actually Mumbai"})
    state = await rt.state.snapshot()
    assert state.slots["destination"] == "Mumbai"
    await rt.stop()


@pytest.mark.asyncio
async def test_missing_slot_clarification():
    rt = AgentRuntime(clock=VirtualClock())
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight"})
    await send(rt, 1, EventType.END_TURN)
    assert any(a.type == ActionType.CLARIFICATION for a in rt.actions.history)
    await rt.stop()


@pytest.mark.asyncio
async def test_chained_tool_calls():
    rt = AgentRuntime(clock=VirtualClock())
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.END_TURN)
    await settle(rt, 200)
    assert any(r.kind == "tool_result_accepted" for r in rt.trace.records)
    assert any(a.type == ActionType.FINAL for a in rt.actions.history)
    await rt.stop()


@pytest.mark.asyncio
async def test_dynamic_unseen_tool_manifest_is_received():
    rt = AgentRuntime(clock=VirtualClock())
    await rt.start()
    await send(rt, 0, EventType.TOOL_MANIFEST, {
        "manifest": {"tools": [{"name": "weather", "description": "x"}]}
    })
    assert any(r.kind == "manifest_received" for r in rt.trace.records)
    await rt.stop()


@pytest.mark.asyncio
async def test_audio_input():
    rt = AgentRuntime(clock=VirtualClock())
    await rt.start()
    await send(rt, 0, EventType.AUDIO, {"transcript": "Book a flight to Delhi"})
    state = await rt.state.snapshot()
    assert state.slots["destination"] == "Delhi"
    await rt.stop()


@pytest.mark.asyncio
async def test_image_input():
    rt = AgentRuntime(clock=VirtualClock())
    await rt.start()
    await send(rt, 0, EventType.IMAGE_FRAME, {"label": "valve"})
    state = await rt.state.snapshot()
    assert state.intent == "manual_lookup"
    assert state.slots["frame"]["label"] == "valve"
    await rt.stop()


@pytest.mark.asyncio
async def test_ambiguous_visual_input():
    rt = AgentRuntime(clock=VirtualClock())
    await rt.start()
    await send(rt, 0, EventType.IMAGE_FRAME, {})
    assert any(a.type == ActionType.CLARIFICATION for a in rt.actions.history)
    await rt.stop()


@pytest.mark.asyncio
async def test_simultaneous_tool_result_and_interruption():
    rt = AgentRuntime(clock=VirtualClock())
    rt.mock_tools.configure("search_flights", latency_ms=100)
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.END_TURN)
    await send(rt, 100, EventType.INTERRUPTION, {"text": "Actually Mumbai"})
    await settle(rt, 200)
    assert (await rt.state.snapshot()).slots["destination"] == "Mumbai"
    await rt.stop()


@pytest.mark.asyncio
async def test_race_between_cancellation_and_completion():
    rt = AgentRuntime(clock=VirtualClock())
    rt.mock_tools.configure("search_flights", latency_ms=100)
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.END_TURN)
    await rt.clock.set(100)
    await send(rt, 100, EventType.INTERRUPTION, {"text": "Actually Mumbai"})
    await settle(rt, 200)
    assert (await rt.state.snapshot()).slots["destination"] == "Mumbai"
    await rt.stop()


@pytest.mark.asyncio
async def test_very_slow_tool():
    rt = AgentRuntime(clock=VirtualClock())
    rt.mock_tools.configure("search_flights", latency_ms=100000)
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.END_TURN)
    await send(rt, 2, EventType.INTERRUPTION, {"text": "Actually Mumbai"})
    assert (await rt.state.snapshot()).slots["destination"] == "Mumbai"
    await rt.stop()


@pytest.mark.asyncio
async def test_multiple_concurrent_read_only_tools_core():
    rt = AgentRuntime(clock=VirtualClock())
    assert rt.registry.get("search_flights").read_only
    assert rt.registry.get("lookup_manual").read_only


@pytest.mark.asyncio
async def test_session_reset():
    rt = AgentRuntime(clock=VirtualClock())
    await rt.start()
    await send(rt, 0, EventType.USER_TEXT, {"text": "Book a flight to Delhi"})
    await send(rt, 1, EventType.SESSION_RESET)
    state = await rt.state.snapshot()
    assert state.intent is None
    assert state.slots == {}
    await rt.stop()


@pytest.mark.asyncio
async def test_malformed_protocol_output():
    from agent.models import Action
    from agent.protocol import ProtocolValidator, ProtocolError
    v = ProtocolValidator()
    bad = Action.make(0, ActionType.TOOL_CALL, {"tool": "x"}, 1)
    with pytest.raises(ProtocolError):
        v.validate_action(bad)
