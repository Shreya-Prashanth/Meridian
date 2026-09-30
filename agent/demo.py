from __future__ import annotations

import asyncio
from .clock import VirtualClock
from .runtime import AgentRuntime
from .replay import ReplayHarness
from .models import EventType


async def main():
    clock = VirtualClock()
    runtime = AgentRuntime(clock=clock)
    runtime.mock_tools.configure("search_flights", latency_ms=400)

    harness = ReplayHarness(runtime)
    await runtime.start()

    # Feed the first request and allow it to start.
    await clock.set(0)
    await runtime.submit(harness.event(0, EventType.USER_TEXT,
                                       {"text": "Book a flight to Delhi"}).event)
    await asyncio.sleep(0)
    await clock.set(100)
    await runtime.submit(harness.event(100, EventType.END_TURN).event)
    await asyncio.sleep(0)
    await clock.set(150)
    await asyncio.sleep(0)
    await clock.set(300)
    await runtime.submit(harness.event(300, EventType.INTERRUPTION,
                                       {"text": "Actually Mumbai"}).event)
    await asyncio.sleep(0)
    await clock.set(500)
    await asyncio.sleep(0)

    print("=== ACTIONS ===")
    for action in runtime.actions.history:
        print(action.to_dict())

    print("\n=== TRACE ===")
    print(runtime.trace.pretty())
    await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
