from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable
from .models import Event
from .runtime import AgentRuntime
from .clock import VirtualClock


@dataclass
class ScheduledEvent:
    timestamp_ms: int
    event: Event


class ReplayHarness:
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self.clock = runtime.clock

    async def run(self, events: Iterable[ScheduledEvent]) -> AgentRuntime:
        await self.runtime.start()
        for item in sorted(events, key=lambda x: x.timestamp_ms):
            await self.clock.set(item.timestamp_ms)
            await self.runtime.submit(item.event)
            # Let the runtime consume this event without advancing wall-clock time.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        await self.runtime.drain()
        return self.runtime

    @staticmethod
    def event(ts: int, event_type, payload=None, event_id=None) -> ScheduledEvent:
        return ScheduledEvent(
            ts, Event.make(ts, event_type, payload, event_id=event_id)
        )
