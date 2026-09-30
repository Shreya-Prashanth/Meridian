from __future__ import annotations

import asyncio


class VirtualClock:
    """Deterministic monotonic clock for tests and demos."""

    def __init__(self, start_ms: int = 0):
        self._now = start_ms
        self._condition = asyncio.Condition()

    @property
    def now_ms(self) -> int:
        return self._now

    async def set(self, value_ms: int) -> None:
        async with self._condition:
            if value_ms < self._now:
                raise ValueError("virtual clock cannot move backwards")
            self._now = value_ms
            self._condition.notify_all()

    async def advance(self, delta_ms: int) -> None:
        if delta_ms < 0:
            raise ValueError("delta must be non-negative")
        await self.set(self._now + delta_ms)

    async def wait_until(self, target_ms: int) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._now >= target_ms)
