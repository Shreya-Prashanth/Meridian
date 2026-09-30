from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class CancellationToken:
    call_id: str
    epoch: int
    event: asyncio.Event

    def cancelled(self) -> bool:
        return self.event.is_set()

    def cancel(self) -> None:
        self.event.set()


class CancellationManager:
    """Epoch-based invalidation plus best-effort asyncio cancellation."""

    def __init__(self):
        self._epoch = 0
        self._lock = asyncio.Lock()
        self._tokens: dict[str, CancellationToken] = {}

    async def current_epoch(self) -> int:
        async with self._lock:
            return self._epoch

    async def advance_epoch(self) -> int:
        async with self._lock:
            self._epoch += 1
            for token in self._tokens.values():
                token.cancel()
            return self._epoch

    async def register(self, call_id: str) -> CancellationToken:
        async with self._lock:
            token = CancellationToken(call_id, self._epoch, asyncio.Event())
            self._tokens[call_id] = token
            return token

    async def cancel(self, call_id: str) -> None:
        async with self._lock:
            token = self._tokens.get(call_id)
            if token:
                token.cancel()

    async def unregister(self, call_id: str) -> None:
        async with self._lock:
            self._tokens.pop(call_id, None)

    async def is_current(self, epoch: int) -> bool:
        async with self._lock:
            return epoch == self._epoch
