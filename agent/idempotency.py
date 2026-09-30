from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class IdempotencyRecord:
    status: str  # in_flight, completed
    result: Any = None


class IdempotencyManager:
    """Session-scoped idempotency. No cross-session persistence."""

    def __init__(self):
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    async def begin(self, key: str) -> tuple[bool, Any]:
        async with self._lock:
            record = self._records.get(key)
            if record is not None:
                return False, record.result
            self._records[key] = IdempotencyRecord("in_flight")
            return True, None

    async def complete(self, key: str, result: Any) -> None:
        async with self._lock:
            if key in self._records:
                self._records[key] = IdempotencyRecord("completed", result)

    async def forget_in_flight(self, key: str) -> None:
        async with self._lock:
            record = self._records.get(key)
            if record and record.status == "in_flight":
                del self._records[key]

    async def contains(self, key: str) -> bool:
        async with self._lock:
            return key in self._records
