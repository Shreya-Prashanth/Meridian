from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable
from .models import StateSnapshot, StateView


class StateManager:
    """Single-writer state manager. Every mutation increments the version."""

    def __init__(self, session_id: str = "session"):
        self._state = StateSnapshot(session_id=session_id)
        self._lock = asyncio.Lock()

    async def snapshot(self) -> StateView:
        async with self._lock:
            return StateView.from_snapshot(self._state)

    async def mutate(self, mutator: Callable[[StateSnapshot], None],
                     *, bump_version: bool = True) -> StateView:
        async with self._lock:
            if bump_version:
                self._state.version += 1
            mutator(self._state)
            return StateView.from_snapshot(self._state)

    async def reset(self) -> StateView:
        async with self._lock:
            sid = self._state.session_id
            self._state = StateSnapshot(session_id=sid)
            self._state.version = 1
            return StateView.from_snapshot(self._state)

    async def set_intent_slots(self, intent: str | None, slots: dict[str, Any],
                               *, merge: bool = True) -> StateView:
        def mutate(s: StateSnapshot) -> None:
            if intent is not None:
                s.intent = intent
            if merge:
                s.slots.update(slots)
            else:
                s.slots = dict(slots)
        return await self.mutate(mutate)

    async def update_slot(self, name: str, value: Any) -> StateView:
        return await self.mutate(lambda s: s.slots.__setitem__(name, value))

    async def add_pending(self, call_id: str, *, bump_version: bool = False) -> StateView:
        return await self.mutate(lambda s: s.pending_calls.add(call_id), bump_version=bump_version)

    async def remove_pending(self, call_id: str, *, completed: bool = False,
                             bump_version: bool = False) -> StateView:
        def mutate(s: StateSnapshot) -> None:
            s.pending_calls.discard(call_id)
            if completed:
                s.completed_calls.add(call_id)
        return await self.mutate(mutate, bump_version=bump_version)
