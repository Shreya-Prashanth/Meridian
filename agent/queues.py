from __future__ import annotations

import asyncio
from .models import Event, Action


class EventQueue:
    def __init__(self):
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    async def put(self, event: Event) -> None:
        await self._queue.put(event)

    async def get(self) -> Event:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def empty(self) -> bool:
        return self._queue.empty()


class ActionQueue:
    def __init__(self):
        self._queue: asyncio.Queue[Action] = asyncio.Queue()
        self.history: list[Action] = []

    async def put(self, action: Action) -> None:
        self.history.append(action)
        await self._queue.put(action)

    async def get(self) -> Action:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()
