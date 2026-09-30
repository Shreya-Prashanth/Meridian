from __future__ import annotations

from typing import Awaitable, Callable
from .models import ActionType, Event, EventType, Action, StateView


class FastPath:
    """Zero-wait conversational responses. No LLM or tool invocation occurs here."""

    def __init__(self, emit: Callable[[Action], Awaitable[None]], now: Callable[[], int]):
        self.emit = emit
        self.now = now

    async def acknowledge(self, state: StateView, text: str) -> None:
        await self.emit(Action.make(
            self.now(), ActionType.ACK, {"text": text}, state.version
        ))

    async def interruption(self, state: StateView) -> None:
        await self.emit(Action.make(
            self.now(), ActionType.ACK,
            {"text": "Got it — I’m switching to your latest request."},
            state.version,
        ))

    async def progress(self, state: StateView, text: str) -> None:
        await self.emit(Action.make(
            self.now(), ActionType.PROGRESS, {"text": text}, state.version
        ))

    async def clarification(self, state: StateView, question: str) -> None:
        await self.emit(Action.make(
            self.now(), ActionType.CLARIFICATION, {"question": question}, state.version
        ))
