from __future__ import annotations

import asyncio
from typing import Any
from .models import Event, EventType, Action, ActionType
from .queues import EventQueue, ActionQueue
from .state import StateManager
from .cancellation import CancellationManager
from .idempotency import IdempotencyManager
from .tools import ToolRegistry, ToolExecutor, MockToolEnvironment, build_default_registry
from .llm import LLMProvider, DeterministicLLMProvider
from .fast_path import FastPath
from .slow_path import SlowPath
from .multimodal import MultimodalProcessor
from .protocol import ProtocolValidator, ProtocolError
from .trace import TraceLogger
from .clock import VirtualClock


class AgentRuntime:
    """
    Main event/action orchestrator.

    Correctness model:
      state version + operation epoch + call_id + idempotency key.

    asyncio cancellation is best-effort only. A result is accepted only when
    its operation context still matches the current session state.
    """

    def __init__(self, *, clock: VirtualClock | None = None,
                 llm: LLMProvider | None = None,
                 registry: ToolRegistry | None = None,
                 session_id: str = "session"):
        self.clock = clock or VirtualClock()
        self.events = EventQueue()
        self.actions = ActionQueue()
        self.state = StateManager(session_id=session_id)
        self.cancellation = CancellationManager()
        self.idempotency = IdempotencyManager()
        self.trace = TraceLogger()
        self.validator = ProtocolValidator()
        self.llm = llm or DeterministicLLMProvider()
        if registry is None:
            env = MockToolEnvironment(self.clock)
            registry = build_default_registry(env)
            self.mock_tools = env
        else:
            self.mock_tools = None
        self.registry = registry
        self.executor = ToolExecutor(self.registry, self.idempotency)
        self.fast = FastPath(self._emit, lambda: self.clock.now_ms)
        self.slow = SlowPath(
            self.state, self.llm, self.registry, self.executor,
            self.cancellation, self.fast, MultimodalProcessor(),
            self._emit, self.trace, lambda: self.clock.now_ms
        )
        self._loop_task: asyncio.Task | None = None
        self._turn_generation = 0
        self._pending_slow_tasks: set[asyncio.Task] = set()
        self._pending_input_task: asyncio.Task | None = None
        self._action_observers = []

    def add_action_observer(self, observer) -> None:
        """Observe emitted actions for UI/telemetry without affecting the agent."""
        self._action_observers.append(observer)

    def remove_action_observer(self, observer) -> None:
        if observer in self._action_observers:
            self._action_observers.remove(observer)

    async def _emit(self, action: Action) -> None:
        try:
            self.validator.validate_action(action)
        except ProtocolError as exc:
            fallback = Action.make(
                self.clock.now_ms, ActionType.ERROR,
                {"error": str(exc)}, action.state_version
            )
            self.actions.history.append(fallback)
            await self.actions.put(fallback)
            self.trace.record(
                timestamp_ms=self.clock.now_ms, kind="protocol_error",
                state_version=action.state_version, note=str(exc)
            )
            return
        await self.actions.put(action)
        self.trace.record(
            timestamp_ms=action.timestamp_ms, kind="action",
            state_version=action.state_version, action=action.to_dict()
        )
        for observer in tuple(self._action_observers):
            try:
                observer(action)
            except Exception:
                # Presentation/telemetry observers must never affect agent correctness.
                pass

    async def submit(self, event: Event) -> None:
        await self.events.put(event)

    async def start(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        for task in list(self._pending_slow_tasks):
            task.cancel()
        if self._pending_slow_tasks:
            await asyncio.gather(*self._pending_slow_tasks, return_exceptions=True)

    async def drain(self) -> None:
        while not self.events.empty():
            await asyncio.sleep(0)
        if self._pending_slow_tasks:
            await asyncio.gather(*list(self._pending_slow_tasks), return_exceptions=True)

    async def _run(self) -> None:
        while True:
            event = await self.events.get()
            try:
                await self._handle(event)
            finally:
                self.events.task_done()

    def _track(self, task: asyncio.Task) -> None:
        self._pending_slow_tasks.add(task)
        task.add_done_callback(self._pending_slow_tasks.discard)

    async def _handle(self, event: Event) -> None:
        state = await self.state.snapshot()
        self.trace.record(
            timestamp_ms=event.timestamp_ms, kind="event",
            event={"event_id": event.event_id, "type": event.type.value, "payload": dict(event.payload)},
            state=state.__dict__, state_version=state.version
        )

        if event.type == EventType.INTERRUPTION:
            await self._handle_interruption(event)
            return

        if event.type == EventType.SESSION_RESET:
            await self.cancellation.advance_epoch()
            self._turn_generation += 1
            await self.state.reset()
            state = await self.state.snapshot()
            await self._emit(Action.make(
                self.clock.now_ms, ActionType.STATE_SNAPSHOT,
                {"state": {
                    "version": state.version,
                    "intent": state.intent,
                    "slots": dict(state.slots)
                }},
                state.version
            ))
            return

        if event.type == EventType.TOOL_MANIFEST:
            # Manifest extension point. The official harness can send new tools.
            manifest = event.payload.get("manifest")
            if manifest:
                implementations = event.payload.get("implementations", {})
                # Serializable manifests can register externally supplied adapters
                # through a separately configured implementation map.
                self.trace.record(
                    timestamp_ms=self.clock.now_ms, kind="manifest_received",
                    note=f"{len(manifest.get('tools', []))} tool definitions"
                )
            return

        if event.type == EventType.USER_TEXT:
            text = str(event.payload.get("text", ""))
            await self.fast.acknowledge(state, "Got it — I’m working on that.")
            task = asyncio.create_task(self.slow.ingest_text(text))
            self._pending_input_task = task
            self._track(task)
            return

        if event.type == EventType.AUDIO:
            await self.fast.acknowledge(state, "I heard you. Let me work that out.")
            task = asyncio.create_task(self.slow.ingest_audio(dict(event.payload)))
            self._track(task)
            return

        if event.type == EventType.IMAGE_FRAME:
            await self.fast.acknowledge(state, "I’ll inspect that frame.")
            task = asyncio.create_task(self.slow.ingest_image(dict(event.payload)))
            self._track(task)
            return

        if event.type == EventType.END_TURN:
            generation = self._turn_generation
            input_task = self._pending_input_task
            async def plan_after_input():
                if input_task is not None:
                    await asyncio.gather(input_task, return_exceptions=True)
                if generation != self._turn_generation:
                    return
                await self.slow.plan_and_execute()
            task = asyncio.create_task(plan_after_input())
            self._track(task)
            return

        if event.type == EventType.TOOL_RESULT:
            # External harness result injection hook. The production executor normally
            # receives results directly, but this branch is useful for adapter harnesses.
            self.trace.record(
                timestamp_ms=self.clock.now_ms, kind="external_tool_result",
                call_id=event.payload.get("call_id"),
                tool=event.payload.get("tool"),
                result=event.payload.get("result"),
                state_version=state.version
            )

    async def _handle_interruption(self, event: Event) -> None:
        # Invalidate all old operations BEFORE ingesting the replacement intent.
        # Emit explicit cancellation actions as part of the protocol so both the
        # benchmark harness and the local demo can observe the interruption.
        old_epoch = await self.cancellation.current_epoch()
        state_before = await self.state.snapshot()
        pending_calls = sorted(state_before.pending_calls)
        new_epoch = await self.cancellation.advance_epoch()
        self._turn_generation += 1
        await self.fast.interruption(state_before)

        for call_id in pending_calls:
            await self._emit(Action.make(
                self.clock.now_ms, ActionType.CANCEL,
                {
                    "call_id": call_id,
                    "reason": "superseded_by_user_interruption",
                    "old_epoch": old_epoch,
                    "new_epoch": new_epoch,
                },
                state_before.version,
            ))

        # Update state first. The version bump is the correctness boundary.
        text = str(event.payload.get("text", ""))
        await self.slow.ingest_text(text)
        new_state = await self.state.snapshot()

        await self._emit(Action.make(
            self.clock.now_ms, ActionType.STATE_SNAPSHOT,
            {"state": {
                "version": new_state.version,
                "intent": new_state.intent,
                "slots": dict(new_state.slots)
            }},
            new_state.version
        ))
        self.trace.record(
            timestamp_ms=self.clock.now_ms, kind="interruption_replan",
            state_version=new_state.version, note=f"epoch={new_epoch}"
        )

        task = asyncio.create_task(self.slow.plan_and_execute())
        self._track(task)
