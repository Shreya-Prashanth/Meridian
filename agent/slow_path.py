from __future__ import annotations

import asyncio
from typing import Any
from .models import *
from .llm import LLMProvider
from .state import StateManager
from .tools import ToolRegistry, ToolExecutor
from .cancellation import CancellationManager
from .fast_path import FastPath
from .multimodal import MultimodalProcessor


class SlowPath:
    def __init__(self, state: StateManager, llm: LLMProvider, registry: ToolRegistry,
                 executor: ToolExecutor, cancellation: CancellationManager,
                 fast: FastPath, multimodal: MultimodalProcessor,
                 emit_action, trace, now):
        self.state = state
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.cancellation = cancellation
        self.fast = fast
        self.multimodal = multimodal
        self.emit_action = emit_action
        self.trace = trace
        self.now = now

    async def ingest_text(self, text: str) -> None:
        state = await self.state.snapshot()
        intent = await self.llm.extract_intent(text, state)
        slots = await self.llm.extract_slots(text, state)
        await self.state.set_intent_slots(intent, slots, merge=True)

    async def ingest_audio(self, payload: dict[str, Any]) -> None:
        obs = await self.multimodal.audio(payload)
        if obs.ambiguous:
            state = await self.state.snapshot()
            await self.fast.clarification(state, "I couldn’t reliably understand the audio. Could you repeat that?")
            return
        await self.ingest_text(obs.data["text"])

    async def ingest_image(self, payload: dict[str, Any]) -> None:
        obs = await self.multimodal.image(payload)
        if obs.ambiguous:
            state = await self.state.snapshot()
            await self.fast.clarification(
                state, "I can see multiple plausible items in that frame. Which one should I use?"
            )
            return
        await self.state.set_intent_slots("manual_lookup", {"frame": obs.data}, merge=True)

    async def plan_and_execute(self) -> None:
        state = await self.state.snapshot()
        tools = [self.registry.get(n) for n in self.registry.names()]
        plans = await self.llm.select_tools(state, tools)

        if not plans:
            if state.intent == "book_flight" and "destination" not in state.slots:
                await self.fast.clarification(state, "Which destination should I use?")
            elif state.intent == "create_ticket" and "subject" not in state.slots:
                await self.fast.clarification(state, "What should the ticket be about?")
            else:
                await self.fast.clarification(state, "What would you like me to do?")
            return

        # Concurrent read-only plans are safe. State-changing plans are serialized by the
        # idempotency manager and are never launched twice for the same key.
        tasks = []
        for plan in plans:
            tasks.append(asyncio.create_task(self._start_tool(plan)))
        if tasks:
            await asyncio.gather(*tasks)

    async def _start_tool(self, plan: dict[str, Any]) -> None:
        state = await self.state.snapshot()
        tool_name = plan["tool"]
        spec = self.registry.get(tool_name)
        args = dict(plan.get("arguments", {}))

        epoch = await self.cancellation.current_epoch()
        call_id = f"call-{self.now()}-{len(state.pending_calls)+1}"
        idem_key = None
        if spec.idempotency_required:
            idem_key = f"{state.session_id}:{tool_name}:{ToolCall.fingerprint(state.intent, args)}"

        call = ToolCall(
            call_id=call_id,
            tool=tool_name,
            arguments=args,
            state_version=state.version,
            epoch=epoch,
            idempotency_key=idem_key,
            intent_fingerprint=ToolCall.fingerprint(state.intent, state.slots),
        )
        token = await self.cancellation.register(call_id)
        await self.state.add_pending(call_id)
        await self.emit_action(Action.make(
            self.now(), ActionType.TOOL_CALL,
            {"call_id": call_id, "tool": tool_name, "arguments": args},
            state.version,
        ))
        self.trace.record(
            timestamp_ms=self.now(), kind="tool_started",
            call_id=call_id, tool=tool_name, state_version=state.version
        )

        task = asyncio.create_task(self.executor.execute(call, token))
        try:
            result = await task
        except asyncio.CancelledError:
            result = ToolResult(call_id, tool_name, False, {}, call.state_version,
                                call.epoch, error="cancelled", cancelled=True)

        current = await self.state.snapshot()
        current_epoch = await self.cancellation.current_epoch()
        stale = (
            result.state_version != current.version
            or result.epoch != current_epoch
            or result.call_id not in current.pending_calls
            and not result.cancelled
        )
        if stale:
            result.stale = True
            self.trace.record(
                timestamp_ms=self.now(), kind="stale_result_rejected",
                call_id=call_id, tool=tool_name,
                state_version=current.version, result=result.output,
                note=f"result_version={result.state_version}, current={current.version}; "
                     f"result_epoch={result.epoch}, current_epoch={current_epoch}"
            )
            await self.state.remove_pending(call_id)
            await self.cancellation.unregister(call_id)
            return

        await self.state.remove_pending(call_id, completed=result.success)
        await self.cancellation.unregister(call_id)

        if result.cancelled:
            self.trace.record(
                timestamp_ms=self.now(), kind="tool_cancelled",
                call_id=call_id, tool=tool_name, cancellation=True,
                state_version=current.version
            )
            return

        if not result.success:
            self.trace.record(
                timestamp_ms=self.now(), kind="tool_failed",
                call_id=call_id, tool=tool_name,
                state_version=current.version, result=result.error
            )
            await self.fast.progress(current, f"{tool_name} failed safely; I can retry it.")
            return

        self.trace.record(
            timestamp_ms=self.now(), kind="tool_result_accepted",
            call_id=call_id, tool=tool_name,
            state_version=current.version, result=result.output
        )

        # Tool result may cause a chain. Flight search -> final recommendation is read-only.
        if tool_name == "search_flights":
            flights = result.output.get("flights", [])
            if flights:
                chosen = flights[0]
                new_state = await self.state.set_intent_slots(
                    None, {"selected_flight": chosen}, merge=True
                )
                await self.emit_action(Action.make(
                    self.now(), ActionType.STATE_SNAPSHOT,
                    {"state": {
                        "version": new_state.version,
                        "intent": new_state.intent,
                        "slots": dict(new_state.slots),
                    }},
                    new_state.version,
                ))
                response = f"I found {len(flights)} flights to {chosen['destination']}. The first option is {chosen['flight_id']}."
                await self.emit_action(Action.make(
                    self.now(), ActionType.FINAL, {"text": response}, new_state.version
                ))
                self.trace.record(
                    timestamp_ms=self.now(), kind="final_response",
                    state_version=new_state.version, final_response=response
                )
        elif tool_name == "lookup_manual":
            if result.output.get("ambiguous"):
                await self.fast.clarification(
                    current, "The frame is ambiguous. Please point the camera at one item."
                )
            else:
                response = result.output.get("answer", "I found the relevant manual section.")
                await self.emit_action(Action.make(
                    self.now(), ActionType.FINAL, {"text": response}, current.version
                ))
        elif tool_name == "create_ticket":
            response = f"Ticket {result.output.get('ticket_id', 'created')} created."
            await self.emit_action(Action.make(
                self.now(), ActionType.FINAL, {"text": response}, current.version
            ))
        elif tool_name == "book_flight":
            response = f"Booking {result.output.get('booking_id', 'created')} completed."
            await self.emit_action(Action.make(
                self.now(), ActionType.FINAL, {"text": response}, current.version
            ))
