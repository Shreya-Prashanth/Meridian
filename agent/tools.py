from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from .models import ToolSpec, ToolCall, ToolResult
from .clock import VirtualClock
from .idempotency import IdempotencyManager
from .cancellation import CancellationToken


ToolFunction = Callable[[dict[str, Any], CancellationToken], Awaitable[dict[str, Any]]]


class ToolRegistry:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._functions: dict[str, ToolFunction] = {}

    def register(self, spec: ToolSpec, function: ToolFunction) -> None:
        self._specs[spec.name] = spec
        self._functions[spec.name] = function

    def register_manifest(self, manifest: dict[str, Any],
                          implementations: dict[str, ToolFunction]) -> None:
        for item in manifest.get("tools", []):
            spec = ToolSpec(
                name=item["name"],
                description=item.get("description", ""),
                input_schema=item.get("input_schema", {}),
                read_only=bool(item.get("read_only", True)),
                cancellable=bool(item.get("cancellable", True)),
                idempotency_required=bool(item.get("idempotency_required", False)),
                state_effects=tuple(item.get("state_effects", [])),
            )
            fn = implementations.get(spec.name)
            if fn is None:
                raise ValueError(f"No implementation registered for manifest tool {spec.name}")
            self.register(spec, fn)

    def get(self, name: str) -> ToolSpec:
        return self._specs[name]

    def function(self, name: str) -> ToolFunction:
        return self._functions[name]

    def names(self) -> list[str]:
        return sorted(self._specs)


class MockToolEnvironment:
    def __init__(self, clock: VirtualClock):
        self.clock = clock
        self.latency_ms: dict[str, int] = {}
        self.failures: dict[str, int] = {}
        self.invocations: list[tuple[str, dict[str, Any]]] = []
        self.bookings: list[dict[str, Any]] = []
        self.tickets: list[dict[str, Any]] = []

    def configure(self, tool: str, latency_ms: int | None = None,
                  failures: int | None = None) -> None:
        if latency_ms is not None:
            self.latency_ms[tool] = latency_ms
        if failures is not None:
            self.failures[tool] = failures

    async def _wait(self, tool: str, token: CancellationToken) -> None:
        target = self.clock.now_ms + self.latency_ms.get(tool, 100)
        while self.clock.now_ms < target:
            if token.cancelled():
                raise asyncio.CancelledError()
            await self.clock.wait_until(target)

    async def _common(self, tool: str, args: dict[str, Any], token: CancellationToken) -> None:
        self.invocations.append((tool, dict(args)))
        await self._wait(tool, token)
        if token.cancelled():
            raise asyncio.CancelledError()
        remaining = self.failures.get(tool, 0)
        if remaining:
            self.failures[tool] = remaining - 1
            raise RuntimeError(f"deterministic injected failure for {tool}")

    async def search_flights(self, args, token):
        await self._common("search_flights", args, token)
        destination = args.get("destination")
        origin = args.get("origin", "Bangalore")
        return {
            "flights": [
                {"flight_id": "AI101", "origin": origin, "destination": destination,
                 "price": 6200, "currency": "INR"},
                {"flight_id": "6E202", "origin": origin, "destination": destination,
                 "price": 5700, "currency": "INR"},
            ]
        }

    async def book_flight(self, args, token):
        await self._common("book_flight", args, token)
        record = {"booking_id": "BK-001", **args}
        self.bookings.append(record)
        return record

    async def create_ticket(self, args, token):
        await self._common("create_ticket", args, token)
        record = {"ticket_id": "TCK-001", **args}
        self.tickets.append(record)
        return record

    async def lookup_manual(self, args, token):
        await self._common("lookup_manual", args, token)
        frame = args.get("frame", {})
        label = frame.get("label") if isinstance(frame, dict) else None
        if not label:
            return {"ambiguous": True, "candidates": ["valve", "switch"]}
        return {"ambiguous": False, "label": label, "answer": f"Manual section for {label}"}

    def implementations(self) -> dict[str, ToolFunction]:
        return {
            "search_flights": self.search_flights,
            "book_flight": self.book_flight,
            "create_ticket": self.create_ticket,
            "lookup_manual": self.lookup_manual,
        }


def build_default_registry(env: MockToolEnvironment) -> ToolRegistry:
    import json
    from pathlib import Path
    manifest_path = Path(__file__).with_name("tools_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = ToolRegistry()
    registry.register_manifest(manifest, env.implementations())
    return registry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, idem: IdempotencyManager):
        self.registry = registry
        self.idem = idem

    async def execute(self, call: ToolCall, token: CancellationToken) -> ToolResult:
        spec = self.registry.get(call.tool)
        key = call.idempotency_key

        if spec.idempotency_required:
            if not key:
                return ToolResult(call.call_id, call.tool, False, {},
                                   call.state_version, call.epoch,
                                   error="missing_idempotency_key")
            allowed, previous = await self.idem.begin(key)
            if not allowed:
                return ToolResult(call.call_id, call.tool, True,
                                   {"idempotent_replay": True, "previous_result": previous},
                                   call.state_version, call.epoch)

        try:
            output = await self.registry.function(call.tool)(call.arguments, token)
            if spec.idempotency_required and key:
                await self.idem.complete(key, output)
            return ToolResult(call.call_id, call.tool, True, output,
                              call.state_version, call.epoch)
        except asyncio.CancelledError:
            if spec.idempotency_required and key:
                await self.idem.forget_in_flight(key)
            return ToolResult(call.call_id, call.tool, False, {},
                              call.state_version, call.epoch,
                              error="cancelled", cancelled=True)
        except Exception as exc:
            if spec.idempotency_required and key:
                await self.idem.forget_in_flight(key)
            return ToolResult(call.call_id, call.tool, False, {},
                              call.state_version, call.epoch,
                              error=str(exc))
