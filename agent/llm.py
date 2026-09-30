from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Mapping
from .models import StateView, ToolSpec


class LLMProvider(ABC):
    @abstractmethod
    async def extract_intent(self, text: str, state: StateView) -> str | None: ...

    @abstractmethod
    async def extract_slots(self, text: str, state: StateView) -> dict[str, Any]: ...

    @abstractmethod
    async def select_tools(self, state: StateView, tools: list[ToolSpec]) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def generate_response(self, state: StateView, context: Mapping[str, Any]) -> str: ...


class DeterministicLLMProvider(LLMProvider):
    """Rule-based provider used for deterministic tests and offline demos."""

    async def extract_intent(self, text, state):
        t = text.lower()
        if "book" in t and "flight" in t:
            return "book_flight"
        if "ticket" in t:
            return "create_ticket"
        if "manual" in t or "look up" in t:
            return "manual_lookup"
        return state.intent

    async def extract_slots(self, text, state):
        t = text.strip()
        low = t.lower()
        out: dict[str, Any] = {}

        cities = ["Delhi", "Mumbai", "Bangalore", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Pune"]
        if "actually" in low:
            for city in cities:
                if city.lower() in low:
                    out["destination"] = "Bangalore" if city == "Bengaluru" else city
                    break
        if "to " in low:
            tail = low.split("to ", 1)[1]
            for city in cities:
                if city.lower() in tail:
                    out["destination"] = "Bangalore" if city == "Bengaluru" else city
                    break
        for marker in ("passengers", "passenger"):
            if marker in low:
                before = low.split(marker, 1)[0].strip()
                digits = "".join(c for c in before[-3:] if c.isdigit())
                if digits:
                    out["passengers"] = int(digits)
        if "tomorrow" in low:
            out["date"] = "tomorrow"
        return out

    async def select_tools(self, state, tools):
        names = {t.name for t in tools}
        if state.intent == "book_flight" and "destination" in state.slots:
            return [{
                "tool": "search_flights",
                "arguments": {
                    "origin": state.slots.get("origin", "Bangalore"),
                    "destination": state.slots["destination"],
                    "date": state.slots.get("date", "unspecified"),
                    "passengers": state.slots.get("passengers", 1),
                },
            }]
        if state.intent == "create_ticket" and "create_ticket" in names:
            if "subject" in state.slots:
                return [{"tool": "create_ticket", "arguments": dict(state.slots)}]
        if state.intent == "manual_lookup" and "lookup_manual" in names:
            if "frame" in state.slots:
                return [{"tool": "lookup_manual", "arguments": {"frame": state.slots["frame"]}}]
        return []

    async def generate_response(self, state, context):
        if context.get("clarification"):
            return context["clarification"]
        if context.get("stale"):
            return "That result is no longer relevant, so I discarded it and continued with your latest request."
        if context.get("tool_failed"):
            return "The operation failed safely. I can retry it without claiming it completed."
        if context.get("tool_output"):
            return "Done — I have the latest result."
        return "I’m working on that."


class HTTPJSONLLMProvider(LLMProvider):
    """
    Provider-neutral HTTP adapter.

    Environment:
      LLM_ENDPOINT=https://...
      LLM_API_KEY=...
      LLM_MODEL=...

    The endpoint is expected to accept:
      {"model": "...", "operation": "...", "input": ...}

    and return:
      {"result": ...}

    It is deliberately isolated from AgentRuntime. If the evaluation kit specifies
    a provider SDK, only this adapter needs to change.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        self.endpoint = endpoint or os.getenv("LLM_ENDPOINT")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "default")
        if not self.endpoint:
            raise ValueError("LLM_ENDPOINT is required for HTTPJSONLLMProvider")

    async def _call(self, operation: str, payload: Any) -> Any:
        body = json.dumps({"model": self.model, "operation": operation, "input": payload}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def request():
            req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())

        data = await asyncio.to_thread(request)
        return data["result"]

    async def extract_intent(self, text, state):
        return await self._call("extract_intent", {"text": text, "state": state.__dict__})

    async def extract_slots(self, text, state):
        return await self._call("extract_slots", {"text": text, "state": state.__dict__})

    async def select_tools(self, state, tools):
        return await self._call("select_tools", {
            "state": state.__dict__,
            "tools": [t.__dict__ for t in tools],
        })

    async def generate_response(self, state, context):
        return await self._call("generate_response", {"state": state.__dict__, "context": dict(context)})
