from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Mapping
import hashlib
import json
import time
import uuid


class EventType(str, Enum):
    USER_TEXT = "user_text"
    END_TURN = "end_turn"
    AUDIO = "audio"
    IMAGE_FRAME = "image_frame"
    INTERRUPTION = "interruption"
    TOOL_RESULT = "tool_result"
    TOOL_MANIFEST = "tool_manifest"
    SESSION_RESET = "session_reset"


@dataclass(frozen=True)
class Event:
    event_id: str
    timestamp_ms: int
    type: EventType
    payload: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def make(timestamp_ms: int, event_type: EventType, payload: Mapping[str, Any] | None = None,
             event_id: str | None = None) -> "Event":
        return Event(
            event_id=event_id or f"evt-{uuid.uuid4().hex[:12]}",
            timestamp_ms=timestamp_ms,
            type=event_type,
            payload=dict(payload or {}),
        )


@dataclass
class StateSnapshot:
    version: int = 0
    intent: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    pending_calls: set[str] = field(default_factory=set)
    completed_calls: set[str] = field(default_factory=set)
    session_id: str = "session"
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "StateSnapshot":
        return StateSnapshot(
            version=self.version,
            intent=self.intent,
            slots=dict(self.slots),
            pending_calls=set(self.pending_calls),
            completed_calls=set(self.completed_calls),
            session_id=self.session_id,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pending_calls"] = sorted(d["pending_calls"])
        d["completed_calls"] = sorted(d["completed_calls"])
        return d


@dataclass(frozen=True)
class StateView:
    version: int
    intent: str | None
    slots: Mapping[str, Any]
    pending_calls: frozenset[str]
    completed_calls: frozenset[str]
    session_id: str

    @staticmethod
    def from_snapshot(s: StateSnapshot) -> "StateView":
        return StateView(
            version=s.version,
            intent=s.intent,
            slots=dict(s.slots),
            pending_calls=frozenset(s.pending_calls),
            completed_calls=frozenset(s.completed_calls),
            session_id=s.session_id,
        )


class ActionType(str, Enum):
    ACK = "ack"
    PROGRESS = "progress"
    TOOL_CALL = "tool_call"
    CANCEL = "cancel"
    CLARIFICATION = "clarification"
    FINAL = "final"
    STATE_SNAPSHOT = "state_snapshot"
    ERROR = "error"


@dataclass
class Action:
    action_id: str
    timestamp_ms: int
    type: ActionType
    payload: dict[str, Any]
    state_version: int

    @staticmethod
    def make(timestamp_ms: int, action_type: ActionType, payload: dict[str, Any],
             state_version: int) -> "Action":
        return Action(
            action_id=f"act-{uuid.uuid4().hex[:12]}",
            timestamp_ms=timestamp_ms,
            type=action_type,
            payload=payload,
            state_version=state_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "timestamp_ms": self.timestamp_ms,
            "type": self.type.value,
            "payload": self.payload,
            "state_version": self.state_version,
        }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    cancellable: bool
    idempotency_required: bool
    state_effects: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool: str
    arguments: dict[str, Any]
    state_version: int
    epoch: int
    idempotency_key: str | None
    intent_fingerprint: str

    @staticmethod
    def fingerprint(intent: str | None, slots: Mapping[str, Any]) -> str:
        blob = json.dumps({"intent": intent, "slots": dict(sorted(slots.items()))},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class ToolResult:
    call_id: str
    tool: str
    success: bool
    output: dict[str, Any]
    state_version: int
    epoch: int
    error: str | None = None
    stale: bool = False
    cancelled: bool = False


@dataclass
class OperationContext:
    call: ToolCall
    task: Any = None
    cancel_event: Any = None
    started_ms: int = 0


def utc_ms() -> int:
    return int(time.time() * 1000)
