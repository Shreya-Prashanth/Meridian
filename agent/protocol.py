from __future__ import annotations

from typing import Any
from .models import Action, ActionType


class ProtocolError(ValueError):
    pass


class ProtocolValidator:
    REQUIRED = {"action_id", "timestamp_ms", "type", "payload", "state_version"}

    def validate_action(self, action: Action) -> None:
        d = action.to_dict()
        missing = self.REQUIRED - set(d)
        if missing:
            raise ProtocolError(f"missing action fields: {sorted(missing)}")
        if not isinstance(d["action_id"], str) or not d["action_id"]:
            raise ProtocolError("invalid action_id")
        if not isinstance(d["timestamp_ms"], int) or d["timestamp_ms"] < 0:
            raise ProtocolError("invalid timestamp_ms")
        if d["type"] not in {x.value for x in ActionType}:
            raise ProtocolError("invalid action type")
        if not isinstance(d["payload"], dict):
            raise ProtocolError("payload must be an object")
        if not isinstance(d["state_version"], int) or d["state_version"] < 0:
            raise ProtocolError("invalid state_version")

        if action.type == ActionType.TOOL_CALL:
            if not action.payload.get("call_id") or not action.payload.get("tool"):
                raise ProtocolError("tool_call requires call_id and tool")
        if action.type == ActionType.CANCEL and not action.payload.get("call_id"):
            raise ProtocolError("cancel requires call_id")
        if action.type == ActionType.STATE_SNAPSHOT and "state" not in action.payload:
            raise ProtocolError("state_snapshot requires state")
