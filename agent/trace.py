from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable


@dataclass
class TraceRecord:
    timestamp_ms: int
    kind: str
    event: Any = None
    state: Any = None
    action: Any = None
    call_id: str | None = None
    tool: str | None = None
    cancellation: bool | None = None
    state_version: int | None = None
    result: Any = None
    final_response: str | None = None
    note: str | None = None


class TraceLogger:
    def __init__(self):
        self.records: list[TraceRecord] = []
        self._observers: list[Callable[[TraceRecord], None]] = []

    def add_observer(self, observer: Callable[[TraceRecord], None]) -> None:
        """Subscribe to new trace records without changing benchmark behavior."""
        self._observers.append(observer)

    def remove_observer(self, observer: Callable[[TraceRecord], None]) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def record(self, **kwargs) -> None:
        record = TraceRecord(**kwargs)
        self.records.append(record)
        for observer in tuple(self._observers):
            try:
                observer(record)
            except Exception:
                # Observability must never break the agent or benchmark path.
                pass

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records]

    def pretty(self) -> str:
        lines = []
        for r in self.records:
            lines.append(
                f"{r.timestamp_ms:>5}ms | {r.kind:<24} | "
                f"v={r.state_version!s:<3} | call={r.call_id or '-':<16} | "
                f"{r.note or ''}"
            )
        return "\n".join(lines)
