from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent.models import Action, Event, EventType
from agent.runtime import AgentRuntime


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000


class DemoClock:
    """Wall-clock adapter with the same interface used by MockToolEnvironment."""

    def __init__(self) -> None:
        self._started = time.monotonic()

    @property
    def now_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000)

    async def wait_until(self, target_ms: int) -> None:
        delay = max(0.0, (target_ms - self.now_ms) / 1000.0)
        if delay:
            await asyncio.sleep(delay)


class DemoHub:
    """Thread-safe fan-out for browser SSE clients."""

    def __init__(self) -> None:
        self._clients: set[queue.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._clients.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._clients.discard(q)

    def publish(self, message: dict[str, Any]) -> None:
        with self._lock:
            clients = tuple(self._clients)
        for q in clients:
            q.put(message)


class DemoApp:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clock = DemoClock()
        self.runtime = AgentRuntime(clock=self.clock, session_id="presentation-session")
        self.hub = DemoHub()
        # A deliberately visible delay makes cancellation/replanning observable.
        self.runtime.mock_tools.configure("search_flights", latency_ms=3000)
        self.runtime.mock_tools.configure("book_flight", latency_ms=1800)
        self.runtime.trace.add_observer(self._on_trace)
        self.runtime.add_action_observer(self._on_action)

    def _on_trace(self, record) -> None:
        self.hub.publish({"kind": "trace", "trace": {
            "timestamp_ms": record.timestamp_ms,
            "kind": record.kind,
            "call_id": record.call_id,
            "tool": record.tool,
            "state_version": record.state_version,
            "result": record.result,
            "note": record.note,
            "final_response": record.final_response,
        }})

    def _on_action(self, action: Action) -> None:
        self.hub.publish({"kind": "action", "action": action.to_dict()})

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        await self.runtime.start()

    async def stop(self) -> None:
        await self.runtime.stop()

    async def submit(self, event_type: EventType, text: str = "") -> None:
        event = Event.make(self.clock.now_ms, event_type, {"text": text} if text else {})
        await self.runtime.submit(event)

    async def reset(self) -> None:
        await self.submit(EventType.SESSION_RESET)

    async def snapshot(self) -> dict[str, Any]:
        state = await self.runtime.state.snapshot()
        epoch = await self.runtime.cancellation.current_epoch()
        return {
            "version": state.version,
            "intent": state.intent,
            "slots": dict(state.slots),
            "pending_calls": sorted(state.pending_calls),
            "completed_calls": sorted(state.completed_calls),
            "session_id": state.session_id,
            "epoch": epoch,
        }


APP = DemoApp()


def run_async(coro):
    if APP.loop is None:
        raise RuntimeError("Agent loop is not ready")
    return asyncio.run_coroutine_threadsafe(coro, APP.loop).result(timeout=10)


def sse_message(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "Theme5Demo/1.0"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, filename: str, content_type: str) -> None:
        path = STATIC / filename
        if not path.exists():
            self._json(404, {"error": "not found"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self._static("index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._static("app.js", "application/javascript; charset=utf-8")
        if path == "/style.css":
            return self._static("style.css", "text/css; charset=utf-8")
        if path == "/api/state":
            return self._json(200, run_async(APP.snapshot()))
        if path == "/api/events":
            return self._events()
        self._json(404, {"error": "not found"})

    def _events(self) -> None:
        q = APP.hub.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.wfile.write(sse_message({"kind": "connected", "state": run_async(APP.snapshot())}))
            self.wfile.flush()
            while True:
                try:
                    payload = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(sse_message(payload))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            APP.hub.unsubscribe(q)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "invalid JSON"})

        if path == "/api/message":
            text = str(payload.get("text", "")).strip()
            if not text:
                return self._json(400, {"error": "text is required"})
            APP.hub.publish({"kind": "user", "text": text, "mode": "message"})
            run_async(APP.submit(EventType.USER_TEXT, text))
            # A browser message represents a complete turn. The runtime still performs
            # the input parse asynchronously before planning.
            run_async(APP.submit(EventType.END_TURN))
            return self._json(202, {"ok": True})

        if path == "/api/interrupt":
            text = str(payload.get("text", "")).strip()
            if not text:
                return self._json(400, {"error": "text is required"})
            APP.hub.publish({"kind": "user", "text": text, "mode": "interrupt"})
            run_async(APP.submit(EventType.INTERRUPTION, text))
            return self._json(202, {"ok": True})

        if path == "/api/reset":
            APP.hub.publish({"kind": "system", "text": "Session reset"})
            run_async(APP.reset())
            return self._json(202, {"ok": True})

        self._json(404, {"error": "not found"})

    def log_message(self, format: str, *args) -> None:
        # Keep the presentation terminal clean.
        return


async def _agent_main(server: ThreadingHTTPServer) -> None:
    await APP.start()
    print("\\nTheme 5 local demo is running")
    print("Open: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.\\n")
    try:
        await asyncio.Event().wait()
    finally:
        await APP.stop()
        server.shutdown()


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def runner():
        await _agent_main(server)

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
