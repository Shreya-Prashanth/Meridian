
import asyncio
import json
import logging
import os
import re
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types


# --------------------------------------------------
# Configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flight-agent")

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE_PATH = BASE_DIR / "frontend" / "index.html"

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-live-preview",
)

INPUT_RATE = 16000

app = FastAPI(title="Interruptible Flight Agent")

# 1. Fixes the blank page when assets are missing
BASE_DIR = Path(__file__).resolve().parent
if (BASE_DIR / "frontend").exists():
    app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend"), name="static")

# 2. Resolves the 404 error your browser was requesting
@app.get("/api/state")
async def get_state():
    return {"status": "idle", "destination": None}

@app.get("/api/state")
async def get_state():
    return {"status": "idle", "destination": None}

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


# --------------------------------------------------
# Gemini configuration
# --------------------------------------------------

def create_live_config():
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="book_flight",
                        description=(
                            "Call this function whenever the user "
                            "requests or updates a flight destination."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "destination": types.Schema(
                                    type=types.Type.STRING,
                                    description="The destination city.",
                                )
                            },
                            required=["destination"],
                        ),
                    )
                ]
            )
        ],
        system_instruction=types.Content(
            parts=[
                types.Part(
                    text=(
                        "You are an interactive voice flight booking "
                        "assistant. When a user requests a destination, "
                        "call the book_flight tool immediately. "
                        "Speak in short, clear sentences. "
                        "This is a simulated flight booking demo. "
                        "Never claim a real airline reservation exists."
                    )
                )
            ]
        ),
    )


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def extract_destination(text: str) -> str:
    """Extract a destination from common correction phrases."""
    text = text.strip()

    patterns = [
        r"^\s*no\s*,?\s*change\s+(?:it\s+)?to\s+",
        r"^\s*change\s+(?:it\s+)?to\s+",
        r"^\s*change\s+to\s+",
        r"^\s*instead\s*,?\s*",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip(" .,!?")


async def safe_send_json(websocket: WebSocket, data: dict) -> bool:
    """Send JSON without crashing on a disconnected client."""
    try:
        await websocket.send_json(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.get("/")
async def get_frontend():
    if not HTML_FILE_PATH.is_file():
        return HTMLResponse(
            content=(
                "Frontend not found. Expected file: "
                f"{HTML_FILE_PATH}"
            ),
            status_code=404,
        )

    return HTMLResponse(
        content=HTML_FILE_PATH.read_text(encoding="utf-8")
    )


# --------------------------------------------------
# WebSocket
# --------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    logger.info("Frontend connected")

    if client is None:
        await safe_send_json(
            websocket,
            {
                "type": "error",
                "message": "GEMINI_API_KEY is not configured.",
            },
        )
        await websocket.close(code=1011)
        return

    current_destination = None
    booking_timer_task = None
    booking_generation = 0

    # Serialize outgoing requests to Gemini.
    gemini_send_lock = asyncio.Lock()

    async def send_text_to_gemini(text: str):
        async with gemini_send_lock:
            await session.send_realtime_input(text=text)

    async def cancel_booking_timer():
        nonlocal booking_timer_task

        task = booking_timer_task

        if task and not task.done():
            task.cancel()

            with suppress(asyncio.CancelledError):
                await task

        booking_timer_task = None

    async def finalize_booking_after_delay(
        destination: str,
        generation: int,
    ):
        try:
            await safe_send_json(
                websocket,
                {
                    "type": "booking_started",
                    "destination": destination,
                    "duration": 5,
                    "message": (
                        f"Booking to {destination} will be "
                        "confirmed in 5 seconds."
                    ),
                },
            )

            await asyncio.sleep(5)

            # Prevent an old timer from confirming a newer booking.
            if generation != booking_generation:
                return

            await safe_send_json(
                websocket,
                {
                    "type": "booking_success",
                    "destination": destination.upper(),
                    "message": (
                        f"Flight booking confirmed for "
                        f"{destination.upper()}."
                    ),
                },
            )

            await send_text_to_gemini(
                "System event: The simulated booking to "
                f"{destination} is confirmed. "
                "Tell the user briefly that the flight "
                "has been booked successfully."
            )

        except asyncio.CancelledError:
            logger.info("Booking timer cancelled: %s", destination)
            raise

        except Exception:
            logger.exception("Booking timer failed")

            await safe_send_json(
                websocket,
                {
                    "type": "error",
                    "message": "The booking countdown failed.",
                },
            )

    try:
        async with client.aio.live.connect(
            model=MODEL_NAME,
            config=create_live_config(),
        ) as session:

            await safe_send_json(
                websocket,
                {
                    "type": "connected",
                    "message": "Connected to the flight agent.",
                },
            )

            # ------------------------------------------
            # Browser -> Gemini
            # ------------------------------------------

            async def receive_from_browser():
                nonlocal current_destination
                nonlocal booking_generation

                while True:
                    message = await websocket.receive()

                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect(
                            message.get("code", 1000)
                        )

                    # Raw microphone PCM bytes
                    audio_data = message.get("bytes")

                    if audio_data:
                        async with gemini_send_lock:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=audio_data,
                                    mime_type=f"audio/pcm;rate={INPUT_RATE}",
                                )
                            )
                        continue

                    # JSON control messages
                    raw_text = message.get("text")

                    if not raw_text:
                        continue

                    try:
                        data = json.loads(raw_text)
                    except json.JSONDecodeError:
                        await safe_send_json(
                            websocket,
                            {
                                "type": "error",
                                "message": "Invalid JSON message.",
                            },
                        )
                        continue

                    if not isinstance(data, dict):
                        continue

                    msg_type = data.get("type")

                    if msg_type == "interrupt":
                        logger.info("User interrupted booking")

                        # Invalidate any timer that may still be running.
                        booking_generation += 1
                        await cancel_booking_timer()

                        await safe_send_json(
                            websocket,
                            {
                                "type": "interrupted",
                                "message": "Booking paused.",
                            },
                        )

                        await send_text_to_gemini(
                            "System event: The user interrupted "
                            "the booking. Tell them briefly that "
                            "the booking is paused and ask what "
                            "they would like to change."
                        )

                    elif msg_type == "text_update":
                        user_text = data.get("text", "")

                        if not isinstance(user_text, str):
                            continue

                        user_text = user_text.strip()

                        if not user_text:
                            continue

                        logger.info("Text update received: %s", user_text)

                        # Invalidate and cancel the old countdown.
                        booking_generation += 1
                        await cancel_booking_timer()

                        new_destination = extract_destination(user_text)

                        if not new_destination:
                            await safe_send_json(
                                websocket,
                                {
                                    "type": "error",
                                    "message": (
                                        "Please enter a destination."
                                    ),
                                },
                            )
                            continue

                        current_destination = new_destination

                        await safe_send_json(
                            websocket,
                            {
                                "type": "status",
                                "state": "updating",
                                "message": (
                                    f"Updating destination to "
                                    f"{new_destination.upper()}..."
                                ),
                            },
                        )

                        # Tell Gemini about the correction.
                        await send_text_to_gemini(
                            "System event: The user changed the "
                            f"flight destination to {new_destination}. "
                            "Acknowledge the change briefly. Do not "
                            "claim that a real airline reservation "
                            "has been made."
                        )

                        # Start a fresh interruption window.
                        booking_generation += 1
                        generation = booking_generation

                        booking_timer_task = asyncio.create_task(
                            finalize_booking_after_delay(
                                new_destination,
                                generation,
                            )
                        )

                    elif msg_type == "ping":
                        await safe_send_json(
                            websocket,
                            {"type": "pong"},
                        )

                    else:
                        logger.warning(
                            "Unknown frontend message type: %s",
                            msg_type,
                        )

            # ------------------------------------------
            # Gemini -> Browser
            # ------------------------------------------

            async def send_to_browser():
                nonlocal current_destination
                nonlocal booking_timer_task
                nonlocal booking_generation

                async for response in session.receive():

                    # 1. Forward Gemini audio to the browser.
                    server_content = getattr(
                        response,
                        "server_content",
                        None,
                    )

                    if server_content is not None:
                        model_turn = getattr(
                            server_content,
                            "model_turn",
                            None,
                        )

                        if model_turn is not None:
                            for part in getattr(
                                model_turn,
                                "parts",
                                [],
                            ):
                                inline_data = getattr(
                                    part,
                                    "inline_data",
                                    None,
                                )

                                audio = (
                                    getattr(inline_data, "data", None)
                                    if inline_data
                                    else None
                                )

                                if audio:
                                    await websocket.send_bytes(audio)

                        # Optional: forward interruption information
                        # if Gemini reports that its own turn was cut.
                        if getattr(
                            server_content,
                            "interrupted",
                            False,
                        ):
                            await safe_send_json(
                                websocket,
                                {
                                    "type": "assistant_interrupted",
                                },
                            )

                    # 2. Handle function calls.
                    tool_call = getattr(
                        response,
                        "tool_call",
                        None,
                    )

                    if tool_call is None:
                        continue

                    for call in getattr(
                        tool_call,
                        "function_calls",
                        [],
                    ):
                        if getattr(call, "name", "") != "book_flight":
                            continue

                        args = getattr(call, "args", {}) or {}
                        new_destination = str(
                            args.get("destination", "")
                        ).strip()

                        if not new_destination:
                            continue

                        current_destination = new_destination

                        # Cancel any previous booking timer.
                        booking_generation += 1
                        await cancel_booking_timer()

                        await safe_send_json(
                            websocket,
                            {
                                "type": "destination_update",
                                "destination": (
                                    new_destination.upper()
                                ),
                                "message": (
                                    f"Preparing flight booking to "
                                    f"{new_destination.upper()}."
                                ),
                            },
                        )

                        # Acknowledge the function call.
                        function_response = types.FunctionResponse(
                            name=call.name,
                            id=getattr(call, "id", None),
                            response={
                                "result": (
                                    "The simulated booking process "
                                    f"has started for {new_destination}. "
                                    "A five-second interruption window "
                                    "is active."
                                )
                            },
                        )

                        async with gemini_send_lock:
                            await session.send_tool_response(
                                function_responses=[
                                    function_response
                                ]
                            )

                        # Start the new countdown.
                        booking_generation += 1
                        generation = booking_generation

                        booking_timer_task = asyncio.create_task(
                            finalize_booking_after_delay(
                                new_destination,
                                generation,
                            )
                        )

            # ------------------------------------------
            # Run both directions concurrently
            # ------------------------------------------

            browser_task = asyncio.create_task(
                receive_from_browser()
            )

            gemini_task = asyncio.create_task(
                send_to_browser()
            )

            tasks = {browser_task, gemini_task}

            try:
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Propagate unexpected errors from completed tasks.
                for task in done:
                    exception = task.exception()

                    if exception is not None:
                        raise exception

            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()

                await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

    except WebSocketDisconnect:
        logger.info("Frontend disconnected")

    except asyncio.CancelledError:
        raise

    except Exception:
        logger.exception("Flight agent WebSocket failed")

        await safe_send_json(
            websocket,
            {
                "type": "error",
                "message": (
                    "The flight agent encountered an error. "
                    "Please reconnect."
                ),
            },
        )

    finally:
        booking_generation += 1

        if booking_timer_task and not booking_timer_task.done():
            booking_timer_task.cancel()

            with suppress(asyncio.CancelledError):
                await booking_timer_task

        with suppress(Exception):
            await websocket.close()
