import asyncio
import os
import sys
import select
import pyaudio
from google import genai
from google.genai import types

# Initialize Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 1024

# Global State Variables
current_destination = None
booking_timer_task = None
is_interrupted = False


def execute_final_booking(destination: str):
    """Prints final text output confirming the flight booking."""
    print("\n=======================================================")
    print(f" 🎉 SUCCESS: Flight officially BOOKED to -> {destination.upper()}")
    print("=======================================================\n")


async def finalize_booking_after_delay(destination: str, session, audio_queue, send_task, play_task, mic_stream, speaker_stream, p):
    """Waits 5 seconds before finalizing booking unless interrupted via keyboard."""
    global is_interrupted, current_destination
    try:
        print("\n⏳ [COUNTDOWN]: 5-second timer active. Press ENTER to interrupt and type changes...")
        
        # Non-blocking 5-second check for Enter keypress
        for _ in range(50):
            await asyncio.sleep(0.1)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()  # Flush enter keypress
                is_interrupted = True
                print("\n=======================================================")
                print(" ⏸️  [INTERRUPTED]: Flight booking paused!")
                print("=======================================================\n")
                
                # Instruct model to speak interruption prompt out loud
                await session.send_realtime_input(
                    text="System Event: User interrupted. Speak out loud: 'Interrupted. Please type what change you would like to make.'"
                )

                # Prompt user for text input in console
                loop = asyncio.get_running_loop()
                user_change = await loop.run_in_executor(
                    None, input, "💬 What change would you like to make? -> "
                )

                if user_change.strip():
                    print(f"\n🔄 [PROCESSING CHANGE]: Executing text update: '{user_change}'")
                    
                    # Instruct AI to speak out loud that the flight has been booked to the new location
                    await session.send_realtime_input(
                        text=f"System Event: The user changed their request to '{user_change}'. Speak out loud concisely: 'Yes, your flight has been booked to the new location successfully.'"
                    )

                    # Wait 2 seconds as requested
                    await asyncio.sleep(2.0)

                    # Extract location or fallback to typed string
                    new_dest_display = user_change.replace("No,", "").replace("change it to", "").replace("Change it to", "").replace("change to", "").strip()
                    if not new_dest_display:
                        new_dest_display = user_change.strip()

                    # Print final success banner
                    execute_final_booking(new_dest_display)

                    # Give extra time for voice playback to finish speaking
                    await asyncio.sleep(2.5)

                    # Clean shutdown and immediate exit
                    send_task.cancel()
                    play_task.cancel()
                    mic_stream.stop_stream()
                    mic_stream.close()
                    speaker_stream.stop_stream()
                    speaker_stream.close()
                    p.terminate()
                    print("[INFO]: Workflow completed successfully. Exiting program.\n")
                    sys.exit(0)
                return

        if not is_interrupted:
            execute_final_booking(destination)

    except asyncio.CancelledError:
        print(f"\n[TIMER CANCELED]: Booking timer stopped for previous destination ({destination.upper()}).")


async def main():
    global current_destination, booking_timer_task, is_interrupted

    p = pyaudio.PyAudio()
    mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    speaker_stream = p.open(format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True, frames_per_buffer=CHUNK_SIZE)

    audio_queue = asyncio.Queue()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="book_flight",
                        description="Call this function whenever the user requests or updates a flight destination.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "destination": types.Schema(
                                    type=types.Type.STRING,
                                    description="The target destination city."
                                )
                            },
                            required=["destination"]
                        )
                    )
                ]
            )
        ],
        system_instruction=types.Content(
            parts=[
                types.Part(
                    text=(
                        "You are an interactive voice flight booking assistant.\n"
                        "When a destination is requested, call the book_flight tool immediately.\n"
                        "Speak out loud in short, clear sentences."
                    )
                )
            ]
        )
    )

    print("\n=======================================================")
    print(" 🎙️ HYBRID VOICE/TEXT FLIGHT BOOKING AGENT READY")
    print(" Step 1: Say 'Book a flight to Dubai'.")
    print(" Step 2: Press [ENTER] to interrupt & type your destination changes.")
    print("=======================================================\n")

    async with client.aio.live.connect(model="gemini-3.1-flash-live-preview", config=config) as session:

        async def send_audio():
            """Continuously streams microphone data to Gemini."""
            try:
                while True:
                    data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                    await asyncio.sleep(0.001)
            except (asyncio.CancelledError, Exception):
                pass

        async def play_audio():
            """Plays received audio responses through speakers."""
            try:
                while True:
                    data = await audio_queue.get()
                    if data:
                        await asyncio.to_thread(speaker_stream.write, data)
                    audio_queue.task_done()
            except asyncio.CancelledError:
                pass

        async def receive_responses():
            """Listens for AI audio parts and tool calls."""
            global current_destination, booking_timer_task, is_interrupted
            try:
                async for response in session.receive():
                    # 1. Audio Playback
                    server_content = getattr(response, "server_content", None)
                    if server_content is not None:
                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn is not None:
                            for part in getattr(model_turn, "parts", []):
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and getattr(inline_data, "data", None):
                                    await audio_queue.put(inline_data.data)

                    # 2. Tool Calls
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call is not None:
                        for call in getattr(tool_call, "function_calls", []):
                            if getattr(call, "name", "") == "book_flight":
                                new_dest = call.args.get("destination", "").strip()

                                if new_dest:
                                    if booking_timer_task and not booking_timer_task.done():
                                        booking_timer_task.cancel()

                                    print(f"\n✈️ [PROCESSING]: Initial booking initiated for -> {new_dest.upper()}")
                                    current_destination = new_dest

                                    # Respond to tool call to acknowledge out loud
                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                name=call.name,
                                                id=getattr(call, "id", None),
                                                response={
                                                    "result": f"Flight is being booked to {new_dest}. Speak acknowledgment out loud."
                                                }
                                            )
                                        ]
                                    )

                                    # Start 5-second countdown with task references passed for clean shutdown
                                    booking_timer_task = asyncio.create_task(
                                        finalize_booking_after_delay(
                                            current_destination, session, audio_queue,
                                            send_task, play_task, mic_stream, speaker_stream, p
                                        )
                                    )

            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[RECEIVER ERROR]: {e}")

        # Start tasks
        send_task = asyncio.create_task(send_audio())
        play_task = asyncio.create_task(play_audio())
        receive_task = asyncio.create_task(receive_responses())

        try:
            await asyncio.gather(send_task, play_task, receive_task)
        except (KeyboardInterrupt, asyncio.CancelledError, SystemExit):
            pass
        finally:
            if booking_timer_task and not booking_timer_task.done():
                booking_timer_task.cancel()
            send_task.cancel()
            play_task.cancel()
            receive_task.cancel()
            mic_stream.stop_stream()
            mic_stream.close()
            speaker_stream.stop_stream()
            speaker_stream.close()
            p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
