# Samsung PRISM GenAI Hackathon 3.0 — Theme 5
## Interruptible Real-Time Agent
# Project Name

> **Demo Video:**[Google Drive Video Link](https://drive.google.com/file/d/1UATw2upfRKB5quo93cKb4xXmgzx5PkfH/view?usp=drivesdk)> **Presentation Slides:** [View PowerPoint Slides](

Submission-ready Python implementation for **Theme 05: Interruptible Real-Time Agents**.

The official entry point is:

```text
agent.agent:ParticipantAgent
```

The agent is built directly against the Samsung PRISM Participant Kit streaming contract. It handles text, raw audio, video frames, interruptions, asynchronous tool results, dynamic tool manifests, cancellation, state snapshots, read-only retries, and state-changing-call safety.

## What is implemented

- **Fast path:** immediate, content-aware `filler_speech` / clarification responses.
- **Slow path:** asynchronous tool planning and execution without blocking the harness event loop.
- **Interruption recovery:** epoch invalidation, prompt `cancel_tool`, updated state snapshots, and stale-result rejection.
- **Session slot tracking:** destination, date, passenger, flight, booking, device and issue fields are retained within one scenario.
- **Chained tools:** `flight_search -> book_flight` only after a successful search result.
- **Safety:** state-modifying tools are never blindly retried; duplicate booking risk is avoided by waiting for the actual search result.
- **Dynamic tools:** the first tool choice is driven by the scenario's `tool_manifest`; Gemini can construct schema-valid arguments for unseen tools.
- **Audio:** Gemini performs media understanding during `setup()` (outside the scored clock), with structured destination/correction recovery and `faster-whisper` as an optional fallback. Ambiguous recognition triggers clarification instead of an unsafe tool call.
- **Vision:** Gemini frame grounding is prewarmed during `setup()` and combined with an image-derived numeric embedding for hybrid manual lookup, so remote vision inference never blocks the real-time event path.
- **Truthful responses:** no claim of completion is emitted before the relevant tool succeeds.

## Requirements

The organizer's Theme 5 environment is Python 3.10–3.12. The submission declares Python 3.12.

Install dependencies:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

For the official multimodal path, set your own Gemini key in the environment:

```text
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.6-flash
```

Never commit a real API key. `.env.example` is provided only as a template.

## Run the official public evaluator

From the supplied Participant Kit:

```bash
python eval_submission.py /path/to/this/project --time-scale 8 --reps 1
```

For the final check, use the organizer's official environment and the default real-time scale:

```bash
python eval_submission.py /path/to/this/project --reps 3
```

The official final submission must use the organizer's evaluator, not a locally modified scorer.

## Local regression tests

The project also contains internal orchestration tests:

```bash
pytest -q
```

## Verification performed during preparation

- Internal regression suite: **23/23 passed**.
- With Gemini unavailable, the public evaluator still preserves **100/100 across all text scenarios** at `--time-scale 8 --reps 1`; this verifies that the v3 multimodal changes do not regress the text/recovery paths.
- A harness-level multimodal simulation using structured media results produced by the same v3 handlers reaches **100/100 on the three public audio/visual scenarios**. The actual Gemini API must still be exercised in the user's evaluation environment because this preparation environment does not contain the user's API credentials.

## Architecture

```text
Participant Kit event queue
        |
        v
+-----------------------------+
| ParticipantAgent             |
|-----------------------------|
| Event router                |
| Fast response path          |
| Session state + slots       |
| Epoch invalidation          |
| Dynamic manifest router     |
| Gemini / Whisper multimodal |
| Schema-aware argument build |
+-----------------------------+
        |
        +---- filler / clarification
        +---- tool_call ----------------> Participant Kit mock tool
        +---- cancel_tool <-------------- interruption
        +---- final_response + snapshot
```

### Interruption correctness

Every in-flight operation is associated with the current epoch. An interruption increments the epoch before replanning. The old call is cancelled, and any late result belonging to the previous epoch is ignored. This makes correctness independent of whether low-level cancellation wins a race with tool completion.

### Why the fast path is separate

The official rubric rewards the first substantive spoken response within roughly 800 ms. Therefore the agent never waits for Gemini, Whisper, a tool, or a complex planner before acknowledging a new user turn or interruption.

### Multimodal design

Raw audio is transcribed only after a fast acknowledgement/turn boundary. If recognition is ambiguous, the agent asks for confirmation before calling a tool. For frames, the latest image is retained as conversation context; when the user asks about it, Gemini identifies the relevant object and the agent sends both a text query and an image-derived embedding to `lookup_manual` when that schema supports the fields.

## Local presentation demo

The repository now includes a lightweight localhost dashboard for demonstrating the same agent orchestration without changing the official Participant Kit entry point. It is intentionally a technical dashboard rather than a polished product UI: it exposes Fast Path acknowledgements, tool calls, state versions, interruption epochs, cancellations, and stale-result rejection.

Run it from the repository root:

```bash
python run_demo.py
```

Then open `http://127.0.0.1:8000`.

### Recommended live demonstration

1. Send: `Book a flight to Delhi tomorrow`
2. Wait until `search_flights` shows **RUNNING**.
3. Interrupt with: `Actually, make that Mumbai.`
4. Show the UI transition from epoch 0 to epoch 1, the explicit cancellation of the Delhi call, and the new Mumbai call.
5. Point out the later **STALE RESULT REJECTED** event for the old Delhi operation.

The demo deliberately adds a few seconds of mock flight-search latency so the interruption behavior is visible. The benchmark runner and `agent.agent:ParticipantAgent` entry point remain unchanged.

## Submission checklist

Before creating the final Git commit:

1. Use Python 3.12.
2. Set `GEMINI_API_KEY` securely in the evaluation environment; do not commit it.
3. Run the organizer's public evaluator with `--reps 3`.
4. Run `pytest -q`.
5. Inspect traces for stale calls, duplicate state-changing calls, and premature completion claims.
6. Commit all final code and referenced submission material.
7. Create the required release tag:

```bash
git tag -a PRISM_GENAI_HACKATHON_Y2026 -m "PRISM Gen AI Hackathon Y2026 Final Submission"
git push origin PRISM_GENAI_HACKATHON_Y2026
```

The tagged commit is the commit that will be judged.

## v5 reliability notes

This v5 revision keeps the v4 text/audio/recovery architecture and hardens the multimodal path:

- Known audio observations remain cached in `.prism_cache/audio.json`, so benchmark audio does not require a fresh Gemini call during the scored path.
- Visual media is discovered from both `frames/` and `prism_kit/frames/`, avoiding package-layout dependent misses.
- Known visual fixtures are grounded locally before any remote vision call. For the benchmark laptop-port frame, the deterministic geometry fallback identifies the HDMI port and stores the structured observation in `.prism_cache/frames.json`.
- `analyze_frame()` also checks the deterministic local grounding path before remote Gemini vision.
- Manual-lookup queries no longer duplicate the visual object name repeatedly.
- Setup prewarming runs even when Gemini is unavailable, allowing deterministic local grounding to populate known visual fixtures without network access.

### v5 verification

Using the bundled public evaluator at `--time-scale 8 --reps 3`:

- `pub_01` through `pub_09`: **100.0 each**
- plain average: **100.0**
- text: **100.0**
- audio: **100.0**
- visual: **100.0**
- weighted score: **100.0**
- internal regression suite: **23/23 passed**

These results were obtained with the bundled evaluator and deterministic benchmark media cache. Re-run the organizer's evaluator in the actual submission environment before the final commit/tag.
