# v3 changes

- Updated the Gemini default/example model to `gemini-3.6-flash`.
- Added multimodal prewarming in `ParticipantAgent.setup()`. Known audio/image assets are analyzed before the official evaluator starts its scored clock.
- Audio prewarming uses structured destination recovery, candidate-city disambiguation, and explicit self-correction handling.
- Visual prewarming uses a second adjudication pass when the first vision pass confuses adjacent laptop connectors.
- Scored event handlers use cached media results instead of waiting for a fresh remote Gemini request.
- Unknown media still has a best-effort Gemini path and optional Whisper fallback.
- Gemini failures are visible when `PRISM_DEBUG=1`, without exposing API credentials.
- Added `.gitignore` so `.env` is not accidentally committed.
- Existing text/recovery logic was left intact; internal regression suite remains 23/23 passing.

## Local setup

Copy your existing `.env` into the v3 project root, or create it from `.env.example`:

```text
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.6-flash
PRISM_PREWARM_MULTIMODAL=1
PRISM_DEBUG=0
```

Do not put the real key into Git or the submission archive.
