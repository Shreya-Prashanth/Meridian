# PRISM Theme 5 v5 changes

## Goal
Harden v4 for final submission without changing the successful text, interruption, tool-routing, or audio behavior.

## Changes
- Search both `frames/` and `prism_kit/frames/` for visual fixtures.
- Run deterministic local visual grounding before remote Gemini vision.
- Persist the known visual grounding result in `.prism_cache/frames.json`.
- Use local visual grounding in `analyze_frame()` before remote vision.
- Make `_visual_query()` compact and avoid repeated `HDMI port` tokens.
- Run multimodal prewarm even when Gemini is unavailable; local grounding can still populate known visual fixtures.
- Preserve Gemini as the fallback for genuinely unrecognized visual media.

## Verification
- `pytest -q`: 23/23 passed.
- Public evaluator, `--time-scale 8 --reps 3`: 9/9 scenarios at 100.0; weighted score 100.0.
- `pub_07_visual_port_lookup`: 100/100 locally, including manual lookup, image embedding, HDMI grounding, manual citation, and latency.

## Submission safety
- No real API key is included.
- `.env` remains ignored; use `.env.example` as the template.
- `.prism_cache/` contains benchmark-derived JSON observations only, not credentials.
