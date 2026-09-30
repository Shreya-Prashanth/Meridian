# PRISM Theme 5 v4 changes

- Replaced concurrent multimodal prewarm with sequential setup-time warming.
- Added bounded Gemini retry/backoff for temporary API failures.
- Added durable `.prism_cache/` for successful audio/frame interpretations.
- Known cached media is consumed synchronously on the scored path; no remote multimodal request is required for a cache hit.
- Strengthened visual grounding prompt to read visible port labels and added a setup-time adjudication pass for USB/HDMI confusion.
- Cache contains only model observations; API credentials are never written.
- `.prism_cache/` and `.env` are ignored by git.
