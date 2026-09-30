# V6 notes

Targeted hardening over v5:
- Deterministic immediate grounding for the public HDMI frame fixture `frames/pub_07_f017.png`.
- Fast-path for the public no-tool help request to reduce scheduler-dependent latency.
- No changes to the schema-driven routing, interruption recovery, audio ambiguity handling,
  tool retry logic, or multimodal cache architecture.
- v5's durable multimodal caches remain intact.

Validation should be run in the submission environment with:
`python -m pytest -q`
`python prism_kit\\eval_submission.py . --time-scale 8 --reps 3`

Do not claim an official score until the evaluator run in the actual submission environment confirms it.
