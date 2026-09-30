from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MultimodalObservation:
    modality: str
    confidence: float
    data: dict[str, Any]
    ambiguous: bool = False


class MultimodalProcessor:
    """
    Deterministic boundary for STT/vision.

    Real STT/vision providers can be injected here without touching the
    orchestration layer.
    """

    async def audio(self, payload: dict[str, Any]) -> MultimodalObservation:
        transcript = payload.get("transcript")
        if transcript:
            return MultimodalObservation("audio", 0.99, {"text": transcript})
        return MultimodalObservation("audio", 0.0, {}, ambiguous=True)

    async def image(self, payload: dict[str, Any]) -> MultimodalObservation:
        # Tests may supply a deterministic fixture label.
        label = payload.get("label")
        if label:
            return MultimodalObservation("image", 0.98, {"label": label})
        return MultimodalObservation(
            "image", 0.0, {"candidates": ["valve", "switch"]}, ambiguous=True
        )
