"""Shared runtime accessor for the active ExtractionPipeline instance."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from extraction.pipeline import ExtractionPipeline

_PIPELINE_INSTANCE: "ExtractionPipeline | None" = None
_PIPELINE_LOCK = threading.Lock()


def set_pipeline_instance(pipeline: "ExtractionPipeline | None") -> None:
    """Register or clear the process-wide ExtractionPipeline instance."""
    global _PIPELINE_INSTANCE
    with _PIPELINE_LOCK:
        _PIPELINE_INSTANCE = pipeline


def get_pipeline_instance() -> "ExtractionPipeline | None":
    """Return the current pipeline instance, or None if not yet initialized."""
    return _PIPELINE_INSTANCE
