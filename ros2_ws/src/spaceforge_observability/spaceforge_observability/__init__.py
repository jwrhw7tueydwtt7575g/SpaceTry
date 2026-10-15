"""Live observability: versioned telemetry schema, replay, and WebSocket bridge.

The existing standalone dashboard (spaceforge_mvp.viz.dashboard) is untouched
and keeps working; this package adds the live path alongside it.
"""
from .schema import (
    SCHEMA_VERSION, STREAMS, SECRET_SUBSTRINGS, is_stale, make_envelope,
    redact, stream_names, validate_envelope,
)
from .replay import (
    ReplayPlayer, load_curriculum_state, load_decision_trace, load_hpo_report,
    load_trace_frames, load_training_report, merge_sorted,
)
from .bridge import (
    BridgeConfig, ClientState, TelemetryBridge, require_websockets, serve, serve_async,
)

__all__ = [
    "SCHEMA_VERSION", "STREAMS", "SECRET_SUBSTRINGS", "is_stale", "make_envelope",
    "redact", "stream_names", "validate_envelope",
    "ReplayPlayer", "load_curriculum_state", "load_decision_trace", "load_hpo_report",
    "load_trace_frames", "load_training_report", "merge_sorted",
    "BridgeConfig", "ClientState", "TelemetryBridge", "require_websockets", "serve", "serve_async",
]
