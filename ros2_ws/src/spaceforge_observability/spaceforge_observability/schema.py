"""Versioned telemetry schema for the observability layer.

Every message on the wire (live bridge or replay file) is an envelope:
  {schema_version, stream, seq, stamp_ns, frame_id, payload}
Timestamps are integer nanoseconds (source clock); frame_id is required on
spatial streams. Payloads are validated per stream; secrets are redacted on
egress. Malformed envelopes are rejected with a reason, never raised past the
bridge edge.
"""
from typing import Any, Dict, List, Optional, Tuple

#: Wire schema version; bump on any envelope/payload change.
SCHEMA_VERSION = 1

#: Streams the bridge and dashboard understand, in dependency order.
STREAMS: Tuple[str, ...] = (
    "odometry",
    "target_pose",
    "commanded",
    "applied_command",
    "fuel",
    "safety_event",
    "episode_metrics",
    "curriculum_state",
    "tensorboard",
    "llm_trace",
)

#: Streams that must carry a non-empty frame_id.
SPATIAL_STREAMS = frozenset({"odometry", "target_pose", "commanded", "applied_command"})

#: Case-insensitive key substrings treated as secrets and redacted on egress.
SECRET_SUBSTRINGS: Tuple[str, ...] = (
    "api_key", "apikey", "token", "secret", "password", "passwd",
    "authorization", "bearer", "private_key", "client_secret",
)

REDACTED = "***REDACTED***"


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value == value and abs(value) != float("inf")


def _is_vector(value: Any, size: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == size \
        and all(_is_finite_number(v) for v in value)


def _is_unit_quaternion(value: Any) -> bool:
    if not _is_vector(value, 4):
        return False
    norm = sum(float(v) ** 2 for v in value) ** 0.5
    return 0.9 <= norm <= 1.1


_VALIDATORS = {
    "odometry": lambda p: isinstance(p, dict) and _is_vector(p.get("position_m"), 3)
                and _is_vector(p.get("velocity_mps"), 3),
    "target_pose": lambda p: isinstance(p, dict) and _is_vector(p.get("translation_m"), 3)
                and _is_unit_quaternion(p.get("quaternion_xyzw"))
                and _is_finite_number(p.get("confidence", 1.0)),
    "commanded": lambda p: isinstance(p, dict) and _is_vector(p.get("force_n"), 3)
                and _is_vector(p.get("torque_nm"), 3),
    "applied_command": lambda p: isinstance(p, dict) and _is_vector(p.get("force_n"), 3)
                and _is_vector(p.get("torque_nm"), 3)
                and (p.get("override") is None or isinstance(p.get("override"), str)),
    "fuel": lambda p: isinstance(p, dict) and _is_finite_number(p.get("fraction"))
                and 0.0 <= float(p["fraction"]) <= 1.0,
    "safety_event": lambda p: isinstance(p, dict) and isinstance(p.get("event_id"), str)
                and isinstance(p.get("reason_code"), str) and isinstance(p.get("reason"), str),
    "episode_metrics": lambda p: isinstance(p, dict) and isinstance(p.get("success"), bool)
                and _is_finite_number(p.get("steps", 0)),
    "curriculum_state": lambda p: isinstance(p, dict) and isinstance(p.get("current_stage"), str)
                and isinstance(p.get("episode_index"), int),
    "tensorboard": lambda p: isinstance(p, dict) and isinstance(p.get("url"), str)
                and isinstance(p.get("run_dir"), str),
    "llm_trace": lambda p: isinstance(p, dict) and isinstance(p.get("trace_id"), str)
                and isinstance(p.get("status"), str),
}


def make_envelope(stream: str, payload: Dict[str, Any], stamp_ns: int,
                  frame_id: str = "", seq: int = 0) -> Dict[str, Any]:
    """Build an envelope; raises ValueError on any contract violation."""
    ok, reason = validate_envelope({"schema_version": SCHEMA_VERSION, "stream": stream,
                                    "seq": seq, "stamp_ns": stamp_ns,
                                    "frame_id": frame_id, "payload": payload})
    if not ok:
        raise ValueError(f"invalid envelope: {reason}")
    return {"schema_version": SCHEMA_VERSION, "stream": stream, "seq": int(seq),
            "stamp_ns": int(stamp_ns), "frame_id": frame_id, "payload": payload}


def validate_envelope(message: Any) -> Tuple[bool, str]:
    """Check wire contract; returns (ok, reason), never raises."""
    if not isinstance(message, dict):
        return False, "envelope must be an object"
    if message.get("schema_version") != SCHEMA_VERSION:
        return False, f"unsupported schema_version: {message.get('schema_version')!r}"
    stream = message.get("stream")
    if stream not in STREAMS:
        return False, f"unknown stream: {stream!r}"
    stamp = message.get("stamp_ns")
    if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0:
        return False, "stamp_ns must be a non-negative integer"
    if not isinstance(message.get("seq", 0), int):
        return False, "seq must be an integer"
    frame_id = message.get("frame_id", "")
    if stream in SPATIAL_STREAMS and not frame_id:
        return False, f"stream {stream} requires a non-empty frame_id"
    payload = message.get("payload")
    try:
        valid = bool(_VALIDATORS[stream](payload))
    except Exception:
        valid = False
    if not valid:
        return False, f"invalid payload for stream {stream}"
    return True, "ok"


def redact(obj: Any) -> Any:
    """Deep-copy with secret values replaced; safe on any JSON-like structure."""
    if isinstance(obj, dict):
        return {k: (REDACTED if any(s in str(k).lower() for s in SECRET_SUBSTRINGS) else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def is_stale(last_seen_ns: Optional[int], now_ns: int, max_age_ns: int) -> bool:
    """True when a stream has never been seen or is older than its bound."""
    if last_seen_ns is None:
        return True
    return (now_ns - last_seen_ns) > max_age_ns


def stream_names() -> List[str]:
    return list(STREAMS)
