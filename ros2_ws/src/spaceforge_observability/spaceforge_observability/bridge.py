"""WebSocket telemetry bridge: ROS-free core, lazy transport.

Data plane is plain JSON envelopes (see schema.py). Control messages carry a
"type" field and are never confused with envelopes:
  client -> server: {"type":"subscribe","streams":[...]} |
                    {"type":"unsubscribe","streams":[...]} |
                    {"type":"ping"} | <telemetry envelope> (publish)
  server -> client: {"type":"hello",...} | {"type":"data","envelopes":[...]} |
                    {"type":"status",...} | {"type":"pong"} |
                    {"type":"error","reason":...}

Robustness contract:
  backpressure: per-client bounded buffer (default 256); full buffers drop the
    oldest envelope and count it — a slow client never blocks others.
  connection loss: disconnects unregister the client and free its buffer; the
    bridge and all other clients continue. Clients reconnect with backoff.
  stale data: per-stream last-seen source stamps; status snapshots flag any
    stream older than its bound.
  privacy: payloads are secret-redacted before history, fan-out, or logging.
  malformed input: validated with a reason, counted, replied as error — the
    connection stays up and nothing raises past the edge.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import time
import uuid

from .schema import (
    SCHEMA_VERSION, STREAMS, is_stale, redact, stream_names, validate_envelope,
)

#: Default per-client buffer bound (envelopes) before oldest-drop backpressure.
DEFAULT_MAX_QUEUE = 256

#: Default staleness bounds in nanoseconds per stream family.
DEFAULT_MAX_AGE_NS: Dict[str, int] = {
    "odometry": 2_000_000_000,
    "target_pose": 2_000_000_000,
    "commanded": 2_000_000_000,
    "applied_command": 2_000_000_000,
    "fuel": 5_000_000_000,
    "safety_event": 30_000_000_000,
    "episode_metrics": 120_000_000_000,
    "curriculum_state": 120_000_000_000,
    "tensorboard": 300_000_000_000,
    "llm_trace": 120_000_000_000,
}

#: Retained envelopes per stream for late joiners.
DEFAULT_HISTORY = 200

#: Heartbeat ping interval in seconds.
DEFAULT_HEARTBEAT_S = 20.0

#: Push cadence for buffered envelopes to idle subscribers.
PUSH_INTERVAL_S = 0.05

#: Status broadcast cadence in push ticks.
STATUS_EVERY_TICKS = 20


@dataclass
class BridgeConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    max_queue: int = DEFAULT_MAX_QUEUE
    history_size: int = DEFAULT_HISTORY
    heartbeat_s: float = DEFAULT_HEARTBEAT_S
    max_age_ns: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MAX_AGE_NS))


@dataclass
class ClientState:
    client_id: str
    subscriptions: Optional[Set[str]]  # None means all streams
    buffer: deque
    dropped: int = 0
    connected: bool = True

    def wants(self, stream: str) -> bool:
        return self.subscriptions is None or stream in self.subscriptions


class TelemetryBridge:
    """Transport-independent fan-out core; unit-testable without sockets."""

    def __init__(self, config: BridgeConfig = BridgeConfig()) -> None:
        if config.max_queue < 1 or config.history_size < 0:
            raise ValueError("max_queue >= 1 and history_size >= 0 required")
        self.config = config
        self.clients: Dict[str, ClientState] = {}
        self.history: Dict[str, deque] = {s: deque(maxlen=config.history_size) for s in STREAMS}
        self.last_seen_ns: Dict[str, Optional[int]] = {s: None for s in STREAMS}
        self.published = 0
        self.malformed = 0
        self.connections = 0
        self._seq = 0

    # ------------------------------------------------------------ clients
    def register(self, subscriptions: Optional[List[str]] = None) -> ClientState:
        subs: Optional[Set[str]] = None
        if subscriptions is not None:
            unknown = set(subscriptions) - set(STREAMS)
            if unknown:
                raise ValueError(f"unknown streams: {sorted(unknown)}")
            subs = set(subscriptions)
        client = ClientState(client_id=f"c-{uuid.uuid4().hex[:8]}", subscriptions=subs,
                             buffer=deque(maxlen=self.config.max_queue))
        self.clients[client.client_id] = client
        self.connections += 1
        return client

    def unregister(self, client_id: str) -> None:
        client = self.clients.pop(client_id, None)
        if client is not None:
            client.connected = False
            client.buffer.clear()

    # ------------------------------------------------------------ ingest
    def ingest(self, message: Any) -> Tuple[bool, str]:
        """Validate, redact, store, and fan out one envelope or JSON string."""
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError) as exc:
                self.malformed += 1
                return False, f"not JSON: {exc}"
        ok, reason = validate_envelope(message)
        if not ok:
            self.malformed += 1
            return False, reason
        envelope = {"schema_version": SCHEMA_VERSION, "stream": message["stream"],
                    "seq": self._seq, "stamp_ns": int(message["stamp_ns"]),
                    "frame_id": message.get("frame_id", ""),
                    "payload": redact(message["payload"])}
        self._seq += 1
        stream = envelope["stream"]
        self.last_seen_ns[stream] = envelope["stamp_ns"]
        if self.config.history_size:
            self.history[stream].append(envelope)
        self.published += 1
        for client in self.clients.values():
            if not client.connected or not client.wants(stream):
                continue
            if len(client.buffer) >= self.config.max_queue:
                client.buffer.popleft()  # drop oldest: live beats late
                client.dropped += 1
            client.buffer.append(envelope)
        return True, "ok"

    def drain(self, client_id: str) -> List[Dict[str, Any]]:
        """Remove and return everything buffered for one client (transport sends it)."""
        client = self.clients.get(client_id)
        if client is None:
            return []
        items = list(client.buffer)
        client.buffer.clear()
        return items

    # ------------------------------------------------------------ status
    def stale_streams(self, now_ns: int) -> Dict[str, bool]:
        return {s: is_stale(self.last_seen_ns[s], now_ns, self.config.max_age_ns.get(s, 0))
                for s in STREAMS}

    def status_snapshot(self, now_ns: Optional[int] = None) -> Dict[str, Any]:
        now_ns = time.time_ns() if now_ns is None else now_ns
        return {"type": "status", "schema_version": SCHEMA_VERSION, "now_ns": now_ns,
                "published": self.published, "malformed": self.malformed,
                "connections": self.connections, "clients": len(self.clients),
                "dropped": {cid: c.dropped for cid, c in self.clients.items()},
                "stale": self.stale_streams(now_ns),
                "last_seen_ns": dict(self.last_seen_ns)}

    def hello(self, client: ClientState) -> Dict[str, Any]:
        return {"type": "hello", "schema_version": SCHEMA_VERSION, "client_id": client.client_id,
                "streams": stream_names(),
                "history": {s: list(self.history[s]) for s in STREAMS if self.history[s]}}

    def handle_message(self, client: ClientState, message: Any) -> Optional[Dict[str, Any]]:
        """One client message -> optional reply. Never raises on bad input."""
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError) as exc:
                self.malformed += 1
                return {"type": "error", "reason": f"not JSON: {exc}"}
        if not isinstance(message, dict):
            self.malformed += 1
            return {"type": "error", "reason": "message must be an object"}
        kind = message.get("type")
        if kind == "ping":
            return {"type": "pong"}
        if kind == "subscribe":
            try:
                client.subscriptions = self._parse_subscription(message.get("streams"))
            except ValueError as exc:
                return {"type": "error", "reason": str(exc)}
            return {"type": "ok", "subscribed": sorted(client.subscriptions or []) or "all"}
        if kind == "unsubscribe":
            remove = set(message.get("streams") or [])
            if client.subscriptions is not None:
                client.subscriptions -= remove
            return {"type": "ok", "subscribed": sorted(client.subscriptions or []) or "all"}
        if kind in (None, "publish"):
            payload = message if kind is None else message.get("envelope")
            ok, reason = self.ingest(payload)
            if not ok:
                return {"type": "error", "reason": reason}
            return None  # fan-out carries it; no reply needed
        self.malformed += 1
        return {"type": "error", "reason": f"unknown message type: {kind!r}"}

    def _parse_subscription(self, streams: Any) -> Set[str]:
        if streams is None:
            return set(STREAMS)
        unknown = set(streams) - set(STREAMS)
        if unknown:
            raise ValueError(f"unknown streams: {sorted(unknown)}")
        return set(streams)


def require_websockets():
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is required to serve the bridge (pip install websockets)") from exc
    return websockets


async def serve_async(bridge: TelemetryBridge, config: BridgeConfig):
    """Accept clients until cancelled; each connection is isolated."""
    websockets = require_websockets()
    asyncio = __import__("asyncio")

    async def handler(websocket):
        client = bridge.register()
        stop = asyncio.Event()

        async def push_loop():
            tick = 0
            while not stop.is_set():
                await asyncio.sleep(PUSH_INTERVAL_S)
                if not client.connected:
                    break
                try:
                    pending = bridge.drain(client.client_id)
                    if pending:
                        await websocket.send(json.dumps({"type": "data", "envelopes": pending}))
                    tick += 1
                    if tick % STATUS_EVERY_TICKS == 0:
                        await websocket.send(json.dumps(bridge.status_snapshot()))
                except Exception:
                    break  # connection loss: finally-block unregisters

        push_task = asyncio.ensure_future(push_loop())
        try:
            await websocket.send(json.dumps(bridge.hello(client)))
            async for raw in websocket:
                reply = bridge.handle_message(client, raw)
                if reply is not None:
                    await websocket.send(json.dumps(reply))
                pending = bridge.drain(client.client_id)
                if pending:
                    await websocket.send(json.dumps({"type": "data", "envelopes": pending}))
        finally:
            stop.set()
            bridge.unregister(client.client_id)
            await push_task

    async with websockets.serve(handler, config.host, config.port,
                                ping_interval=config.heartbeat_s, ping_timeout=config.heartbeat_s):
        await asyncio.Future()  # run until cancelled


def serve(bridge: TelemetryBridge, config: BridgeConfig) -> None:
    """Blocking entry point: spaceforge-observability --host .. --port ..."""
    import asyncio
    try:
        asyncio.run(serve_async(bridge, config))
    except KeyboardInterrupt:
        pass
