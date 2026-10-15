"""Live bridge round-trip over localhost. Skipped if sockets are unavailable."""
import asyncio
import json
import pytest

websockets = pytest.importorskip("websockets", reason="websockets unavailable; live bridge not claimed")

from spaceforge_observability.bridge import BridgeConfig, TelemetryBridge


def envelope():
    return {"schema_version": 1, "stream": "fuel", "stamp_ns": 7,
            "frame_id": "", "payload": {"fraction": 0.5}}


def test_live_round_trip_with_replay_history():
    async def run():
        config = BridgeConfig(host="127.0.0.1", port=18765, max_queue=16, history_size=8)
        bridge = TelemetryBridge(config)
        assert bridge.ingest(envelope())[0]

        async def handler(websocket):
            client = bridge.register()
            try:
                await websocket.send(json.dumps(bridge.hello(client)))
                async for raw in websocket:
                    reply = bridge.handle_message(client, raw)
                    if reply is not None:
                        await websocket.send(json.dumps(reply))
                    for pending in [bridge.drain(client.client_id)]:
                        if pending:
                            await websocket.send(json.dumps({"type": "data", "envelopes": pending}))
            finally:
                bridge.unregister(client.client_id)

        try:
            server = await websockets.serve(handler, config.host, config.port)
        except OSError:
            pytest.skip("localhost sockets unavailable in this sandbox")
        try:
            async with websockets.connect(f"ws://{config.host}:{config.port}") as ws:
                hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                assert hello["type"] == "hello" and hello["schema_version"] == 1
                assert hello["history"]["fuel"][0]["payload"] == {"fraction": 0.5}
                await ws.send(json.dumps({"type": "ping"}))
                assert json.loads(await asyncio.wait_for(ws.recv(), timeout=5)) == {"type": "pong"}
                await ws.send(json.dumps({"type": "subscribe", "streams": ["fuel"]}))
                reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                assert reply["type"] == "ok"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())
