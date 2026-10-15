"""Bridge entry point: serve live telemetry with optional replay and ROS sources.

Examples:
  spaceforge-observability --replay spaceforge_mvp/viz/scenarios/docking.json
  spaceforge-observability --replay runs/demo/telemetry.json --replay-speed 4
  spaceforge-observability --port 8765 --max-queue 512
  spaceforge-observability --ros   # also attach ROS topic sources (needs sourced ROS 2)
"""
import argparse
import asyncio
from .bridge import BridgeConfig, TelemetryBridge, serve_async
from .replay import (
    ReplayPlayer, load_curriculum_state, load_decision_trace, load_hpo_report,
    load_trace_frames, load_training_report, merge_sorted,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceForge telemetry WebSocket bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--replay", action="append", default=[],
                        help="JSON artifact to stream (repeatable): scenario/telemetry trace, "
                             "training_report.json, hpo_report.json, decision trace, curriculum state")
    parser.add_argument("--replay-speed", type=float, default=4.0)
    parser.add_argument("--max-queue", type=int, default=256)
    parser.add_argument("--ros", action="store_true",
                        help="Attach ROS topic sources in a background thread (needs ROS 2)")
    return parser


def load_replay_file(path: str):
    """First matching loader wins; raises ValueError when nothing recognizes it."""
    loaders = (load_trace_frames, load_training_report, load_hpo_report,
               load_decision_trace, load_curriculum_state)
    errors = []
    for loader in loaders:
        try:
            return loader(path)
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise ValueError(f"unrecognized replay file {path}: {'; '.join(errors)}")


async def _publish_replay(bridge: TelemetryBridge, events, speed: float):
    player = ReplayPlayer(events, speed=speed)
    for delay, envelope in player:
        if delay:
            await asyncio.sleep(delay)
        bridge.ingest(envelope)


def _start_ros_source(bridge: TelemetryBridge):
    """Spin the ROS source node in a daemon thread; raises RuntimeError without ROS."""
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        raise RuntimeError("ROS 2 (rclpy) is required for --ros") from exc
    import threading
    from . import ros_source

    rclpy.init()
    node_holder = {}

    class SourceNode(Node):
        def __init__(self):
            super().__init__("observability_source")
            from geometry_msgs.msg import WrenchStamped
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry, "odometry",
                                     lambda msg: bridge.ingest(ros_source.odometry_from_msg(msg)), 10)
            self.create_subscription(WrenchStamped, "commanded_wrench",
                                     lambda msg: bridge.ingest(ros_source.wrench_envelope("commanded", msg)), 10)
            self.create_subscription(WrenchStamped, "applied_wrench",
                                     lambda msg: bridge.ingest(ros_source.wrench_envelope("applied_command", msg)), 10)

    node_holder["node"] = SourceNode()
    thread = threading.Thread(target=rclpy.spin, args=(node_holder["node"],), daemon=True)
    thread.start()
    return node_holder["node"]


async def _amain(args) -> None:
    config = BridgeConfig(host=args.host, port=args.port, max_queue=args.max_queue)
    bridge = TelemetryBridge(config)
    tasks = []
    if args.replay:
        events = merge_sorted(*(load_replay_file(p) for p in args.replay))
        print(f"replaying {len(events)} envelopes from {len(args.replay)} file(s) at {args.replay_speed}x")
        tasks.append(asyncio.ensure_future(_publish_replay(bridge, events, args.replay_speed)))
    node = _start_ros_source(bridge) if args.ros else None
    print(f"serving telemetry bridge on ws://{config.host}:{config.port}")
    try:
        await serve_async(bridge, config)
    finally:
        for task in tasks:
            task.cancel()
        if node is not None:
            import rclpy
            rclpy.shutdown()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.replay_speed <= 0 or args.max_queue < 1:
        raise SystemExit("--replay-speed must be positive and --max-queue at least 1")
    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
