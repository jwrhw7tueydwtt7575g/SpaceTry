# SpaceForge mission_ctrl

mission_ctrl is the deterministic safety-critical boundary for goals and actuator commands. It supports APPROACH, INSPECT, SAFE_HOLD, RETREAT, and ABORT. Goal validation is separate from action filtering; every override is recorded as an OverrideEvent.

The core is ROS-independent and testable with typed Python objects. The optional ROS node accepts JSON goals, odometry telemetry, and commanded WrenchStamped messages, then publishes only the filtered WrenchStamped and JSON override events.

No LLM, policy, planner, perception, or learned controller is imported in this package (the only external import is the braking law from spaceforge_mvp.control.rta). Invalid or unsafe goals are rejected with stable reason codes and produce a safe hold or braking.

Tests (all ROS-independent except the dependency-gated live node test): typed validation per goal type, every ReasonCode reachable, 500-trial hostile property test (inf/nan commands, random telemetry, determinism, actuator bounds, override-event audit), brake equivalence with the RTA law, wire-boundary parsing, and message-payload schemas.
