# mission_ctrl safety contract

mission_ctrl is a deterministic safety-critical boundary. It validates typed goals before filtering actions and owns the final actuator command. No LLM, learned policy, planner, or perception implementation is imported or trusted here.

Supported goal types are APPROACH, INSPECT, SAFE_HOLD, RETREAT, and ABORT. Every goal has a finite target position in target_lvlh metres, timeout seconds, desired closing speed in m/s, force limits in N, torque limits in N m, and a stale telemetry bound.

Telemetry uses target_lvlh relative position/velocity, fuel fraction, actuator limits, and a monotonic source timestamp. A sample older than stale_telemetry_timeout_s is unsafe.

Validation failures use stable reason codes. The action filter then emits zero force for fuel-floor and SAFE_HOLD/ABORT cases, or certified braking force opposite relative velocity for stale, keep-out, or excessive-closing-speed cases. Braking reuses the proven Python RTA law (spaceforge_mvp.control.rta.certified_brake_action: maximum deceleration anti-parallel to velocity, provably never increasing speed), clipped into this boundary's per-axis force authority. Commands are always clipped to configured actuator limits. Every substituted or clipped action emits one OverrideEvent.

Goals arrive as JSON via GoalSpec.from_dict (strict: unknown types, non-finite vectors, and bad timeouts raise ValueError before validation). The ROS node never lets a malformed goal replace the last valid one and never crashes on hostile input, so no producer can bypass the boundary with a bad message.

The boundary output is a physical force/torque command in SI units. Policy and goal producers may request commands but cannot bypass validation or filtering.
