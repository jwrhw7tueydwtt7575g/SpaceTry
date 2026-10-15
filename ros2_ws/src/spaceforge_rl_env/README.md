# Blueprint rl_env

This package freezes the 18-dimensional observation and 6-dimensional normalized force/torque action contract in contract.py. It runs on the pure Python SpacecraftEnv through MockTransport for fast deterministic unit tests, or on ROS 2 topics through RosTransportAdapter.

Observation ordering:
1. relative position x,y,z in target LVLH frame, metres divided by 20
2. relative velocity vx,vy,vz in target LVLH frame, m/s divided by 2
3. target orientation quaternion xyzw
4. vehicle orientation quaternion xyzw
5. vehicle angular velocity x,y,z, rad/s divided by 1
6. fuel fraction

Action ordering is normalized [Fx,Fy,Fz,Tx,Ty,Tz]. Force limits are 1 N per axis and torque limits are 1 N m per axis by default; conversion to physical units happens before transport. Actions are clipped and reported.

Each step publishes one WrenchStamped, waits for a strictly newer simulator timestamp, and terminates safely on missing or stale sensor data. max_steps produces truncation. reset calls the adapter simulator reset contract. No policy, safety, perception, or LLM logic is in this package.

MockTransport models two Phase 6 domain effects explicitly: `mass_kg`
(force-to-acceleration conversion) and `latency_ms` as a whole-control-step
delay line (`delay_steps = ceil(latency_ms/1000/dt)`, zero by default), plus
seeded `thrust_noise_std`. At dt=1 s any nonzero latency costs one step
(conservative quantization); use backend dt <= 0.01 s for realistic 1-10 ms
studies. All draws come from a per-episode RNG seeded by `reset_simulator`,
so identical seeds replay identically.

Before RL training, run:
  PYTHONPATH=ros2_ws/src/spaceforge_rl_env python3 -m spaceforge_rl_env.classical
The command must report success for the deterministic 1-D 5 m glideslope approach. Gazebo/ROS integration tests are environment-gated and are not claimed on hosts without those tools.
