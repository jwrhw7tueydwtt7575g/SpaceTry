# SpaceForge observability telemetry schema v1

All wire traffic (live bridge, replay files, dashboard state) uses one envelope:

  {schema_version: 1, stream, seq, stamp_ns, frame_id, payload}

`stamp_ns` is an integer source timestamp in nanoseconds; `frame_id` is required
on spatial streams. Unknown streams, version mismatches, and invalid payloads
are rejected with a reason and counted, never crashing the bridge.

## Streams and payloads

| stream | payload | frame required |
|---|---|---|
| odometry | position_m[3], velocity_mps[3] | yes |
| target_pose | translation_m[3], quaternion_xyzw[4] (~unit), confidence | yes |
| commanded | force_n[3], torque_nm[3] (requested) | yes |
| applied_command | force_n[3], torque_nm[3], override (reason string or null) | yes |
| fuel | fraction in [0, 1] | no |
| safety_event | event_id, goal_type, reason_code, reason, commanded/applied forces | no |
| episode_metrics | label, seed, steps, start/end dist, end speed, success, fuel_used, dv, overrides, terminal_reason (+ optional success_rate, episodes, objective_value, best_params) | no |
| curriculum_state | current_stage, episode_index, window_episodes, success_rate, transitions | no |
| tensorboard | url, run_dir | no |
| llm_trace | trace_id, prompt_version, schema_version, provider, status, goal_type, reason, token_count, cost_usd, attempts | no |

## Control messages (bridge only, never envelopes)

Client→server: subscribe/unsubscribe ({streams}), ping, or a raw envelope to
publish. Server→client: hello (streams + recent history), data ({envelopes}),
status (counters, per-stream staleness, drop counts), pong, ok, error{reason}.

## Robustness rules

Backpressure: per-client bounded buffers drop the oldest envelope and count it;
slow clients never block others. Connection loss unregisters the client and
frees its buffer. Stale streams are flagged in status snapshots against
per-stream age bounds. Secrets (api_key/token/secret/password/authorization/
bearer/private_key, case-insensitive) are redacted before history, fan-out, or
logging. Malformed input yields an error reply and a counter increment.
