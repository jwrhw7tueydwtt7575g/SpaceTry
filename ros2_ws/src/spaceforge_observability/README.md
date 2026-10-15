# SpaceForge observability

Live telemetry alongside the standalone replay — which is untouched and keeps
working (`spaceforge_mvp/viz/dashboard.html`, built by `spaceforge_mvp.viz.dashboard`).

## Schema first (TELEMETRY_SCHEMA.md)

Every message is `{schema_version: 1, stream, seq, stamp_ns, frame_id, payload}`
over ten streams: odometry, target_pose, commanded, applied_command, fuel,
safety_event, episode_metrics, curriculum_state, tensorboard, llm_trace.
Unknown streams, version mismatches, and invalid payloads are rejected with a
reason and counted — never crashing the bridge.

## Bridge (`spaceforge-observability`)

`spaceforge-observability --replay <artifact.json> [--replay-speed 4] [--port 8765]`
streams recorded artifacts with no ROS or Gazebo: scenario/telemetry traces,
rl_trainer reports, HPO reports, LLM decision traces, curriculum states.
`--ros` additionally attaches ROS topic sources (needs a sourced ROS 2 host).

Robustness, all tested without sockets: per-client bounded buffers drop the
oldest envelope and count it (slow clients never block others); disconnects
unregister and free buffers; status snapshots flag stale streams against
per-stream age bounds with heartbeat pings; secrets (api_key/token/secret/…)
are redacted before history, fan-out, or logging; malformed input gets an error
reply and a counter increment while the connection stays up.

## Dashboard (`dashboard/`)

React panels + Three.js 3D trajectory at `dashboard/index.html` (CDN React,
ReactDOM, Three). Live mode connects to the bridge with exponential-backoff
reconnect, per-stream filter chips (unsubscribe), stale banners, and scored
safety-event feeds. Replay mode loads envelope JSON files via file picker.
Pure state logic in `dashboard/src/state.js` is UMD and node-tested
(`state.test.js`); without CDN access the page degrades to panels + replay
instead of failing. Serve locally: `python3 -m http.server` in `dashboard/`,
then open the page and connect to the bridge.
