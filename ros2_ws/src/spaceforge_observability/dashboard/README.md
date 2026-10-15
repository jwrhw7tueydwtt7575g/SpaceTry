# Live dashboard

Mission console for live bridge streams and recorded replays. The offline-first
standalone replay (`spaceforge_mvp/viz/dashboard.html`) remains the no-network
option and is untouched by this package.

## Views (deep-linked via `#/mission|telemetry|events|replay`)

- **Mission**: Three.js stage (station model, gradient trail, keep-out and
  capture geometry, orbit/zoom/pan camera with presets and chaser follow),
  live HUD chips, replay transport with event markers, instrument rail.
  Without WebGL it falls back to a top-down 2D plot automatically.
- **Telemetry**: range (log), speed, and fuel chart strips with safety
  thresholds, hover readout, and the latest episode table.
- **Events**: safety log (click an entry to seek a loaded replay), planner
  traces, curriculum state, and training links.
- **Replay**: one-click scenario cards, file picker, and drag-and-drop for
  `record_episode` JSON or envelope arrays.

Keyboard: `1–4` views, `space` play/pause, `←/→` seek, `F` follow camera.
State deep-links via hash; connection status is announced via aria-live.

## Files

- `index.html` — fonts (preconnect + swap), theme-color, CDN React/ReactDOM/Three.
- `src/state.js` — pure store, trace converter, and deterministic playback
  engine (UMD, node-tested; no DOM or network).
- `src/app.js` — connection manager, views, 3D stage, charts, transport.
- `smoke/` — browser harness with real React + Three in jsdom
  (`install.sh` fetches vendor bundles + jsdom, nothing committed).

Serve locally: `python3 -m http.server` in `dashboard/`, then connect to the
bridge (`spaceforge-observability --replay <trace.json>`). Serve the repo root
instead and the scenario cards can fetch `spaceforge_mvp/viz/scenarios/*.json`
directly.
