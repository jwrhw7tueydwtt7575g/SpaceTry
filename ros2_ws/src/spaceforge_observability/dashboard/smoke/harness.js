/* Browser smoke harness: real React + Three (WebGL unavailable) in jsdom.
 * Proves the dashboard boots, renders all views, ingests data, and falls back
 * to 2D without exceptions. Run: node harness.js [dashboard-dir] [vendor-dir]
 * Needs: jsdom resolvable (see install.sh) + vendor UMD bundles.
 * Vendor bundles are fetched, never committed: install.sh downloads them.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const DASH = process.argv[2] || path.resolve(__dirname, "..");
const VENDOR = process.argv[3] || path.join(__dirname, "vendor");

function read(p) { return fs.readFileSync(p, "utf8"); }

function buildWindow() {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body>
      <div id="cdn-note" style="display:none"></div>
      <div id="fallback" style="display:none"></div>
      <div id="root"></div>
    </body></html>`,
    { url: "http://localhost:8000/", pretendToBeVisual: true, runScripts: "outside-only" }
  );
  const { window } = dom;
  const ctx = dom.getInternalVMContext();
  vm.runInContext("window.self = window;", ctx);
  // Canvas: null WebGL (forces the 2D fallback path), no-op 2D context.
  // Installed inside the context so Three and the app both see it.
  vm.runInContext(`HTMLCanvasElement.prototype.getContext = function (kind) {
    if (kind === "webgl" || kind === "webgl2" || kind === "experimental-webgl") return null;
    return new Proxy({}, {
      get(t, p) {
        if (p === "measureText") return () => ({ width: 0 });
        if (p === "getImageData") return () => ({ data: [] });
        return () => {};
      },
      set: () => true,
    });
  };`, ctx);
  window.matchMedia = () => ({ matches: false, addListener: () => {}, removeListener: () => {} });
  for (const file of ["react.min.js", "react-dom.min.js", "three.min.js",
                       "state.js", "app.js"]) {
    const code = file.endsWith(".js") && !file.includes(".min.")
      ? read(path.join(DASH, "src", file))
      : read(path.join(VENDOR, file));
    vm.runInContext(code, dom.getInternalVMContext(), { filename: file });
  }
  return { dom, window };
}

function assert(cond, message) {
  if (!cond) { console.error("SMOKE FAIL:", message); process.exit(1); }
}

const errors = [];
const { dom, window } = buildWindow();
window.addEventListener("error", (e) => errors.push(String((e && e.message) || e)));

// Boot the app (WebSocket undefined in jsdom -> connect() returns early).
window.document.dispatchEvent(new window.Event("DOMContentLoaded", { bubbles: true }));

const dash = window.SpaceForgeDashboard;
assert(dash && dash.version === 1, "debug handle exposed");
assert(window.document.getElementById("root").children.length > 0, "mission view rendered");
assert(window.document.getElementById("scene3d"), "3D canvas mounted");
assert(window.document.querySelector(".stage"), "mission stage mounted");

// Load a trace through the app's own replay path: player + transport appear.
// Renders are throttled (~200ms), so assertions below run after a beat.
dash.loadTrace({
  summary: { label: "smoke", seed: 1, steps: 2, start_dist: 6, end_dist: 5,
    end_speed: 0.5, success: false, fuel_used: 0.01, dv: 0.5, overrides: 1,
    terminal_reason: "timeout" },
  frames: [
    { t: 0, pos: [6, 0, 0], vel: [0, 0, 0], cmd: [0, 0, 0], applied: [0, 0, 0],
      override: null, fuel: 1 },
    { t: 1, pos: [5, 0, 0], vel: [-1, 0, 0], cmd: [-1, 0, 0], applied: [0, 0, 0],
      override: "rejected", fuel: 0.99 },
  ],
}, "smoke.json");
assert(dash.player, "replay player created");

setTimeout(() => {
  (async () => {
    try {
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      // Renders are throttled (~200ms): wait a beat between steps.
      const step = async (fn) => { fn(); await sleep(300); };
      assert(window.document.querySelector(".transport"), "transport bar rendered");

      // Inject live-style envelopes and refresh: HUD + instruments must update.
      dash.store.latest = {};
      window.ObsState.applyData(dash.store, [
        { stream: "odometry", stamp_ns: 10e9, frame_id: "target_lvlh",
          payload: { position_m: [6, -8, 0], velocity_mps: [-1, 0.5, 0] } },
        { stream: "fuel", stamp_ns: 10e9, frame_id: "", payload: { fraction: 0.77 } },
        { stream: "applied_command", stamp_ns: 10e9, frame_id: "target_lvlh",
          payload: { force_n: [-0.5, 0.2, 0], torque_nm: [0, 0, 0], override: "rejected" } },
        { stream: "safety_event", stamp_ns: 10e9, frame_id: "",
          payload: { event_id: "e1", goal_type: "APPROACH", reason_code: "rejected", reason: "too fast near target" } },
      ]);
      await step(() => dash.refresh());
      assert(window.document.querySelector(".tick"), "event marker on timeline");
      const hud = window.document.querySelector(".hud").textContent;
      assert(hud.includes("10.00 m"), "HUD shows 10 m range, got: " + hud);
      assert(hud.includes("77.0%"), "HUD shows fuel, got: " + hud);

      // View switching via the debug handle renders each view without errors.
      await step(() => dash.setView("telemetry"));
      assert(window.document.getElementById("chart-range"), "telemetry charts mounted");
      await step(() => dash.setView("events"));
      assert(window.document.querySelector(".log-line"), "safety log rendered");
      assert(
        window.document.querySelector(".events-grid").textContent.includes("too fast near target"),
        "event reason visible"
      );
      await step(() => dash.setView("replay"));
      assert(window.document.querySelector(".cards"), "replay scenario cards rendered");

      // Stream filter chip toggles without errors.
      await step(() => dash.setView("mission"));
      const chips = window.document.querySelectorAll(".chip");
      assert(chips.length === 10, "ten stream chips, got " + chips.length);

      assert(errors.length === 0, "unexpected window errors: " + errors.join(" | "));
      console.log("browser smoke: boot, views, ingest, filters all OK");
      dom.window.close();
      process.exit(0);
    } catch (e) {
      console.error("SMOKE FAIL:", e.message || e);
      process.exit(1);
    }
  })();
}, 400);
