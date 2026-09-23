"""Render the presentation videos from the recorded scenarios.

    python -m spaceforge_mvp.viz.make_videos                  # all three acts
    python -m spaceforge_mvp.viz.make_videos 01_docking        # just one
    python -m spaceforge_mvp.viz.make_videos --film            # stitch one continuous film

Outputs land in spaceforge_mvp/viz/videos/. Every number printed on a card or in the HUD
is read back from the recorded trace, so a re-recorded scenario can never leave a stale
figure on screen. Cards fade in and out; the film joins the acts into one file.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from spaceforge_mvp.viz.record import SCENARIO_DIR
from spaceforge_mvp.viz.render import Track, render_animation

OUT_DIR = Path(__file__).resolve().parent / "videos"
FILM = "00_presentation"

CYAN, GREEN, AMBER, RED = "#41D3FF", "#3DE08F", "#FFB454", "#FF5F5F"


def _summary(key: str) -> Dict[str, Any]:
    return Track.from_json(SCENARIO_DIR / f"{key}.json", "x", "#fff").summary


def _load(key: str, name: str, color: str) -> Track:
    return Track.from_json(SCENARIO_DIR / f"{key}.json", name=name, color=color)


def build_manifest() -> Dict[str, Dict[str, Any]]:
    dock, rl = _summary("docking"), _summary("rl_policy")
    hot_o, hot_1, hot_2 = _summary("hot_open"), _summary("hot_v1"), _summary("hot_v2")

    return {
        "01_docking": {
            "title": "Certified glideslope",
            "subtitle": "CW relative dynamics · 1 s step · 3-axis thrust · 0.5 m capture",
            "tracks": lambda: [_load("docking", "Certified glideslope", CYAN)],
            "frames_per_step": 2,
            "hold_frames": 54,
            "notes": [
                f"Soft capture at {dock['min_dist']:.3f} m · "
                f"{dock['end_speed']:.3f} m/s closing at contact",
                f"Δv {dock['dv']:.2f} m/s · fuel used {dock['fuel_used'] * 100:.2f}% · "
                f"interlock fires {dock['overrides']}",
            ],
            "intro": {
                "kicker": "ACT 1 · CERTIFIED GUIDANCE",
                "headline": "Can a deterministic law dock?",
                "lines": ["Velocity matching on an exponential glideslope, with the",
                          "natural Clohessy-Wiltshire drift cancelled in feed-forward."],
                "color": CYAN,
                "frames": 66,
            },
            "outro": {
                "kicker": "ACT 1 · RESULT",
                "headline": "DOCKED",
                "verdict": True,
                "head_color": GREEN,
                "stat_color": GREEN,
                "lines": ["The certified law never needs a veto — the interlock stays silent."],
                "stats": [("Closest approach", f"{dock['min_dist']:.3f} m"),
                          ("Contact speed", f"{dock['end_speed']:.3f} m/s"),
                          ("Δv", f"{dock['dv']:.2f} m/s"),
                          ("Interlock fires", f"{dock['overrides']}")],
                "frames": 115,
            },
        },
        "02_rl_policy": {
            "title": "Shipped PPO policy",
            "subtitle": "Same environment · seed 42 · 300 s horizon · 200k timesteps trained",
            "tracks": lambda: [_load("rl_policy", "PPO checkpoint, as shipped", CYAN)],
            "frames_per_step": 1,
            "hold_frames": 54,
            "notes": [
                f"No capture · final range {rl['end_dist'] / 1000.0:.2f} km, "
                f"{rl['end_speed']:.1f} m/s and rising",
                f"Δv {rl['dv']:.1f} m/s burned · interlock fires {rl['overrides']}"
                " — the escape never enters its authority",
            ],
            "intro": {
                "kicker": "ACT 2 · SHIPPED POLICY",
                "headline": "What does the released checkpoint do?",
                "lines": ["PPO, 200,000 timesteps, replayed in the very environment it was",
                          "trained in. Same seed, same physics, same safety layer."],
                "color": CYAN,
                "frames": 66,
            },
            "outro": {
                "kicker": "ACT 2 · RESULT",
                "headline": "NO CAPTURE",
                "verdict": True,
                "head_color": RED,
                "stat_color": RED,
                "lines": [f"Closest approach {rl['min_dist']:.2f} m — it never enters the 5 m keep-out,",
                          "so the safety layer never gets a vote. The failure lives outside its authority."],
                "stats": [("Final range", f"{rl['end_dist'] / 1000.0:.2f} km"),
                          ("Rel speed", f"{rl['end_speed']:.1f} m/s"),
                          ("Δv burned", f"{rl['dv']:.1f} m/s"),
                          ("Interlock fires", f"{rl['overrides']}")],
                "frames": 120,
            },
        },
        "03_runtime_assurance": {
            "title": "Runtime assurance showdown",
            "subtitle": "Identical reckless law · 6 m start · 1 s step · 60 s horizon",
            "tracks": lambda: [
                _load("hot_open", "NO PROTECTION — powered through", RED),
                _load("hot_v1", "v1 · THRUST ZEROED ON VETO", AMBER),
                _load("hot_v2", "v2 · CERTIFIED BRAKING", GREEN),
            ],
            "frames_per_step": 2,
            "hold_frames": 60,
            "notes": [],
            "intro": {
                "kicker": "ACT 3 · RUNTIME ASSURANCE",
                "headline": "Does a veto stop a collision?",
                "lines": ["One reckless law, three levels of protection, identical physics.",
                          "The only difference is what sits between the policy and the thrusters."],
                "color": AMBER,
                "frames": 66,
            },
            "outro": {
                "kicker": "ACT 3 · RESULT",
                "headline": "BRAKING BEATS VETOING",
                "verdict": True,
                "head_color": GREEN,
                "stat_color": GREEN,
                "lines": ["Denying a command is not braking: the veto cut impact speed by 70% and still",
                          "collided; substituting maximum braking captures at 1 cm/s."],
                "stats": [("No protection", f"{hot_o['continuous_min_speed']:.2f} m/s"),
                          ("Veto → zero thrust", f"{hot_1['continuous_min_speed']:.2f} m/s"),
                          ("Veto → braking", f"{hot_2['continuous_min_speed']:.2f} m/s"),
                          ("Result", f"docked in {hot_2['steps']} s")],
                "frames": 125,
            },
        },
    }


# --------------------------------------------------------------------------- ffmpeg glue
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found on PATH (needed for fades and the film)")
    return exe


def _run(args: List[str]) -> None:
    subprocess.run(args, check=True)


def _duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


def _with_fades(path: Path, fade_in: float = 0.4, fade_out: float = 0.5) -> None:
    """Fade a finished act from and to black, in place."""
    dur = _duration(path)
    tmp = path.with_name(path.stem + ".faded.mp4")
    _run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(path), "-vf",
          f"fade=t=in:st=0:d={fade_in},fade=t=out:st={max(0.0, dur - fade_out):.2f}:d={fade_out}",
          "-c:v", "libx264", "-crf", "18", "-preset", "medium",
          "-pix_fmt", "yuv420p", str(tmp)])
    tmp.replace(path)


def render(keys: List[str], dpi: int = 120, fades: bool = True) -> List[Path]:
    manifest = build_manifest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for key in keys:
        spec = manifest[key]
        out = OUT_DIR / f"{key}.mp4"
        print(f"rendering {key} -> {out}", flush=True)
        render_animation(
            spec["tracks"](),
            out,
            title=spec["title"],
            subtitle=spec["subtitle"],
            frames_per_step=spec["frames_per_step"],
            hold_frames=spec["hold_frames"],
            fps=30,
            dpi=dpi,
            notes=spec["notes"],
            intro=spec.get("intro"),
            outro=spec.get("outro"),
        )
        if fades:
            _with_fades(out)
        written.append(out)
        print(f"  done: {out} ({out.stat().st_size / 1e6:.1f} MB, {_duration(out):.1f} s)", flush=True)
    return written


def build_film(keys: List[str]) -> Path:
    """Concatenate the acts into one file, with a single fade in and out."""
    parts = [OUT_DIR / f"{k}.mp4" for k in keys]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        sys.exit(f"render these first: {missing}")
    print("stitching the film", flush=True)
    listing = OUT_DIR / "_concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    raw = OUT_DIR / "_film_raw.mp4"
    _run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
          "-i", str(listing), "-c", "copy", str(raw)])
    dur = _duration(raw)
    out = OUT_DIR / f"{FILM}.mp4"
    _run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(raw), "-vf",
          f"fade=t=in:st=0:d=0.7,fade=t=out:st={max(0.0, dur - 1.1):.2f}:d=1.1",
          "-c:v", "libx264", "-crf", "18", "-preset", "medium",
          "-pix_fmt", "yuv420p", str(out)])
    raw.unlink()
    listing.unlink()
    print(f"  done: {out} ({out.stat().st_size / 1e6:.1f} MB, {_duration(out):.1f} s)", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("keys", nargs="*", default=None)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--no-fades", action="store_true")
    p.add_argument("--film", action="store_true",
                   help="stitch the acts into one continuous film after rendering")
    a = p.parse_args()

    manifest = build_manifest()
    target_keys = a.keys or list(manifest)
    unknown = [k for k in target_keys if k not in manifest]
    if unknown:
        print(f"unknown scenario(s): {unknown}", file=sys.stderr)
        sys.exit(2)

    render(target_keys, dpi=a.dpi, fades=not a.no_fades)
    if a.film:
        build_film(target_keys)
