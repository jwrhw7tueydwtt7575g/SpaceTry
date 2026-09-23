"""Render individual frames of a recorded trace to PNG for visual checking.

Usage:
    python -m spaceforge_mvp.viz.preview docking 0 30 60 105 --title "..." --out /tmp/frames
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spaceforge_mvp.viz.record import SCENARIO_DIR
from spaceforge_mvp.viz.render import AX_RECT, BG, FIG_H_IN, FIG_W_IN, Track, _draw_frame


def render_frames(
    scenario: str,
    indices,
    out_dir: Path,
    title: str = "SpaceForge",
    subtitle: str = "preview",
    notes=None,
    dpi: int = 110,
) -> list:
    data = Track.from_json(SCENARIO_DIR / f"{scenario}.json", name=scenario, color="#41D3FF")
    data._xy = data.xy()  # type: ignore[attr-defined]
    rng = np.random.default_rng(11)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    scale = None
    for i in indices:
        fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=dpi)
        fig.patch.set_facecolor(BG)
        ax = fig.add_axes(AX_RECT)
        ax.set_facecolor(BG)
        scale = _draw_frame(fig, ax, [data], i, rng, scale, title, subtitle, notes, dpi, False)
        out = out_dir / f"{scenario}_{i:03d}.png"
        fig.savefig(out, facecolor=BG)
        plt.close(fig)
        written.append(out)
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scenario")
    p.add_argument("indices", nargs="+", type=int)
    p.add_argument("--title", default="SpaceForge RPO — certified glideslope")
    p.add_argument("--subtitle", default="Clohessy-Wiltshire relative dynamics · 1 s step")
    p.add_argument("--out", default=str(SCENARIO_DIR.parent / "preview"))
    a = p.parse_args()
    for path in render_frames(a.scenario, a.indices, Path(a.out), title=a.title, subtitle=a.subtitle):
        print(path)
