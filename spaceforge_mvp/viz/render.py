"""Broadcast-style renderer: telemetry traces -> MP4 animations.

Design goals: legible at a glance, no default-matplotlib look, and every number
on screen traceable to a recorded frame (no invented motion). Layout is a
mission-display scene on the left and a telemetry HUD on the right; the camera
eases in and out automatically as the range closes (log-eased, so a 3 km escape
and a 0.3 m capture both stay watchable).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

# --------------------------------------------------------------------------- theme
BG = "#060910"
PANEL = "#0C141F"
PANEL_EDGE = "#1D2C3F"
GRID = "#15202D"
GRID_STRONG = "#213650"
FG = "#E8EFF7"
MUTED = "#7A8DA6"
FAINT = "#46586E"
CYAN = "#41D3FF"
GREEN = "#3DE08F"
AMBER = "#FFB454"
RED = "#FF5F5F"
WHITE = "#FFFFFF"

MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"

SUCCESS_DIST = 0.5
KEEP_OUT = 5.0

# --------------------------------------------------------------------------- layout
FIG_W_IN, FIG_H_IN = 16.0, 9.0
AX_RECT: Tuple[float, float, float, float] = (0.022, 0.045, 0.655, 0.845)
AX_W_IN = FIG_W_IN * AX_RECT[2]
AX_H_IN = FIG_H_IN * AX_RECT[3]
AX_ASPECT = AX_W_IN / AX_H_IN

HUD_X = 0.695
HUD_W = 0.285

# Oblique projection: radial -> screen up, in-track -> screen right, cross-track
# -> up-left as a depth cue. In-plane geometry stays undistorted (aspect equal),
# so the capture / keep-out rings are true circles.
Z_RIGHT = -0.36
Z_UP = 0.24

HH_MIN, HH_MAX = 0.75, 5200.0
CAM_EASE = 0.085


def project(p: Sequence[float]) -> Tuple[float, float]:
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    return (y + Z_RIGHT * z, x + Z_UP * z)


def _fmt_meters(v: float) -> str:
    if v >= 1000.0:
        return f"{v / 1000.0:.2f} km"
    if v >= 100.0:
        return f"{v:.1f} m"
    if v >= 10.0:
        return f"{v:.2f} m"
    if v >= 1.0:
        return f"{v:.3f} m"
    return f"{v * 100:.1f} cm"


def _nice_step(span: float) -> float:
    for c in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0):
        if span / c <= 6.5:
            return c
    return 5000.0


def _nice_bar(span_world: float) -> float:
    """Largest round length that fits in ``span_world`` (for the scale bar)."""
    nice = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0)
    best = nice[0]
    for c in nice:
        if c <= span_world:
            best = c
        else:
            break
    return best


@dataclass
class Track:
    """One recorded episode to draw."""

    name: str
    color: str
    summary: Dict[str, Any]
    frames: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path, name: str, color: str) -> "Track":
        data = json.loads(Path(path).read_text())
        return cls(name=name, color=color, summary=data["summary"], frames=data["frames"])

    def xy(self) -> List[Tuple[float, float]]:
        return [project(f["pos"]) for f in self.frames]


# --------------------------------------------------------------------------- HUD helpers
def _text(fig, x: float, y: float, s: str, *, size=10.0, color=FG, family=SANS, ha="left", weight="normal", alpha=1.0) -> None:
    fig.text(x, y, s, size=size, color=color, family=family, ha=ha, va="center", weight=weight, alpha=alpha)


def _round_rect(fig, x, y, w, h, *, fc=PANEL, ec=PANEL_EDGE, lw=0.9) -> None:
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=2))


# --------------------------------------------------------------------------- scene drawing
def _draw_stars(ax, rng: np.random.Generator, n: int = 240) -> None:
    stars = rng.random((n, 2))
    sizes = rng.random(n) ** 3 * 3.0 + 0.15
    alphas = rng.random(n) * 0.4 + 0.07
    ax.scatter(
        stars[:, 0], stars[:, 1], s=sizes,
        c=[(1.0, 1.0, 1.0, float(a)) for a in alphas],
        transform=ax.transAxes, zorder=0, linewidths=0,
    )


def _draw_grid(ax, cx: float, cy: float, step: float, hw: float, hh: float) -> None:
    x0, x1 = cx - hh * 1.4, cx + hh * 1.4
    y0, y1 = cy - hw * 1.2, cy + hw * 1.2
    i0, i1 = int(math.floor(x0 / step)) - 1, int(math.ceil(x1 / step)) + 1
    j0, j1 = int(math.floor(y0 / step)) - 1, int(math.ceil(y1 / step)) + 1

    for i in range(i0, i1 + 1):
        if abs(i) > 60:
            continue
        ua, va = project((i * step, y0, 0.0))
        ub, vb = project((i * step, y1, 0.0))
        strong = i == 0
        ax.plot([ua, ub], [va, vb], color=GRID_STRONG if strong else GRID,
                lw=1.0 if strong else 0.7, zorder=1, solid_capstyle="butt")
    for j in range(j0, j1 + 1):
        if abs(j) > 60:
            continue
        ua, va = project((x0, j * step, 0.0))
        ub, vb = project((x1, j * step, 0.0))
        strong = j == 0
        ax.plot([ua, ub], [va, vb], color=GRID_STRONG if strong else GRID,
                lw=1.0 if strong else 0.7, zorder=1, solid_capstyle="butt")


def _draw_ring(ax, radius: float, color: str, label: str, dash, alpha: float, scale: float) -> None:
    px = radius * scale
    if px < 34.0:
        return
    ax.add_patch(Circle((0.0, 0.0), radius, fill=False, ec=color, lw=1.5, ls=dash, alpha=alpha, zorder=2))
    ax.annotate(
        label, xy=(0.0, radius * 0.7071), xytext=(7, 8), textcoords="offset points",
        color=color, alpha=min(1.0, alpha + 0.3), fontsize=9.5, family=SANS, zorder=3,
    )


def _draw_target(ax, px: Tuple[float, float], px_per_m: float, size_px: float = 30.0) -> None:
    """Fixed-pixel-size target station icon (body + two solar wings)."""
    u, v = px
    s = size_px / max(px_per_m, 1e-9)
    ax.add_patch(Rectangle((u - s * 0.11, v - s * 0.20), s * 0.22, s * 0.40,
                           fc="#DCE7F3", ec="#93A8BE", lw=0.7, zorder=7))
    for sgn in (-1, 1):
        x0 = u + sgn * s * 0.13 if sgn > 0 else u - s * 0.13 - s * 0.42
        ax.add_patch(Rectangle((x0, v - s * 0.10), s * 0.42, s * 0.20,
                               fc="#17304F", ec=CYAN, lw=0.8, zorder=7))
    ax.plot([u - s * 0.72, u + s * 0.72], [v, v], color=CYAN, lw=0.6, alpha=0.4, zorder=6)


def _draw_craft(ax, px: Tuple[float, float], angle_deg: float, color: str, thrust: float,
                px_per_m: float, size_px: float = 26.0) -> None:
    """Fixed-pixel-size chaser icon with a thrust plume, drawn at world point px."""
    u, v = px
    s = size_px / max(px_per_m, 1e-9)
    ang = math.radians(angle_deg)

    def rot(dx, dy):
        return (u + dx * math.cos(ang) - dy * math.sin(ang), v + dx * math.sin(ang) + dy * math.cos(ang))

    nose = rot(s * 0.58, 0.0)
    left = rot(-s * 0.34, s * 0.36)
    tail = rot(-s * 0.16, 0.0)
    right = rot(-s * 0.34, -s * 0.36)
    ax.add_patch(Polygon([nose, left, tail, right], closed=True, fc=color, ec=WHITE, lw=0.7, zorder=10))

    if thrust > 0.012:
        plume = s * (0.35 + 1.9 * min(1.0, thrust / 0.6))
        tip = rot(-s * 0.16 - plume, 0.0)
        bl = rot(-s * 0.16, s * 0.20)
        br = rot(-s * 0.16, -s * 0.20)
        ax.add_patch(Polygon([tail, bl, tip, br], closed=True, fc=AMBER, ec="none", alpha=0.9, zorder=9))


# --------------------------------------------------------------------------- main renderer
def render_animation(
    tracks: List[Track],
    out_path: Path,
    title: str,
    subtitle: str,
    fps: int = 30,
    frames_per_step: int = 2,
    hold_frames: int = 45,
    dpi: int = 110,
    notes: Optional[List[str]] = None,
    intro: Optional[Dict[str, Any]] = None,
    outro: Optional[Dict[str, Any]] = None,
) -> Path:
    """Render one or more recorded episodes to an MP4, with optional title/result cards."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for t in tracks:
        t._xy = t.xy()  # type: ignore[attr-defined]

    n_steps = max(len(t.frames) for t in tracks)
    rng = np.random.default_rng(11)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes(AX_RECT)
    ax.set_facecolor(BG)

    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=7000,
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium"])

    scale = None
    card_rng = np.random.default_rng(7)  # fixed starfield behind the cards
    with writer.saving(fig, str(out_path), dpi):
        if intro:
            n_card = int(intro.get("frames", 48))
            for k in range(n_card):
                _draw_card(fig, ax, intro, card_rng, (k + 1) / n_card)
                writer.grab_frame(facecolor=BG)
        for idx in range(n_steps):
            reps = frames_per_step if idx < n_steps - 1 else 1
            for _ in range(reps):
                scale = _draw_frame(fig, ax, tracks, idx, rng, scale, title, subtitle, notes, dpi, False)
                writer.grab_frame(facecolor=BG)
        for _ in range(hold_frames):
            scale = _draw_frame(fig, ax, tracks, n_steps - 1, rng, scale, title, subtitle, notes, dpi, True)
            writer.grab_frame(facecolor=BG)
        if outro:
            n_card = int(outro.get("frames", 72))
            for k in range(n_card):
                _draw_card(fig, ax, outro, card_rng, (k + 1) / n_card)
                writer.grab_frame(facecolor=BG)

    plt.close(fig)
    return out_path


def _fig_line(fig, x0: float, x1: float, y: float, *, color: str, lw: float, alpha: float = 1.0) -> None:
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=color, lw=lw,
                          alpha=alpha, solid_capstyle="butt"))


def _wrap_to_width(fig, s: str, max_px: float, size: float) -> List[str]:
    """Wrap `s` so no rendered line is wider than `max_px` (measured, not guessed).

    A character budget is not safe here: at 120 dpi a 54-character note already runs past
    the HUD panel, and anything past the panel edge is clipped by the frame edge.
    """
    renderer = fig.canvas.get_renderer()
    words, lines, cur = s.split(), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        probe = fig.text(0.0, 0.0, cand, size=size)
        width = probe.get_window_extent(renderer=renderer).width
        probe.remove()
        if not cur or width <= max_px:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_card(fig, ax, card: Dict[str, Any], rng: np.random.Generator, progress: float) -> None:
    """Title or result card: same chrome as the scene frames, no telemetry.

    `progress` runs 0 -> 1 across the card's frames; the accent rule wipes in and the
    copy then lands line by line, so a static freeze-frame still looks composed.
    """
    del fig.texts[:]
    del fig.patches[:]
    del fig.lines[:]
    del fig.artists[:]
    ax.clear()
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _draw_stars(ax, rng, n=300)

    color = card.get("color", CYAN)

    wipe = min(1.0, progress * 3.2)
    if wipe > 0:
        _fig_line(fig, 0.085, 0.085 + 0.105 * wipe, 0.578, color=color, lw=3.4)
    _fig_line(fig, 0.085, 0.915, 0.578, color=PANEL_EDGE, lw=1.0,
              alpha=min(1.0, max(0.0, (progress - 0.20) * 3.0)))

    kicker = min(1.0, max(0.0, (progress - 0.03) * 3.6))
    if kicker > 0 and card.get("kicker"):
        _text(fig, 0.085, 0.615, card["kicker"], size=10.5, color=color, weight="bold",
              family=MONO, alpha=kicker)

    head = min(1.0, max(0.0, (progress - 0.12) * 4.0))
    if head > 0 and card.get("headline"):
        _text(fig, 0.085, 0.500, card["headline"],
              size=40.0 if card.get("verdict") else 31.0, weight="bold",
              color=card.get("head_color", FG), alpha=head)

    for n, line in enumerate(card.get("lines") or []):
        a = min(1.0, max(0.0, (progress - 0.20 - 0.05 * n) * 5.0))
        if a > 0:
            _text(fig, 0.085, 0.408 - 0.050 * n, line, size=13.0, color=MUTED, alpha=a)

    for n, (label, value) in enumerate(card.get("stats") or []):
        a = min(1.0, max(0.0, (progress - 0.28 - 0.05 * n) * 5.2))
        if a <= 0:
            continue
        x0 = 0.085 + n * 0.205
        _text(fig, x0, 0.292, label.upper(), size=8.8, color=FAINT, family=MONO, alpha=a)
        _text(fig, x0, 0.242, value, size=19.0, color=card.get("stat_color", FG),
              family=MONO, weight="bold", alpha=a)

    if card.get("footer", True):
        _text(fig, 0.085, 0.075,
              "SpaceForge  ·  CW rendezvous testbed  ·  every figure on screen comes from recorded telemetry",
              size=8.6, color=FAINT, alpha=min(1.0, max(0.0, (progress - 0.05) * 3.0)))


def _draw_overview(fig, tracks: List[Track], idx: int, dpi: float = 110.0) -> None:
    """Corner inset showing the whole episode at a fixed scale.

    The main camera chases the action and would otherwise crop the trail, so this
    pane always shows the full path, the rings and where the craft currently is.
    """
    rect = (AX_RECT[0] + AX_RECT[2] - 0.128, AX_RECT[1] + 0.028, 0.106, 0.188)
    holder = getattr(fig, "_viz_axes", None)
    if holder is None:
        holder = fig._viz_axes = {}
    ax = holder.get("overview")
    if ax is None:
        ax = fig.add_axes(rect)
        holder["overview"] = ax
    ax.clear()
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    lim = 1.0
    for t in tracks:
        for (u, v) in t._xy:  # type: ignore[attr-defined]
            lim = max(lim, abs(u), abs(v))
    lim *= 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    px_per_m = (min(rect[2] * FIG_W_IN, rect[3] * FIG_H_IN) * dpi) / (2.0 * lim)
    if KEEP_OUT * px_per_m > 4.0:
        ax.add_patch(Circle((0.0, 0.0), KEEP_OUT, fill=False, ec=AMBER, lw=0.6, ls=(0, (3, 3)), alpha=0.5))
    if SUCCESS_DIST * px_per_m > 2.0:
        ax.add_patch(Circle((0.0, 0.0), SUCCESS_DIST, fill=False, ec=GREEN, lw=0.9, alpha=0.9))

    for t in tracks:
        arr = np.asarray(t._xy, dtype=float)  # type: ignore[attr-defined]
        if len(arr) > 1:
            ax.plot(arr[:, 0], arr[:, 1], color=t.color, lw=0.9, alpha=0.85)
        i = min(idx, len(arr) - 1)
        ax.plot([arr[i, 0]], [arr[i, 1]], marker="o", ms=3.2, color=t.color)

    ax.plot([0.0], [0.0], marker="+", ms=7, mew=1.2, color=CYAN)
    ax.text(0.05, 0.95, "OVERVIEW", transform=ax.transAxes, color=FAINT, fontsize=6.4, va="top")


def _draw_frame(fig, ax, tracks: List[Track], idx: int, rng, scale, title, subtitle, notes, dpi, terminal: bool):
    # Figure-level artists (HUD text, panels, bars) are not children of the scene
    # axes, so ax.clear() would leave them behind and they would accumulate one
    # set per frame -- which makes rendering slow down quadratically. They are all
    # ours, so drop the whole figure-level layer before redrawing.
    del fig.texts[:]
    del fig.patches[:]
    del fig.lines[:]
    del fig.artists[:]

    ax.clear()
    ax.axis("off")
    ax.set_facecolor(BG)
    _draw_stars(ax, rng)

    cur = []
    for t in tracks:
        i = min(idx, len(t.frames) - 1)
        cur.append((t, t.frames[i], t._xy[i]))  # type: ignore[attr-defined]

    # ---------------------------------------------------------------- camera
    # Frame the midpoint between the target (always at the origin) and the craft,
    # so both ends of the problem stay on screen instead of chasing the craft.
    px = float(np.mean([c["pos"][0] for _, c, _ in cur])) * 0.5
    py = float(np.mean([c["pos"][1] for _, c, _ in cur])) * 0.5
    pz = float(np.mean([c["pos"][2] for _, c, _ in cur])) * 0.5
    center = project((px, py, pz))

    fit = 1.2
    for _, _, (u, v) in cur:
        fit = max(fit, abs(u - center[0]) * 1.6, abs(v - center[1]) * 1.6)
    target_hh = float(np.clip(fit, HH_MIN, HH_MAX))

    if scale is None:
        hh = target_hh
    else:
        hh = math.exp(math.log(scale["hh"]) + (math.log(target_hh) - math.log(scale["hh"])) * CAM_EASE)
    hw = hh * AX_ASPECT
    px_per_m = (AX_H_IN * dpi) / (2.0 * hh)

    ax.set_xlim(center[0] - hw, center[0] + hw)
    ax.set_ylim(center[1] - hh, center[1] + hh)

    _draw_grid(ax, px, py, _nice_step(hh), hw, hh)

    max_range = max(f["dist"] for _, f, _ in cur)
    if KEEP_OUT * px_per_m > 34.0:
        _draw_ring(ax, KEEP_OUT, AMBER, f"keep-out {KEEP_OUT:g} m", (0, (6, 4)), 0.55, px_per_m)
    _draw_ring(ax, SUCCESS_DIST, GREEN, f"capture {SUCCESS_DIST:g} m", (0, (2, 2)), 0.8, px_per_m)

    # ---------------------------------------------------------------- trails
    for t, _, _ in cur:
        pts = np.asarray(t._xy[: min(idx, len(t._xy) - 1) + 1], dtype=float)  # type: ignore[attr-defined]
        if len(pts) > 1:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            applied = np.array([f["applied"] for f in t.frames[: len(pts)]], dtype=float)
            mags = np.linalg.norm(applied[1:], axis=1) if len(applied) > 1 else np.zeros(0)
            intensity = np.clip(mags / 0.4, 0.0, 1.0)
            colors = [(t.color, 0.14 + 0.74 * a) for a in intensity]
            ax.add_collection(LineCollection(segs, colors=colors, linewidths=2.2, capstyle="round", zorder=4))

    _draw_target(ax, project((0.0, 0.0, 0.0)), px_per_m)

    for t, c, (u, v) in cur:
        angle = math.degrees(math.atan2(v, u))
        thrust = float(np.linalg.norm(c["applied"]))
        ax.plot([0.0, u], [0.0, v], color=t.color, lw=0.8, ls=(0, (1, 3)), alpha=0.35, zorder=3)
        sx, sy = t._xy[0]  # type: ignore[attr-defined]
        ax.plot([sx], [sy], marker="o", ms=6.0, mfc="none", mec=t.color, mew=1.1, alpha=0.75, zorder=5)
        _draw_craft(ax, (u, v), angle, t.color, thrust, px_per_m)
        if t.summary.get("success") and idx >= len(t.frames) - 1:
            ax.add_patch(Circle((0.0, 0.0), SUCCESS_DIST, fill=False, ec=GREEN, lw=2.4, alpha=0.9, zorder=11))
            ax.add_patch(Circle((0.0, 0.0), SUCCESS_DIST * 2.6, fill=False, ec=GREEN, lw=1.0, alpha=0.4, zorder=11))

    # ---------------------------------------------------------------- scale bar
    bar_world = _nice_bar(0.30 * 2.0 * hw)
    bar_frac = (bar_world * px_per_m) / (AX_W_IN * dpi)
    ax.plot([0.06, 0.06 + bar_frac], [0.045, 0.045], transform=ax.transAxes,
            color=FG, lw=2.2, zorder=12, solid_capstyle="butt")
    ax.text(0.06, 0.062, _fmt_meters(bar_world), transform=ax.transAxes,
            color=FG, fontsize=9.5, family=MONO, zorder=12)

    # ---------------------------------------------------------------- overview inset
    _draw_overview(fig, tracks, idx, dpi)

    # ---------------------------------------------------------------- HUD
    n_tracks = len(tracks)
    _text(fig, HUD_X, 0.950, title, size=15.5, weight="bold")
    _text(fig, HUD_X, 0.916, subtitle, size=9.8, color=MUTED)

    row_gap = 0.0286 if n_tracks == 1 else (0.0265 if n_tracks == 2 else 0.0245)
    y = 0.868 if n_tracks == 1 else 0.858
    for t, c, _ in cur:
        if n_tracks > 1:
            _text(fig, HUD_X, y, t.name, size=10.5, color=t.color, weight="bold")
            y -= 0.030

        over = c["override"]
        dists = [f["dist"] for f in t.frames]
        i = min(idx, len(dists) - 1)
        lookback = min(20, i)
        trending_out = lookback > 4 and dists[i] > dists[i - lookback] + 0.15

        if t.summary.get("success") and idx >= len(t.frames) - 1:
            status, scolor = "DOCKED — SOFT CAPTURE", GREEN
        elif over == "emergency_hold":
            status, scolor = "INTERLOCK — EMERGENCY HOLD", RED
        elif over == "rejected":
            status, scolor = "INTERLOCK — THRUST REJECTED", RED
        elif trending_out and c["dist"] > 2.0:
            status, scolor = "DIVERGING — NO CAPTURE", RED
        elif c["dist"] < KEEP_OUT:
            status, scolor = "INSIDE KEEP-OUT — GLIDESLOPE", AMBER
        else:
            status, scolor = "APPROACH — CLOSING", CYAN

        closing = -float(np.dot(c["pos"], c["vel"])) / max(c["dist"], 1e-9)
        rows_all = [
            ("RANGE", _fmt_meters(c["dist"]), FG),
            ("CLOSING RATE", f"{closing:+.3f} m/s", GREEN if closing > 0 else AMBER),
            ("REL SPEED", f"{c['speed']:.3f} m/s", FG),
            ("APPLIED THRUST", f"{float(np.linalg.norm(c['applied'])):.3f} m/s2", AMBER),
            ("MET", f"{int(c['t']) // 60:02d}:{int(c['t']) % 60:02d}", FG),
            ("FUEL", f"{c['fuel'] * 100:.2f} %", GREEN if c["fuel"] > 0.05 else RED),
            ("TOTAL dv", f"{c['dv']:.3f} m/s", FG),
            ("INTERLOCK FIRES", f"{t.summary['overrides']}", RED if t.summary["overrides"] else MUTED),
        ]
        if n_tracks == 1:
            rows = rows_all
        elif n_tracks == 2:
            rows = [rows_all[k] for k in (0, 1, 2, 3, 7)]
        else:
            rows = [rows_all[k] for k in (0, 2, 7)]

        _text(fig, HUD_X, y, status, size=11.0 if n_tracks == 1 else 10.0, color=scolor, weight="bold")
        y -= 0.030
        for label, value, color in rows:
            _text(fig, HUD_X, y, label, size=9.2, color=MUTED)
            _text(fig, HUD_X + HUD_W, y, value, size=10.6, color=color, family=MONO, ha="right")
            y -= row_gap
        y -= 0.020

    # ---------------------------------------------------------------- facts card
    if n_tracks == 1:
        card_top, card_h = 0.548, 0.158
        _round_rect(fig, HUD_X, card_top - card_h, HUD_W, card_h)
        _text(fig, HUD_X + 0.014, card_top - 0.028, "SCENARIO", size=8.6, color=FAINT, weight="bold")
        lines = [
            ("Dynamics", "CW, exact STM"),
            ("Step / horizon", "1.0 s  ·  300 steps"),
            ("Actuator", "u ∈ [-1, 1] m/s²"),
            ("Guidance", tracks[0].name[:34]),
            ("Assurance", "mission_ctrl interlock"),
        ]
        yy = card_top - 0.052
        for k, v in lines:
            _text(fig, HUD_X + 0.014, yy, k, size=8.4, color=MUTED)
            _text(fig, HUD_X + HUD_W - 0.014, yy, v, size=8.4, color=FG, family=MONO, ha="right")
            yy -= 0.0235

    # ---------------------------------------------------------------- range strip
    _range_strip(fig, HUD_X, 0.245, HUD_W, 0.130, tracks, idx)
    _fuel_bar(fig, HUD_X, 0.185, HUD_W, cur)

    if notes and n_tracks <= 2:
        yy = 0.115
        max_px = HUD_W * FIG_W_IN * dpi * 0.97   # stay inside the panel, not just the frame
        for note in notes:
            for line in _wrap_to_width(fig, note, max_px, 8.8):
                _text(fig, HUD_X, yy, line, size=8.8, color=AMBER if terminal else MUTED)
                yy -= 0.021

    out = {"hh": hh}
    return out


def _fuel_bar(fig, x: float, y: float, w: float, cur) -> None:
    _text(fig, x, y + 0.022, "FUEL REMAINING", size=8.6, color=FAINT, weight="bold")
    for k, (t, c, _) in enumerate(cur):
        by = y - k * 0.026
        _round_rect(fig, x, by, w, 0.013, fc=PANEL, ec=PANEL_EDGE)
        frac = max(0.0, min(1.0, float(c["fuel"])))
        fig.patches.append(Rectangle((x, by), max(w * frac, 0.002), 0.013, transform=fig.transFigure,
                                     fc=GREEN if frac > 0.05 else RED, ec="none", zorder=4))
        fig.text(x + w - 0.006, by + 0.0065, f"{frac * 100:.2f}%", size=8.2, color=FG,
                 family=MONO, ha="right", va="center")


def _range_strip(fig, x: float, y: float, w: float, h: float, tracks: List[Track], idx: int) -> None:
    """Log-scale range-versus-time strip with decade gridlines and a live playhead."""
    holder = getattr(fig, "_viz_axes", None)
    if holder is None:
        holder = fig._viz_axes = {}
    ax = holder.get("strip")
    if ax is None:
        ax = fig.add_axes((x, y, w, h))
        holder["strip"] = ax
    ax.clear()
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.9)
    ax.set_yticks([])
    ax.set_xticks([])

    n_max = max(len(t.frames) for t in tracks)
    r_max = max(max(f["dist"] for f in t.frames) for t in tracks)
    y_lo = math.log10(0.05)
    y_hi = max(1.2, math.log10(max(1.0, r_max)) + 0.05)

    for decade in range(0, 6):
        val = 10 ** decade
        lv = math.log10(val)
        if y_lo < lv < y_hi:
            ax.axhline(lv, color=GRID_STRONG, lw=0.7, alpha=0.9, zorder=1)
            ax.text(0.985, lv, _fmt_meters(val).replace(" ", ""), transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", color=FAINT, fontsize=6.6, family=MONO)

    for t in tracks:
        dists = np.maximum(np.array([f["dist"] for f in t.frames]), 0.05)
        steps = np.arange(len(dists))
        logs = np.log10(dists)
        # future of the episode stays faint; the part already played is solid,
        # so the playhead's progress is unmistakable
        ax.plot(steps, logs, color=t.color, lw=1.3, alpha=0.28, zorder=2)
        played = steps <= idx
        ax.plot(steps[played], logs[played], color=t.color, lw=1.9, alpha=0.98, zorder=3)
        ax.fill_between(steps[played], y_lo, logs[played], color=t.color, alpha=0.13, zorder=2)

    ax.axhline(math.log10(SUCCESS_DIST), color=GREEN, lw=1.0, ls=(0, (3, 3)), alpha=0.9, zorder=4)
    ax.axvline(idx, color=FG, lw=1.2, alpha=0.85, zorder=5)
    ax.set_xlim(0, max(2, n_max - 1))
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.03, 0.90, "RANGE  (log)", transform=ax.transAxes, color=MUTED, fontsize=7.6, va="top", zorder=6)
