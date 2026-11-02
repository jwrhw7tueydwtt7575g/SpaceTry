"""Bundle the browser docking simulator into one self-contained HTML file.

    python -m spaceforge_mvp.viz.dashboard

Reads the sources in ``spaceforge_mvp/viz/lab/`` (``index.html``,
``styles.css``, ``policy.js``, ``sim.js``, ``app.js``) and writes
``spaceforge_mvp/viz/dashboard.html`` with every stylesheet and script inlined:
no server, no build tooling, no CDN required. Open it from disk and it runs.
The only external reference is an optional web-font stylesheet; offline, the
page falls back to system fonts.

During development, serve the repository root and open
``spaceforge_mvp/viz/lab/index.html`` directly instead.

The simulation inside is a JavaScript port of the Python environment.
``lab/sim.test.js`` checks it against fixtures from
``python -m spaceforge_mvp.viz.lab_export``; rerun that first if the Python
physics, safety filter or checkpoint changed.
"""

import re
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent / "lab"
OUT = Path(__file__).resolve().parent / "dashboard.html"

STYLESHEETS = ("styles.css",)
SCRIPTS = ("policy.js", "sim.js", "app.js")


def _inline_script(source: str) -> str:
    # A literal "</script" inside a script body would end the element early.
    return source.replace("</script", "<\\/script")


def render(lab_dir: Path = LAB_DIR) -> str:
    html = (lab_dir / "index.html").read_text(encoding="utf-8")
    for name in STYLESHEETS:
        tag = f'<link rel="stylesheet" href="{name}">'
        if tag not in html:
            raise ValueError(f"index.html no longer references {name}")
        html = html.replace(tag, "<style>\n" + (lab_dir / name).read_text(encoding="utf-8") + "</style>")
    for name in SCRIPTS:
        tag = f'<script src="{name}"></script>'
        if tag not in html:
            raise ValueError(f"index.html no longer references {name}")
        body = _inline_script((lab_dir / name).read_text(encoding="utf-8"))
        html = html.replace(tag, "<script>\n" + body + "</script>")
    leftover = re.findall(r'<script[^>]+src="(?!https?:)[^"]+"', html)
    if leftover:
        raise ValueError(f"unbundled local scripts: {leftover}")
    return html


def build(out: Path = None) -> Path:
    target = Path(out) if out is not None else OUT
    target.write_text(render(), encoding="utf-8")
    return target


if __name__ == "__main__":
    path = build()
    print(f"dashboard written: {path}  ({path.stat().st_size / 1024:.0f} KB)")
