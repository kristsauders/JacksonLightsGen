"""Why is this prop doing that, right now?

    python custom_tools/explain_moment.py "Gooey (v153)" 60.0 "Fence Matrix 1"
    python custom_tools/explain_moment.py "Gooey (v153)" 60.0            # whole house

Lists every effect covering a moment on a prop -- including the ones reaching
it through a GROUP, which is most of them and the usual reason a prop does
something no effect on its own name explains -- and names the pass that placed
each one, from the `X_Src` tag generate_song.py writes on every placement.

This exists because the passes interact in ways reading them does not predict.
Three ordering bugs turned up in a single day (a truncation pass running before
the passes that add effects; the pulse-rate ceiling running before the pass
that shortens spans, so shortened curves were never re-checked; a cohesion
sweep counting coverage from before its own fills), and a strobe regression
survived six straight hypotheses that each measured completely inert. The
question "which pass put this here" has to be answerable directly.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SHOW = Path(json.loads((Path(__file__).resolve().parents[1] / "config.json")
                       .read_text())["show_folders"]["christmas2026"])
GEN = SHOW / ".generation"


def pass_label(line: int, src: list[str]) -> str:
    if line <= 0 or not src:
        return "?"
    for i in range(min(line, len(src)) - 1, max(0, line - 400), -1):
        t = src[i].strip()
        if not t.startswith("#"):
            continue
        t = t.lstrip("# ").rstrip()
        if not t or t.startswith("---") or t.startswith("==="):
            continue
        letters = [c for c in t if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7 \
                and len(t) < 80:
            return t.rstrip(".")
    return f"line {line}"


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    name, when = sys.argv[1], float(sys.argv[2])
    prop = sys.argv[3] if len(sys.argv) > 3 else None

    src = (Path(__file__).parent / "generate_song.py").read_text().split("\n")
    groups = json.loads((GEN / "layout.json").read_text())["xlights_model_groups"]
    owners = None
    if prop:
        owners = {n for n, mem in groups.items() if prop in mem} | {prop}

    root = ET.parse(SHOW / f"{name}.xsq").getroot()
    db = [e.text or "" for e in root.find("EffectDB")]
    pals = [e.text or "" for e in root.find("ColorPalettes")]

    rows = []
    for el in root.find("ElementEffects"):
        nm = el.get("name", "")
        if owners is not None and nm not in owners and not nm.startswith(prop + "/"):
            continue
        for li, layer in enumerate(el.findall("EffectLayer")):
            for ef in layer.findall("Effect"):
                s = int(ef.get("startTime")) / 1000.0
                e = int(ef.get("endTime")) / 1000.0
                if not (s <= when < e):
                    continue
                st = db[int(ef.get("ref"))] if ef.get("ref") else ""
                m = re.search(r"X_Src=(\d+)", st)
                line = int(m.group(1)) if m else 0
                shader = re.search(r"Shaders/([^.]+)\.fs", st)
                pal = pals[int(ef.get("palette"))] if ef.get("palette") else ""
                cols = re.findall(r"#[0-9A-Fa-f]{6}", pal)[:2]
                vc = "curve" if "C_VALUECURVE_Brightness" in pal else ""
                rows.append((nm, li, s, e, ef.get("name") or "?",
                             shader.group(1) if shader else "", line, cols, vc))

    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"{name} @ {when:.2f}s"
          + (f" -- everything reaching {prop!r}" if prop else " -- whole house")
          + f"   ({len(rows)} effects)\n")
    for nm, li, s, e, kind, shader, line, cols, vc in rows:
        print(f"  {nm[:26]:26s} L{li} {s:7.2f}-{e:7.2f} {kind:13s} "
              f"{shader[:18]:18s} {','.join(cols):16s} {vc:5s}")
        print(f"       from line {line}: {pass_label(line, src)}")


if __name__ == "__main__":
    main()
