"""Is the show VISIBLY on the beat? Measured from the render, not the file.

    python custom_tools/beat_pulse.py "<song>" v1 v2 ...

"I feel like we've lost some of the overall rhythm... it needs to catch your
eye and be clearly on the beat."

Rhythm you can see is a BRIGHTNESS JUMP that lands on the beat and is bigger
than what happens between beats. So:

  salience = (mean biggest frame-to-frame jump within +/-60ms of a beat)
             / (mean biggest jump in the same-width windows OFF the beat)

1.0 means the house is equally busy on and off the beat -- no visible pulse,
however many effects are "beat-locked" in the file. Also reported per prop
family, because rhythm can move from the props to the canvases and vanish:
a big canvas changing texture is not a hit.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/custom_tools")
from fseq_reader import read_fseq  # noqa: E402

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
STEP = 0.025          # 40fps, the render rate
NEAR = 0.060          # a jump this close to the beat reads as ON it


def fam(m: str) -> str:
    l = m.lower()
    if "arch" in l or "tunnel" in l:
        return "Arches/Tunnel"
    if "matri" in l or l == "garage roof":
        return "Matrices"
    if m.startswith("Single Line"):
        return "Roof/Gutters"
    if m.startswith("Vertical Strip"):
        return "Verticals"
    if any(h in m for h in ("Flake", "Snowflake", "Spinner", "Rosa", "Illusion",
                            "Wreath", "Star")):
        return "Flakes/Spinners"
    if "Cane" in m or "Mini Tree" in m:
        return "Canes/Trees"
    if m == "Mega Tree":
        return "Mega Tree"
    return "other"


def main(song: str, versions: list[str]) -> None:
    a = json.loads((SHOW / ".generation" / song / "analysis.json").read_text())
    beats = np.asarray(a["beats"]["beat_times"], dtype=float)
    cm = json.loads((SHOW / ".generation" / "channel_map.json").read_text())["models"]
    fams: dict[str, list] = {}
    for m, v in cm.items():
        fams.setdefault(fam(m), []).append((max(0, v["first"] - 1), v["last"]))

    print(f"\n=== {song}   beat salience (1.00 = no visible pulse)")
    for ver in versions:
        f = SHOW / f"{song} (v{ver}).fseq"
        if not f.exists():
            print(f"  v{ver}: no render")
            continue
        fs = read_fseq(f)
        ts = np.arange(1.0, fs.duration_s - 0.5, STEP)
        frames = [np.frombuffer(fs.frame_at(t), dtype=np.uint8) for t in ts]

        def salience(sl):
            if sl is None:
                series = np.array([fr.mean() for fr in frames])
            else:
                series = np.array([np.mean([fr[x:y].mean() for x, y in sl])
                                   for fr in frames])
            d = np.abs(np.diff(series))
            dt = ts[1:]
            onb = np.zeros(len(dt), dtype=bool)
            for b in beats:
                onb |= (np.abs(dt - b) <= NEAR)
            if not onb.any() or not (~onb).any():
                return 0.0, 0.0
            # biggest jump in each beat window vs each off-beat window
            on = []
            for b in beats:
                w = d[np.abs(dt - b) <= NEAR]
                if w.size:
                    on.append(w.max())
            off = d[~onb]
            return (float(np.mean(on)) / max(float(np.mean(off)), 1e-9),
                    float(np.mean(on)))

        s, mag = salience(None)
        parts = []
        for name in ("Mega Tree", "Arches/Tunnel", "Matrices", "Roof/Gutters",
                     "Flakes/Spinners", "Canes/Trees", "Verticals"):
            if name in fams:
                fs2, _ = salience(fams[name])
                parts.append(f"{name.split('/')[0]} {fs2:.2f}")
        print(f"  v{ver:<4s} WHOLE HOUSE {s:5.2f}x   on-beat jump {mag:5.2f} "
              f"| " + "  ".join(parts))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
