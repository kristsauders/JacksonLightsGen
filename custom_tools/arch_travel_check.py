"""Does a light TRAVEL along each arch, or does the arch just light up whole?

    python custom_tools/arch_travel_check.py "<song>" <version>

The step device (one whole arch per beat) and the leap device (a dot arcing
along the arch's own 300 nodes) look identical to any measurement that reads a
model's MEAN brightness -- both make "Arch 2" bright. The difference is WHERE
inside the arch the light is, so this reports the centroid of the lit nodes
along each arch, frame by frame. A leap sweeps the centroid 0 -> 1 across the
arch; a flash pins it at the middle with everything lit.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/custom_tools")
from fseq_reader import read_fseq  # noqa: E402

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")


def main(song: str, ver: str) -> None:
    cm = json.loads((SHOW / ".generation" / "channel_map.json").read_text())["models"]
    arches = [f"Arch {i}" for i in (1, 2, 3, 4)]
    xs = (SHOW / f"{song} (v{ver}).xsq").read_text()
    db = re.findall(r"<Effect>(.*?)</Effect>",
                    xs.split("<EffectDB>")[1].split("</EffectDB>")[0], re.S)
    leaps = []
    for m in re.finditer(r'<Effect ([^>]*)/>', xs):
        at = m.group(1)
        ref = int(re.search(r'ref="(\d+)"', at).group(1))
        if ref < len(db) and "X_ArchLeap" in db[ref]:
            leaps.append((int(re.search(r'startTime="(\d+)"', at).group(1)) / 1000.0,
                          int(re.search(r'endTime="(\d+)"', at).group(1)) / 1000.0))
    leaps.sort()
    print(f"\n=== {song} (v{ver}): {len(leaps)} leap bars")
    if not leaps:
        return
    f = read_fseq(SHOW / f"{song} (v{ver}).fseq")
    for a0, b0 in leaps[:3]:
        print(f"\n--- leap bar {a0:.2f}-{b0:.2f}s")
        print(f"  {'time':>7s} " + "  ".join(f"{'A'+n[-1]:>13s}" for n in arches))
        print(f"  {'':>7s} " + "  ".join(f"{'lit% pos':>13s}" for _ in arches))
        for t in np.arange(a0, b0, 0.08):
            cells = []
            for n in arches:
                sl = np.frombuffer(f.frame_at(t), dtype=np.uint8)[
                    max(0, cm[n]["first"] - 1):cm[n]["last"]]
                px = sl[:len(sl) // 3 * 3].reshape(-1, 3).max(axis=1)
                lit = px > 25
                if lit.sum():
                    pos = float(np.flatnonzero(lit).mean()) / max(len(px) - 1, 1)
                    cells.append(f"{100*lit.mean():4.0f}% {pos:6.2f}")
                else:
                    cells.append(f"{'.':>13s}")
            print(f"  {t:7.2f} " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
