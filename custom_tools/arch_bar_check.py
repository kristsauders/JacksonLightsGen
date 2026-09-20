"""Is it ONE BAR across all three rows, or three rows chasing in turn?

    python custom_tools/arch_bar_check.py "<song>" <version>

Each arch is `LayerSizes=100,100,100` -- three concentric rows of 100 nodes,
stored one row after another. That is the fact every earlier attempt missed:
`Single Line` runs the length of row 1, then row 2, then row 3, which Krists
saw as "each row chasing very very very fast, one after another".

So this splits each arch's 300 nodes into its three rows and reports, for each
row, WHERE the lit nodes are (0.0-1.0 along the arch). A bar spanning all three
rows shows the same position in all three; three rows chasing in turn shows one
row lit at a time.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/custom_tools")
from fseq_reader import read_fseq  # noqa: E402

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
ROWS = 3


def main(song: str, ver: str) -> None:
    cm = json.loads((SHOW / ".generation" / "channel_map.json").read_text())["models"]
    xs = (SHOW / f"{song} (v{ver}).xsq").read_text()
    db = re.findall(r"<Effect>(.*?)</Effect>",
                    xs.split("<EffectDB>")[1].split("</EffectDB>")[0], re.S)
    cur, leaps = None, []
    for m in re.finditer(r'<Element [^>]*name="([^"]+)"|<Effect ([^>]*)/>', xs):
        if m.group(1):
            cur = m.group(1); continue
        at = m.group(2)
        ref = int(re.search(r'ref="(\d+)"', at).group(1))
        if ref < len(db) and "X_ArchLeap" in db[ref] and cur and cur.startswith("Arch "):
            leaps.append((int(re.search(r'startTime="(\d+)"', at).group(1)) / 1000.0,
                          int(re.search(r'endTime="(\d+)"', at).group(1)) / 1000.0, cur))
    leaps.sort()
    print(f"\n=== {song} (v{ver}): {len(leaps)} per-arch leap sweeps")
    if not leaps:
        return
    f = read_fseq(SHOW / f"{song} (v{ver}).fseq")
    for a0, b0, name in leaps[:4]:
        print(f"\n--- {name}  {a0:.2f}-{b0:.2f}s   (position of lit nodes per row)")
        print(f"  {'time':>7s}  {'row1':>12s} {'row2':>12s} {'row3':>12s}   verdict")
        for t in np.arange(a0, b0, 0.06):
            sl = np.frombuffer(f.frame_at(t), dtype=np.uint8)[
                max(0, cm[name]["first"] - 1):cm[name]["last"]]
            px = sl[:len(sl) // 3 * 3].reshape(-1, 3).max(axis=1)
            per = len(px) // ROWS
            cells, poss = [], []
            for r in range(ROWS):
                seg = px[r * per:(r + 1) * per]
                lit = seg > 25
                if lit.sum():
                    pos = float(np.flatnonzero(lit).mean()) / max(per - 1, 1)
                    poss.append(pos)
                    cells.append(f"{100*lit.mean():3.0f}% {pos:5.2f}")
                else:
                    cells.append(f"{'.':>12s}")
            # A BAR IS NARROW. The first version of this called 100%-lit
            # "BAR across all 3 rows" because the three rows agreed on a
            # position of 0.50 -- which is what a WHOLE-ARCH FLASH looks like.
            # Agreement between rows is necessary and nowhere near sufficient.
            _wide = max((100 * (px[r*per:(r+1)*per] > 25).mean())
                        for r in range(ROWS))
            if len(poss) == ROWS and (max(poss) - min(poss)) < 0.15 and _wide < 40:
                verdict = "BAR across all 3 rows"
            elif len(poss) == ROWS and _wide >= 80:
                verdict = "WHOLE ARCH LIT (flash, not a bar)"
            elif len(poss) == 1:
                verdict = "only one row lit"
            elif not poss:
                verdict = ""
            else:
                verdict = "rows out of step"
            print(f"  {t:7.2f}  " + " ".join(cells) + f"   {verdict}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
