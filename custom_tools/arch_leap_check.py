"""Do the arches actually LEAP? Measured from the render, per arch, per beat.

    python custom_tools/arch_leap_check.py "<song>" [version]

Settings can be perfect and the gesture still invisible, so this asks the only
question that matters: on each beat of a leap bar, WHICH arch is the brightest?
A leap looks like 1,2,3,4 then 4,3,2,1. A smear or a flat flash looks like the
same arch every beat, or no clear winner at all.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/custom_tools")
from fseq_reader import read_fseq  # noqa: E402

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")


def main(song: str, ver: str | None = None) -> None:
    vs = sorted(int(m.group(1)) for p in SHOW.glob(f"{song} (v*).fseq")
                if (m := re.search(r"\(v(\d+)\)\.fseq$", p.name)))
    v = int(ver) if ver else vs[-1]
    cm = json.loads((SHOW / ".generation" / "channel_map.json").read_text())["models"]
    arches = [f"Arch {i}" for i in (1, 2, 3, 4) if f"Arch {i}" in cm]
    a = json.loads((SHOW / ".generation" / song / "analysis.json").read_text())
    beats = np.asarray(a["beats"]["beat_times"], dtype=float)

    # where the leaps are
    xs = (SHOW / f"{song} (v{v}).xsq").read_text()
    db = re.findall(r"<Effect>(.*?)</Effect>",
                    xs.split("<EffectDB>")[1].split("</EffectDB>")[0], re.S)
    leaps = []
    for m in re.finditer(r'<Effect ([^>]*)/>', xs):
        at = m.group(1)
        ref = int(re.search(r'ref="(\d+)"', at).group(1))
        if ref < len(db) and "X_ArchFour" in db[ref]:
            leaps.append((int(re.search(r'startTime="(\d+)"', at).group(1)) / 1000.0,
                          int(re.search(r'endTime="(\d+)"', at).group(1)) / 1000.0))
    leaps.sort()
    f = read_fseq(SHOW / f"{song} (v{v}).fseq")
    print(f"\n=== {song} (v{v})  {len(leaps)} leap bars, arches {arches}")

    def bright(t):
        fr = np.frombuffer(f.frame_at(t), dtype=np.uint8)
        return [fr[max(0, cm[n]["first"] - 1):cm[n]["last"]].mean() for n in arches]

    seqs, clean = [], 0
    for (a0, b0) in leaps[:14]:
        bs = [x for x in beats if a0 - 0.02 <= x < b0]
        if len(bs) < 3:
            continue
        row = []
        for bt in bs[:4]:
            vals = bright(bt + 0.10)          # just after the beat lands
            row.append(int(np.argmax(vals)) + 1 if max(vals) > 8 else 0)
        seqs.append((a0, row, [round(x) for x in bright(bs[0] + 0.10)]))
        if len(set(row)) >= 3:                # the winner actually moves
            clean += 1
    for a0, row, lev in seqs:
        print(f"  {a0:7.2f}s  brightest arch per beat: {row}   "
              f"levels on beat 1: {lev}")
    if seqs:
        print(f"\n  bars where the brightest arch MOVES across >=3 arches: "
              f"{clean}/{len(seqs)}   (a leap should be nearly all of them)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
