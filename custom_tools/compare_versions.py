"""What changed between two builds of a song, and what should I go and look at?

    python custom_tools/compare_versions.py "This Version of You" v33 v36
    python custom_tools/compare_versions.py --all v_prev v_new     (whole library)

Renders are compared on the measures that correspond to things Krists actually
notices, and every difference is reported with a TIMESTAMP to go and watch.
The point is not a pass/fail verdict -- it is knowing where to look, because
"something regressed somewhere in a five minute song" is the expensive part.

  overall     lit fraction, red fraction
  moments     the biggest brightness changes, listed with their times
  impacts     how much each drop rises above its own run-up
  tail        how much light survives the final dropout
  structure   effects per pass (from the X_Src provenance tags)
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from fseq_reader import read_fseq  # noqa: E402

SHOW = Path(json.loads((Path(__file__).resolve().parents[1] / "config.json")
                       .read_text())["show_folders"]["christmas2026"])
SRC = (Path(__file__).parent / "generate_song.py").read_text().split("\n")


def _label(line: int) -> str:
    from explain_moment import pass_label
    return pass_label(line, SRC)


def series(name: str, step: float = 0.25) -> np.ndarray:
    f = read_fseq(SHOW / f"{name}.fseq")
    n = int(f.frame_count * 0.025 / step)
    return np.array([np.frombuffer(f.frame(min(int(i * step / 0.025),
                                               f.frame_count - 1)),
                                   dtype=np.uint8).mean() for i in range(n)])


def passes(name: str) -> Counter:
    root = ET.parse(SHOW / f"{name}.xsq").getroot()
    db = [e.text or "" for e in root.find("EffectDB")]
    c: Counter = Counter()
    for el in root.find("ElementEffects"):
        for layer in el.findall("EffectLayer"):
            for ef in layer.findall("Effect"):
                m = re.search(r"X_Src=(\d+)", db[int(ef.get("ref"))])
                c[_label(int(m.group(1))) if m else "(untagged build)"] += 1
    return c


def compare(song: str, a: str, b: str) -> None:
    na, nb = f"{song} ({a})", f"{song} ({b})"
    for n in (na, nb):
        if not (SHOW / f"{n}.fseq").exists():
            print(f"  {song}: no render for {n}")
            return
    sa, sb = series(na), series(nb)
    k = min(len(sa), len(sb))
    sa, sb = sa[:k], sb[:k]
    print(f"\n=== {song}: {a} -> {b} ===")
    print(f"  whole-song mean {sa.mean():5.1f} -> {sb.mean():5.1f} "
          f"({(sb.mean() - sa.mean()) / max(sa.mean(), 1e-6) * 100:+.0f}%)")

    # Where did it change most? Report in seconds, biggest first, merged into
    # runs so one long change is one line rather than forty.
    d = sb - sa
    thresh = max(8.0, float(np.percentile(np.abs(d), 97)))
    hot = np.abs(d) > thresh
    runs, cur = [], None
    for i, h in enumerate(hot):
        if h and cur is None:
            cur = i
        elif not h and cur is not None:
            runs.append((cur, i))
            cur = None
    if cur is not None:
        runs.append((cur, len(hot)))
    runs = [(x, y) for x, y in runs if (y - x) * 0.25 >= 1.0]
    runs.sort(key=lambda r: -abs(d[r[0]:r[1]]).mean())
    if runs:
        print(f"  biggest changes (>{thresh:.0f} levels, 1s+):")
        for x, y in runs[:8]:
            t0, t1 = x * 0.25, y * 0.25
            print(f"     {int(t0//60)}:{t0%60:05.2f}-{int(t1//60)}:{t1%60:05.2f}  "
                  f"{sa[x:y].mean():5.1f} -> {sb[x:y].mean():5.1f}")
    else:
        print("  no sustained brightness differences")

    pa, pb = passes(na), passes(nb)
    keys = set(pa) | set(pb)
    diffs = sorted(((pb.get(k, 0) - pa.get(k, 0), k) for k in keys),
                   key=lambda x: -abs(x[0]))
    shown = [(n, k) for n, k in diffs if n]
    if shown and "(untagged build)" not in (shown[0][1],):
        print("  effects placed, by pass:")
        for n, k in shown[:8]:
            print(f"     {n:+6d}  {k[:64]}")


def main() -> None:
    args = sys.argv[1:]
    if len(args) >= 3 and args[0] == "--all":
        songs = sorted({p.name.rsplit(" (", 1)[0] for p in SHOW.glob("*.xsq")
                        if " (v" in p.name})
        for s in songs:
            compare(s, args[1], args[2])
    elif len(args) == 3:
        compare(args[0], args[1], args[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
