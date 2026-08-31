"""Do the drops land? Measure each one, and what competes with it.

    python custom_tools/verify_drops.py "Gooey (v170)" ["Gooey (v134)" ...]

A drop reads as a moment when three things happen: the bar before it gets out
of the way, the level jumps, and nothing else changes at the same time. Each
is measured here, because "the drops have lost their impact" is otherwise a
feeling with no handle on it.

  dip     how far the 2s before the drop falls below the 8s before that
          (0.70 = the run-up is 30% darker than the passage -- a real hush)
  rise    peak in the second after the drop, over the 2s before it
  busy    effect starts near the drop that are neither ON it nor on a beat --
          genuinely competing moments, the "other changes happening right
          before or after that distract from the main moment". Starts on later
          beats are the rhythm resuming and are not counted.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from fseq_reader import read_fseq  # noqa: E402

SHOW = Path(json.loads((Path(__file__).resolve().parents[1] / "config.json")
                       .read_text())["show_folders"]["christmas2026"])


def drop_times(song: str) -> list[tuple[float, str]]:
    """Cue times WITH their kind. An instrument dropout is supposed to take
    light away, so judging it by how much it rises is meaningless -- lumping
    the three kinds together made three correct moments look like failures."""
    gen = SHOW / ".generation" / song
    cues = json.loads((gen / "cues.json").read_text())
    out = {}
    for k, short in (("energy_drops", "energy"),
                     ("predrop_cutouts", "cutout"),
                     ("instrument_dropouts", "dropout")):
        for ev in cues.get(k) or []:
            t = ev.get("drop_time") or ev.get("time")
            if t:
                out.setdefault(round(float(t), 2), short)
    return sorted(out.items())


def main() -> None:
    for name in sys.argv[1:]:
        song = name.rsplit(" (", 1)[0]
        f = read_fseq(SHOW / f"{name}.fseq")

        def lvl(a: float, b: float) -> float:
            idx = [min(int(t / 0.025), f.frame_count - 1)
                   for t in np.arange(a, b, 0.05)]
            return float(np.mean([np.frombuffer(f.frame(i), dtype=np.uint8).mean()
                                  for i in idx])) if idx else 0.0

        def peak(a: float, b: float) -> float:
            idx = [min(int(t / 0.025), f.frame_count - 1)
                   for t in np.arange(a, b, 0.05)]
            return float(max([np.frombuffer(f.frame(i), dtype=np.uint8).mean()
                              for i in idx])) if idx else 0.0

        beats = np.array(json.loads(
            (SHOW / ".generation" / song / "analysis.json").read_text()
        )["beats"]["beat_times"])
        root = ET.parse(SHOW / f"{name}.xsq").getroot()
        starts: list[float] = []
        for el in root.find("ElementEffects"):
            for layer in el.findall("EffectLayer"):
                for ef in layer.findall("Effect"):
                    starts.append(int(ef.get("startTime")) / 1000.0)
        starts.sort()
        sa = np.array(starts)

        print(f"\n=== {name} ===")
        print(f"{'drop':>8} {'kind':8s} {'dip':>5} {'rise':>5} {'busy':>5}   verdict")
        rows = []
        for d, kind in drop_times(song):
            if d < 6:
                continue
            passage = lvl(max(0, d - 10), max(0.2, d - 2))
            runup = lvl(max(0, d - 2), d)
            after = peak(d, d + 1.0)
            dip = runup / max(passage, 1e-6)
            rise = after / max(runup, 1e-6)
            # A start within 120ms of the drop IS the drop -- the whole house
            # arriving together is the point. What distracts is a SEPARATE
            # moment just before or after it, so count only starts that are
            # far enough away to read as their own event.
            # A start on a LATER BEAT is the rhythm resuming, not a competing
            # moment. The first version of this counted those and reported 600
            # "competing starts" at a drop whose surroundings were in fact
            # clean -- the clusters sat at +0.70s and +1.40s, which at 86 BPM
            # are simply the next two beats. What distracts is a change that
            # is neither on the drop nor on the grid.
            near = sa[(sa > d - 1.5) & (sa < d + 1.5)]
            off = np.abs(near - d)
            candidates = near[(off > 0.30) & (off < 1.5)]
            busy = int(sum(1 for t in candidates
                           if beats.size == 0
                           or float(np.min(np.abs(beats - t))) > 0.08))
            bad = []
            if kind != "dropout":
                if dip > 0.92:
                    bad.append("no hush")
                if rise < 1.25:
                    bad.append("no lift")
            elif rise > 1.15:
                bad.append("dropout that gets BRIGHTER")
            if busy > 60:
                bad.append(f"{busy} competing starts")
            rows.append((dip, rise, busy))
            print(f"{d:8.2f} {kind:8s} {dip:5.2f} {rise:5.2f} {busy:5d}   "
                  + ("OK" if not bad else ", ".join(bad)))
        if rows:
            a = np.array(rows)
            print(f"{'mean':>8} {'':8s} {a[:,0].mean():5.2f} {a[:,1].mean():5.2f} "
                  f"{a[:,2].mean():5.0f}")


if __name__ == "__main__":
    main()
