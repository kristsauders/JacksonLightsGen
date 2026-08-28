"""Find each song's ambient dim factor by rendering and measuring.

    python custom_tools/calibrate_ambient_dim.py "Cruise" "Meridian" ...

Quiet passages have to sit BELOW loud ones or the drops cannot stand above
them -- "the drops and peak climax moments don't seem as big because they
can't be brighter than the rest." Krists set the target at 15-30% darker,
explicitly not half.

A single multiplier cannot deliver that across songs: what fraction of a
song's light comes from the ambient pass, as opposed to every other pass
running underneath, varies. Measured with a uniform 0.50 the library landed
between -18% and -46%, straddling the band in both directions.

So measure it. Render, compare mean level inside ambient passages against
everything else, and solve for the factor that would have hit the middle of
the band. The relationship is close enough to linear over this range that one
correction step lands within a couple of points; the result is cached per song
in .generation/<song>/ambient_dim.json and picked up on the next build.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ambient_sections import analyse  # noqa: E402
from fseq_reader import read_fseq  # noqa: E402

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
XLIGHTS = Path("/Applications/xLights.app/Contents/MacOS/xLights")
TARGET = -0.22          # middle of Krists's 15-30% band
# The factor may exceed 1.0 -- i.e. BRIGHTEN. Half the library turned out to
# have quiet passages naturally far below the target (Ojai measured -57.8%
# with no dimming at all, Aperture -36.4% at 0.99), and a mechanism that can
# only darken has no way to bring those up. Krists's band is a target, not a
# ceiling: too dark misses it as surely as too bright.
FLOOR, CEIL = 0.25, 1.9


def measure(song: str, seq: str) -> float | None:
    """quiet/loud - 1, i.e. -0.22 means quiet is 22% below loud."""
    ap = [p for p in analyse(song)["passages"] if p["ambient"] and p["dur"] >= 8]
    if not ap:
        return None
    f = read_fseq(SHOW / f"{seq}.fseq", frame_stride=8)
    lvl = np.array([np.frombuffer(f.frame(i), dtype=np.uint8).mean()
                    for i in range(len(f))])
    ts = np.arange(len(lvl)) * 8 * f.step_time_ms / 1000
    m = np.zeros(len(ts), bool)
    for p in ap:
        m |= (ts >= p["t0"]) & (ts < p["t1"])
    if not m.any() or m.all():
        return None
    return float(lvl[m].mean() / max(lvl[~m].mean(), 1e-9) - 1.0)


def main() -> None:
    for song in sys.argv[1:]:
        gen = SHOW / ".generation" / song
        cache = gen / "ambient_dim.json"
        cur = 0.5
        try:
            cur = float(json.loads(cache.read_text())["dim"])
        except (OSError, ValueError, KeyError):
            pass
        out = SHOW / f"_cal_{song}.xsq"
        subprocess.run([".venv/bin/python3", "custom_tools/generate_song.py", song,
                        "--out", out.name], capture_output=True)
        if not out.exists():
            print(f"{song}: generate failed")
            continue
        subprocess.run([str(XLIGHTS), "-s", str(SHOW), "-r", str(out)],
                       capture_output=True)
        got = measure(song, out.stem)
        if got is None:
            print(f"{song}: no ambient passages, nothing to calibrate")
            out.unlink(missing_ok=True)
            continue
        # ratio scales roughly with the factor: solve for the target.
        want = cur * (1.0 + TARGET) / max(1.0 + got, 1e-6)
        want = max(FLOOR, min(CEIL, want))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"dim": round(want, 3),
                                     "measured_at_dim": cur,
                                     "measured_ratio": round(got, 4)}))
        print(f"{song[:34]:34} at dim {cur:.2f} measured {got*100:+6.1f}%  ->  "
              f"dim {want:.2f}")
        out.unlink(missing_ok=True)
        (SHOW / f"{out.stem}.fseq").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
