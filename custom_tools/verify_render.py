"""Render a sequence with xLights headlessly and measure what it ACTUALLY looks like.

    python custom_tools/verify_render.py "Song (v5).xsq" [--no-render]

This is the ground-truth check. `verify_sequence.py` inspects the .xsq (what we
asked for); this renders it and reads the resulting .fseq (what xLights
produces). The difference matters: four rounds of this project were spent
hunting a supposed xLights "red flash render bug" on large props that turned
out to be the generator's own colour palettes leading with pure #FF0000 --
something no amount of .xsq inspection would ever have revealed, and which
this tool shows in one run.

Reference numbers, measured from Krists's own finished sequences:
    red-of-lit   mean 0.3-11%, peak <= 27%     (Silent Night 2.3%,
                                                Blinding Lights 11.2%,
                                                Pink Pony Club 0.3%)
    lit fraction ~50% of pixels                (Silent Night 50%,
                                                Pink Pony Club 54%)
A generated sequence sitting far above the red band, ESPECIALLY sustained over
many seconds rather than as brief accents, means a palette problem.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fseq_reader import frame_color_stats, read_fseq  # noqa: E402

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
XLIGHTS = Path("/Applications/xLights.app/Contents/MacOS/xLights")

RED_MEAN_OK = 0.12     # human mean tops out ~11%
RED_PEAK_OK = 0.30     # human peak tops out ~27%


def render(xsq: Path) -> Path:
    """Render to .fseq via the xLights CLI. Takes ~20-60s for a 3-minute show."""
    print(f"rendering {xsq.name} ...", flush=True)
    res = subprocess.run(
        [str(XLIGHTS), "-s", str(SHOW_PATH), "-r", str(xsq)],
        capture_output=True, text=True)
    fseq = xsq.with_suffix(".fseq")
    if not fseq.exists():
        print(res.stdout[-2000:])
        raise SystemExit(f"render produced no {fseq.name}")
    return fseq


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    xsq = Path(args[0])
    if not xsq.is_absolute():
        xsq = SHOW_PATH / xsq
    fseq = xsq.with_suffix(".fseq")
    if "--no-render" not in sys.argv or not fseq.exists():
        fseq = render(xsq)

    stride = 40  # one sample per second at 25ms/frame
    f = read_fseq(fseq, frame_stride=stride)
    print(f"\n{fseq.name}: {f.frame_count} frames @ {f.step_time_ms}ms "
          f"({f.duration_s:.0f}s), sampled {len(f)}")

    reds, lits, worst = [], [], []
    for i in range(len(f)):
        s = frame_color_stats(f.frame(i))
        if s["lit_frac"] <= 0.02:
            continue
        t = i * stride * f.step_time_ms / 1000.0
        reds.append(s["red_frac"])
        lits.append(s["lit_frac"])
        worst.append((s["red_frac"], t))

    if not reds:
        print("sequence renders essentially dark -- nothing to measure")
        return

    mean_red, peak_red = statistics.mean(reds), max(reds)
    print(f"  lit fraction : mean {statistics.mean(lits) * 100:.0f}%   "
          f"[human ~50%]")
    print(f"  red of lit   : mean {mean_red * 100:.1f}%  peak {peak_red * 100:.1f}%   "
          f"[human mean 0.3-11%, peak <=27%]")

    worst.sort(reverse=True)
    hot = [(r, t) for r, t in worst if r > RED_PEAK_OK]
    if hot:
        print(f"  reddest moments: " + ", ".join(
            f"{int(t) // 60}:{int(t) % 60:02d} ({r * 100:.0f}%)" for r, t in hot[:8]))

    ok = mean_red <= RED_MEAN_OK and peak_red <= RED_PEAK_OK
    print(f"\n  VERDICT: {'OK' if ok else 'TOO RED -- check palette lead colours'}")


if __name__ == "__main__":
    main()
