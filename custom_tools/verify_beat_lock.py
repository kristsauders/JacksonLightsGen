"""Does the render actually ACCENT THE BEAT, and does it DIP before a section?

    python custom_tools/verify_beat_lock.py "Gooey (v20)" Gooey

Two complaints have come back from Krists more times than any others, in
almost the same words each round:

  * "the shaders have always been not tied to the beat enough"
  * "the dynamic detection that would get a bit dimmer before a new phrase or
    drop or fill isn't working, it's too static and always lit up"

Both are about the RELATIONSHIP between light and time, which neither of the
other verifiers can see: verify_sequence.py checks what we asked for, and
verify_render.py checks aggregate colour. A sequence can pass both while
having no rhythmic relationship to the music whatsoever -- measured, that is
exactly what was happening (ratio 0.98, i.e. beats were very slightly DARKER
than the gaps between them).

Two numbers, both from the real .fseq:

  BEAT-LOCK RATIO -- mean brightness within +-60ms of a detected beat, over
    mean brightness off-beat. 1.00 means no beat relationship at all. Measured
    reference points on Gooey after the 2026-08-11 fix: a Shader carrying a
    per-beat brightness value curve reads ~1.44, a built-in wash effect with
    the same curve at 60% depth ~1.15. Anything at 1.0x +-0.02 is a prop that
    is on but not participating in the music -- treat that as a bug, not taste.

  RUN-UP DIP -- lit fraction in the last beat before a section boundary over
    the lit fraction of the preceding seven beats. 1.00 means the display
    barrels into the new section at full brightness with no breath before it.
    Post-fix Gooey: median 0.73, dipping on 39 of 56 approaches.

Per-model beat-lock needs the measured channel map (custom_tools/channel_map.py);
the whole-display numbers work without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from fseq_reader import read_fseq  # noqa: E402

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")

# Props worth checking individually: the big canvases that carry shaders and
# washes, which is where "not tied to the beat" was actually visible.
DEFAULT_PROBES = ["Mega Tree", "Sun Room Window Matrix", "Garage Roof",
                  "Tunnel Matrix Real"]

BEAT_TOL_S = 0.06          # +-60ms: about how tight a viewer reads as "on it"


def beat_lock(fseq, beats: list[float], t0: float, t1: float,
              lo: int | None = None, hi: int | None = None) -> dict | None:
    """Brightness on-beat vs off-beat over [t0, t1], optionally for one model's
    channel range."""
    step = fseq.step_time_ms / 1000.0
    i0, i1 = max(0, int(t0 / step)), min(len(fseq) - 1, int(t1 / step))
    if i1 <= i0:
        return None
    ts, bs = [], []
    for i in range(i0, i1 + 1):
        arr = np.frombuffer(fseq.frame(i), dtype=np.uint8)
        if lo is not None:
            arr = arr[lo:hi]
        ts.append(i * step)
        bs.append(float(arr.mean()))
    ts, bs = np.array(ts), np.array(bs)
    if bs.mean() < 1.0:
        return None                       # dark here; nothing to say about it
    on = np.zeros(len(ts), bool)
    n = 0
    for b in beats:
        if t0 <= b <= t1:
            on |= np.abs(ts - b) <= BEAT_TOL_S
            n += 1
    off = ~on
    if n < 4 or not on.any() or not off.any():
        return None
    return {"beats": n, "on": float(bs[on].mean()), "off": float(bs[off].mean()),
            "ratio": float(bs[on].mean() / bs[off].mean())}


def run_up_dip(fseq, section_starts: list[float], beat_s: float) -> dict | None:
    """Lit fraction in the last beat before each section vs the seven before."""
    step = fseq.step_time_ms / 1000.0

    def lit(t0: float, t1: float) -> float | None:
        i0, i1 = max(0, int(t0 / step)), min(len(fseq) - 1, int(t1 / step))
        if i1 <= i0:
            return None
        v = [float((np.frombuffer(fseq.frame(i), dtype=np.uint8) > 40).mean())
             for i in range(i0, i1 + 1)]
        return float(np.mean(v))

    ratios = []
    for s in section_starts:
        dip = lit(s - beat_s, s)
        base = lit(s - 9 * beat_s, s - 2 * beat_s)
        # Skip approaches that are already dark -- a ratio against near-zero
        # is noise, and a section arriving out of silence needs no dip.
        if dip is None or base is None or base < 0.02:
            continue
        ratios.append(dip / base)
    if not ratios:
        return None
    return {"n": len(ratios), "median": float(np.median(ratios)),
            "dipping": sum(r < 0.95 for r in ratios)}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    seq, song = sys.argv[1], sys.argv[2]
    a = json.loads((SHOW_PATH / ".generation" / song / "analysis.json").read_text())
    beats = a["beats"]["beat_times"]
    beat_s = 60.0 / a["beats"]["tempo"]
    section_starts = [s["start_time"] for s in a["sections"]][1:]
    f = read_fseq(SHOW_PATH / f"{seq}.fseq", frame_stride=1)
    dur = len(f) * f.step_time_ms / 1000.0

    print(f"{seq}   ({dur:.0f}s, tempo {a['beats']['tempo']:.1f})\n")

    # Whole display, over the middle 60% of the song (skip intro/outro, which
    # are legitimately sparse and would drag the average toward 1.0).
    t0, t1 = dur * 0.2, dur * 0.8
    whole = beat_lock(f, beats, t0, t1)
    print("BEAT LOCK  (brightness on-beat / off-beat; 1.00 = no relationship)")
    if whole:
        print(f"  whole display {t0:.0f}-{t1:.0f}s : {whole['ratio']:6.3f}"
              f"   ({whole['beats']} beats)")

    try:
        import channel_map
        cmap = channel_map.build()["models"]
    except Exception as exc:                       # noqa: BLE001
        print(f"  (per-prop skipped: {exc})")
        cmap = {}
    for name in DEFAULT_PROBES:
        info = cmap.get(name)
        if not info:
            continue
        r = beat_lock(f, beats, t0, t1, info["first"], info["last"] + 1)
        if r is None:
            print(f"  {name:24}: dark in this window")
            continue
        flag = "" if r["ratio"] > 1.03 else "   <-- NOT participating"
        print(f"  {name:24}: {r['ratio']:6.3f}{flag}")

    print("\nRUN-UP DIP  (lit in last beat / preceding 7 beats before a section)")
    d = run_up_dip(f, section_starts, beat_s)
    if d:
        flag = "" if d["median"] < 0.95 else "   <-- no dip; display is static"
        print(f"  median {d['median']:.3f}{flag}")
        print(f"  approaches dipping >5%: {d['dipping']}/{d['n']}")


if __name__ == "__main__":
    main()
