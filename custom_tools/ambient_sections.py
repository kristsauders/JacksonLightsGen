"""Find the sections that should NOT be sequenced rhythmically.

    python custom_tools/ambient_sections.py "This Version of You"

The generator's whole vocabulary assumes a beat to hang things on: an anchor
prop flashing the kick, voices mapped to props, gutter flashes, submodel
stepping, leaping arches. On a chorus that is exactly right. On a drumless
intro or a breakdown it produces "lights pulsing and being very rhythmic when
they need to be more smooth and flowy".

So those passages have to be identified before anything is placed, and given
a different treatment entirely -- one built around the section's SHAPE rather
than its pulse:

  * `quiet`     the section's own loudness against the song's loudest
  * `drum/s`    kick+snare onsets per second -- the thing that makes a section
                sequenceable rhythmically at all
  * `events`    the sparse hits that ARE the rhythm here: bass notes, chord
                changes, the occasional kick. In a drumless passage these
                often land once every bar or two, and they are what a viewer
                actually perceives as timing.
  * `build`     how much the section's energy rises from its first quarter to
                its last -- a section that builds wants a gesture that goes
                somewhere, one that is flat wants texture that develops.
  * `enters`    voices that start or stop inside the section. A new instrument
                arriving is the most legible moment a quiet passage has, and
                the show should mark it.

Printed as a plan per song so the choices can be reviewed before generating,
which is the part Krists asked for: "review the plan for each section but
think about what data to use to help you."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from instrument_voices import detect_instrument_voices  # noqa: E402

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
# Fewer drum hits per second than this and there is no pulse to sequence to.
# One hit every ~2.5s: slower than any groove, and about where a passage stops
# reading as having a beat at all.
AMBIENT_DRUM_RATE = 0.40
# ...or the section is simply quiet, whatever its drums are doing.
AMBIENT_LOUD = 0.55


def analyse(song: str) -> dict:
    gen = SHOW_PATH / ".generation" / song
    a = json.loads((gen / "analysis.json").read_text())
    beats = a["beats"]["beat_times"]
    voices = detect_instrument_voices(SHOW_PATH / "stems" / song, beats,
                                      a["beats"]["tempo"])
    rms = np.array(a["spectrum"]["rms_energy"])
    rt = np.array(a["spectrum"]["rms_times"])
    peak = float(np.percentile(rms, 99)) + 1e-9

    def band(t0, t1):
        m = (rt >= t0) & (rt < t1)
        return rms[m] if m.any() else np.array([0.0])

    rows = []
    for i, s in enumerate(a["sections"]):
        t0, t1 = s["start_time"], s["end_time"]
        dur = max(t1 - t0, 1e-6)
        drum = sum(len(voices[v].in_window(t0, t1))
                   for v in ("kick", "snare") if v in voices)
        ev = sorted(t for v in ("kick", "bass", "chord")
                    if v in voices for t in voices[v].in_window(t0, t1))
        # Thin to what a viewer would count as separate moments.
        sparse, last = [], -9.0
        for t in ev:
            if t - last >= 0.9:
                sparse.append(t)
                last = t
        e = band(t0, t1)
        q = max(1, len(e) // 4)
        build = float(e[-q:].mean() / max(e[:q].mean(), 1e-9))
        enters = []
        for v in ("kick", "snare", "bass", "chord", "vocal"):
            if v not in voices:
                continue
            before = len(voices[v].in_window(max(0, t0 - 6), t0))
            during = len(voices[v].in_window(t0, min(t1, t0 + 6)))
            if during >= 3 and before == 0:
                enters.append("+" + v)
            elif before >= 3 and during == 0:
                enters.append("-" + v)
        rows.append({
            "i": i, "t0": t0, "t1": t1, "dur": dur,
            "loud": float(e.mean() / peak),
            "drum_rate": drum / dur,
            "sparse": sparse,
            "build": build,
            "enters": enters,
        })

    for r in rows:
        r["ambient"] = (r["drum_rate"] < AMBIENT_DRUM_RATE
                        or r["loud"] < AMBIENT_LOUD)

    # MERGE INTO PASSAGES. The section detector emits slivers -- 100 of them
    # in 172s on This Version of You, about one per bar -- and a plan written
    # against those is unreadable and, worse, would have the show restate
    # itself every 1.1 seconds. What matters here is the contiguous stretch
    # of drumless time and how it develops from end to end, so adjacent
    # sections sharing a classification are fused and the measurements
    # recomputed over the whole span.
    passages = []
    for r in rows:
        if passages and passages[-1]["ambient"] == r["ambient"]:
            passages[-1]["t1"] = r["t1"]
            passages[-1]["members"].append(r["i"])
        else:
            passages.append({"t0": r["t0"], "t1": r["t1"],
                             "ambient": r["ambient"], "members": [r["i"]]})
    for p in passages:
        t0, t1 = p["t0"], p["t1"]
        p["dur"] = t1 - t0
        e = band(t0, t1)
        q = max(1, len(e) // 4)
        p["loud"] = float(e.mean() / peak)
        p["build"] = float(e[-q:].mean() / max(e[:q].mean(), 1e-9))
        p["drum_rate"] = sum(len(voices[v].in_window(t0, t1))
                             for v in ("kick", "snare") if v in voices) / max(p["dur"], 1e-6)
        ev = sorted(t for v in ("kick", "bass", "chord")
                    if v in voices for t in voices[v].in_window(t0, t1))
        sparse, last = [], -9.0
        for t in ev:
            if t - last >= 0.9:
                sparse.append(t)
                last = t
        p["sparse"] = sparse
        seen = []
        for r in rows:
            if r["i"] in p["members"]:
                for x in r["enters"]:
                    if x not in seen:
                        seen.append(x)
        p["enters"] = seen
    return {"song": song, "rows": rows, "passages": passages}


def main() -> None:
    for song in sys.argv[1:]:
        p = analyse(song)
        rows = p["passages"]
        amb = [r for r in rows if r["ambient"]]
        secs = sum(r["dur"] for r in amb)
        total = sum(r["dur"] for r in rows)
        print(f"\n{'=' * 76}\n{song}   {len(amb)}/{len(rows)} passages ambient "
              f"({secs:.0f}s of {total:.0f}s, {secs/max(total,1)*100:.0f}%)")
        print(f"{'#':>3} {'span':>14} {'loud':>5} {'drum/s':>7} {'events':>7} "
              f"{'build':>6}  what")
        for r in rows:
            tag = "AMBIENT" if r["ambient"] else "rhythmic"
            ev = len(r["sparse"])
            rate = f"{ev/r['dur']:.2f}/s" if r["dur"] else ""
            print(f"{'':>3} {r['t0']:6.1f}-{r['t1']:6.1f} {r['loud']:5.2f} "
                  f"{r['drum_rate']:7.2f} {ev:4}{'':3} {r['build']:6.2f}  {tag}"
                  + (f"  {' '.join(r['enters'])}" if r["enters"] else "")
                  + (f"  [{rate}]" if r["ambient"] and ev else ""))


if __name__ == "__main__":
    main()
