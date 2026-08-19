"""Find anything on the house that flashes fast enough to read as a strobe.

    python custom_tools/verify_strobe.py "Gooey (v124)" ...
    python custom_tools/verify_strobe.py --all

"No strobing" is the one hard rule in this project -- "it can give people
migraines" -- and floods are the worst offender because they are big, bright
and close to the viewer. Every other check here looks at what was PLACED;
this one looks at what a person standing in the yard would actually see.

Method: for each prop, take its whole-prop MEAN brightness per frame and
count how often that mean swings by more than `MIN_SWING` between adjacent
frames. Whole-prop mean is the right signal, and per-pixel change is not: a
shader with bands sliding across it changes most of its pixels every frame
and is completely calm to look at, while a prop whose entire surface jumps
from dark to bright and back at 25ms is a strobe even if the pixels move
only a little. Measured on the shader library, that distinction is a factor
of four (Oil Film: 0.038 per-pixel, 0.010 whole-prop).

A run of alternating swings is what makes a strobe rather than a single
accent, so the report counts SUSTAINED runs -- at least `MIN_RUN` consecutive
swings -- and prints where they are, so the responsible pass can be found by
looking at that moment in the .xsq.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import channel_map  # noqa: E402
from fseq_reader import read_fseq  # noqa: E402

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
# Fraction of full brightness the whole prop must swing, frame to frame.
# Peak-to-trough swing of the WHOLE prop before it counts as a flash at all.
# 0.10 was far too low: a shader with fine moving detail ripples the prop mean
# by that much every couple of frames, and the detector reported 20 Hz
# "strobes" that are just texture. A flash people would call a flash takes the
# prop most of the way down and back.
MIN_SWING = 0.30
FLOOD_SWING = 0.20          # floods held to a stricter standard
# Faster than this many flashes per second reads as a strobe. The standing
# rule is 2-4/sec; 8th notes at this library's tempos are 4.3-4.8 Hz.
MAX_HZ = 4.5
# How many peaks in a row before it stops being an accent and starts being
# flicker.
MIN_PEAKS = 4
# Below this spacing it is not a sequence of flashes anyone can count -- it is
# shimmer inside one effect, and belongs to the texture, not the rhythm.
MIN_GAP_FR = 3


def scan(seq: str) -> list[dict]:
    f = read_fseq(SHOW_PATH / f"{seq}.fseq", frame_stride=1)
    cm = channel_map.build()["models"]
    props = {n: (v["first"], v["last"] + 1) for n, v in cm.items()
             if v["count"] >= 90}
    n = len(f)
    frames = np.zeros((n, len(props)), dtype=np.float32)
    spans = list(props.values())
    for i in range(n):
        row = np.frombuffer(f.frame(i), dtype=np.uint8).astype(np.float32)
        for j, (a, b) in enumerate(spans):
            frames[i, j] = row[a:b].mean()
    frames /= 255.0

    hits = []
    fps = 1000.0 / f.step_time_ms
    for j, name in enumerate(props):
        lvl = frames[:, j]
        is_flood = "flood" in name.lower()
        prom = FLOOD_SWING if is_flood else MIN_SWING
        # PEAKS, not frame-to-frame swings. A single hit has a sharp attack
        # and decay and trips any swing test; a strobe is the RATE at which
        # peaks repeat. Krists put a number on it -- "too fast, it looks like
        # strobing, where it's to 8th notes or something" -- and 8ths at this
        # library's tempos are 4.3-4.8 Hz, against a standing rule of no more
        # than 2-4 flashes a second.
        peaks = []
        for i in range(1, len(lvl) - 1):
            if lvl[i] >= lvl[i - 1] and lvl[i] > lvl[i + 1]:
                lo = min(lvl[max(0, i - 6):i].min(initial=lvl[i]),
                         lvl[i + 1:i + 7].min(initial=lvl[i]))
                if lvl[i] - lo >= prom:
                    peaks.append(i)
        # A run of >=MIN_PEAKS consecutive peaks spaced faster than MAX_HZ.
        k = 0
        while k < len(peaks) - 1:
            m = k
            while (m < len(peaks) - 1
                   and MIN_GAP_FR <= peaks[m + 1] - peaks[m]
                   and (peaks[m + 1] - peaks[m]) / fps < 1.0 / MAX_HZ):
                m += 1
            if m - k + 1 >= MIN_PEAKS:
                span = (peaks[m] - peaks[k]) / fps
                hz = (m - k) / max(span, 1e-6)
                hits.append({"prop": name, "t": peaks[k] / fps,
                             "n": m - k + 1, "hz": hz, "flood": is_flood})
            k = m + 1
    return hits


def main() -> None:
    args = sys.argv[1:]
    if args == ["--all"]:
        seqs = []
        for mp3 in sorted(SHOW_PATH.glob("*.mp3")):
            vs = sorted(SHOW_PATH.glob(f"{mp3.stem} (v*).fseq"),
                        key=lambda p: int(p.stem.split("(v")[1].rstrip(")")))
            if vs:
                seqs.append(vs[-1].stem)
    else:
        seqs = args
    total = 0
    for seq in seqs:
        try:
            hits = scan(seq)
        except (OSError, KeyError) as e:
            print(f"{seq}: {e}")
            continue
        if not hits:
            print(f"{seq}: clean")
            continue
        total += len(hits)
        print(f"\n{seq}: {len(hits)} strobe runs (>{MAX_HZ} Hz, >={MIN_PEAKS} flashes)")
        worst = sorted(hits, key=lambda h: (-h["flood"], -h["hz"]))[:10]
        for h in worst:
            print(f"    {h['t']:7.1f}s  {h['prop'][:34]:34} "
                  f"{h['n']:3} flashes @ {h['hz']:4.1f} Hz"
                  + ("   <-- FLOOD" if h["flood"] else ""))
    print(f"\n{total} runs across {len(seqs)} sequences")


if __name__ == "__main__":
    main()
