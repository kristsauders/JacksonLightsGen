"""A song's structure as CHANGE POINTS, for hand-mapping SONG_SECTIONS.

    python custom_tools/section_plan.py "<song>"

song_map.py prints every bar, which is the right tool when you are studying one
song closely. This is the condensed form for planning a grid: it reports only
the bar lines where something actually turns -- total energy, bass, or hats --
plus the drop cues and how uniform the detector's own sections look.

Read it like a songwriter: bass in/out marks the drop and the breakdown, hats
in/out marks the lift inside an otherwise flat chorus, and a long run with no
change point is ONE section however many the detector claims.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")


def bar_lines(beats: np.ndarray, downbeats: np.ndarray, dur: float) -> np.ndarray:
    """Bar starts AT THE TRACKED DOWNBEATS, extended past the last one.

    A uniform grid (first downbeat + k * 4 * median beat) drifts: madmom
    reports beats on a 10 ms frame, so the median gap is quantized -- 12 to 12
    read 480 ms for a ~486 ms beat, and by bar 40 the grid sat 0.66 s (more
    than a beat) off the real downbeat, which then went into hand-mapped
    section times. Reading the bars off the downbeats themselves cannot drift.
    """
    beat_s = float(np.polyfit(np.arange(beats.size), beats, 1)[0]) if beats.size > 4 else 0.5
    if downbeats.size >= 4:
        bar_s = float(np.polyfit(np.arange(downbeats.size), downbeats, 1)[0])
        starts = list(downbeats)
    else:
        bar_s = beat_s * 4
        starts = [float(beats[0]) if beats.size else 0.0]
    while starts[-1] + bar_s < dur:
        starts.append(starts[-1] + bar_s)
    return np.asarray(starts + [starts[-1] + bar_s], dtype=float)


def main(song: str) -> None:
    gen = SHOW / ".generation" / song
    a = json.loads((gen / "analysis.json").read_text())
    cues = json.loads((gen / "cues.json").read_text()) if (gen / "cues.json").exists() else {}
    beats = np.asarray(a["beats"]["beat_times"], dtype=float)
    downbeats = np.asarray(a["beats"].get("downbeat_times") or [], dtype=float)
    dur = float(a["duration_seconds"])
    lines = bar_lines(beats, downbeats, dur)
    beat_s = float(np.polyfit(np.arange(beats.size), beats, 1)[0])   # not the quantized median
    bar_s = float(np.median(np.diff(lines)))

    y, sr = librosa.load(str(SHOW / f"{song}.mp3"), sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    rms = librosa.feature.rms(S=S)[0]
    peak = float(np.percentile(rms, 99)) + 1e-9
    low = S[freqs < 200].sum(axis=0); lowm = low.mean() + 1e-9
    hi = S[freqs > 4000].sum(axis=0); him = hi.mean() + 1e-9

    b0 = float(lines[0])
    n_bars = len(lines) - 1
    rows = []
    for k in range(n_bars):
        s0 = float(lines[k])
        m = (t >= s0) & (t < lines[k + 1])
        if not m.any():
            continue
        rows.append((k, s0, rms[m].mean() / peak, low[m].mean() / lowm, hi[m].mean() / him))

    print(f"\n=== {song}   {dur:.0f}s  tempo {60/beat_s:.1f}  bar {bar_s:.3f}s  "
          f"first downbeat {b0:.2f}s  ({len(rows)} bars)")
    print(f"    detector: {len(a['sections'])} sections, median "
          f"{np.median([x['end_time']-x['start_time'] for x in a['sections']]):.1f}s")
    drops = sorted({round(float(ev['drop_time']), 2)
                    for key in ("energy_drops", "predrop_cutouts", "instrument_dropouts")
                    for ev in cues.get(key, []) if ev.get("drop_time") is not None})
    if drops:
        print(f"    drop cues: {drops}")

    # COMPARE PHRASES, NOT BARS. A hi-hat pattern that drops out every other
    # bar is texture, not structure -- on Corners of the Earth a bar-to-bar
    # reading reported forty "hats IN / hats OUT" turns, none of them a section
    # change. Music is organised in four-bar phrases, so each phrase is
    # compared with the one before it and only real turns survive.
    PH = 4
    ph = []
    for i in range(0, len(rows) - PH + 1, PH):
        win = rows[i:i + PH]
        ph.append((win[0][0], win[0][1],
                   float(np.mean([w[2] for w in win])),
                   float(np.mean([w[3] for w in win])),
                   float(np.mean([w[4] for w in win]))))

    print(f"\n    {'bar':>4s} {'time':>8s} {'e':>5s} {'bass':>5s} {'hats':>5s}   what turned (4-bar phrases)")
    prev = None
    for k, s0, e, lo, ht in ph:
        if prev is None:
            print(f"    {k:4d} {s0:8.2f} {e:5.2f} {lo:5.2f} {ht:5.2f}   START")
            prev = (e, lo, ht); continue
        pe, plo, pht = prev
        why = []
        if abs(e - pe) > 0.10:
            why.append(f"energy {'UP' if e > pe else 'DOWN'} {pe:.2f}->{e:.2f}")
        if abs(lo - plo) > 0.30:
            why.append(f"bass {'IN' if lo > plo else 'OUT'} {plo:.2f}->{lo:.2f}")
        if abs(ht - pht) > 0.45:
            why.append(f"hats {'IN' if ht > pht else 'OUT'} {pht:.2f}->{ht:.2f}")
        if why:
            print(f"    {k:4d} {s0:8.2f} {e:5.2f} {lo:5.2f} {ht:5.2f}   " + ", ".join(why))
        prev = (e, lo, ht)
    print(f"\n    last bar {rows[-1][0]} at {rows[-1][1]:.2f}s; audio ends ~"
          f"{float(t[np.flatnonzero(rms > 0.02*rms.max())[-1]]):.1f}s")


if __name__ == "__main__":
    main(sys.argv[1])
