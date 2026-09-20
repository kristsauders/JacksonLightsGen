"""The song's shape on one screen, bar by bar, for deciding sections by hand.

    python custom_tools/song_map.py "<song>"

Step 3 of docs/new-song-runbook.md, made repeatable. The structure detector is
good at finding THAT something changes and poor at finding WHEN: measured on the
Bluey trap remix its eight uniform blocks scored 0.331 mean novelty at their
boundaries while a hand-read grid put every boundary on a bar line. Reading a
bar map is what makes that call possible in a couple of minutes.

What it prints:

  1. tempo, beat length, jitter, and where the beat tracker wobbled (usually a
     near-empty passage, which is harmless -- worth knowing, not fixing)
  2. one line per bar: energy, bass share, hat share, and a bar chart
  3. the DETECTED sections, each with the energy change across its boundary
     (novelty) and its distance to the nearest beat and downbeat
  4. drop candidates from cues.json, each with the measured hush and lift
  5. a SUGGESTED grid: bar lines where the energy step is large enough to read
     as a section change, ready to paste into SONG_SECTIONS

Nothing here writes anything. The decisions stay with the human.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")


def main(song: str) -> None:
    gen = SHOW / ".generation" / song
    a = json.loads((gen / "analysis.json").read_text())
    cues = json.loads((gen / "cues.json").read_text()) if (gen / "cues.json").exists() else {}
    beats = np.asarray(a["beats"]["beat_times"], dtype=float)
    downbeats = np.asarray(a["beats"].get("downbeat_times") or [], dtype=float)
    dur = float(a["duration_seconds"])
    tempo = float(a["beats"]["tempo"])

    y, sr = librosa.load(str(SHOW / f"{song}.mp3"), sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    rms = librosa.feature.rms(S=S)[0]
    peak = float(np.percentile(rms, 99)) + 1e-9
    low = S[freqs < 200].sum(axis=0)
    hi = S[freqs > 4000].sum(axis=0)
    lowm, him = low.mean() + 1e-9, hi.mean() + 1e-9

    d = np.diff(beats)
    beat_s = float(np.median(d))
    bar_s = beat_s * 4
    print(f"\n=== {song}")
    print(f"  {dur:.1f}s, tempo {tempo:.1f} ({60/beat_s:.1f} from spacing), "
          f"beat {1000*beat_s:.0f}ms, bar {bar_s:.3f}s, "
          f"{len(beats)} beats, {len(downbeats)} downbeats, jitter sd {1000*np.std(d):.0f}ms")
    odd = np.flatnonzero(np.abs(d - beat_s) > 0.06)
    if odd.size:
        print("  beat-tracker wobbles: "
              + ", ".join(f"{beats[i]:.2f}s({1000*d[i]:.0f}ms)" for i in odd[:10]))

    def win(t0: float, t1: float):
        m = (t >= t0) & (t < t1)
        if not m.any():
            return None
        return (rms[m].mean() / peak, low[m].mean() / lowm, hi[m].mean() / him)

    b0 = float(downbeats[0]) if downbeats.size else float(beats[0])
    n_bars = int((dur - b0) // bar_s) + 1
    print(f"\n  bar map from the first downbeat ({b0:.2f}s), one line per bar:")
    prev_e = None
    for k in range(n_bars):
        s0 = b0 + k * bar_s
        v = win(s0, s0 + bar_s)
        if v is None:
            continue
        e, lo, ht = v
        step = "" if prev_e is None else ("  <<< step up" if e - prev_e > 0.12
                                          else "  <<< step down" if prev_e - e > 0.12 else "")
        prev_e = e
        print(f"   bar{k:3d} {s0:7.2f}s |{'#'*int(e*40):40s}| e{e:4.2f} bass{lo:4.2f} hats{ht:4.2f}{step}")

    def novelty(tt: float) -> float:
        b = (t >= tt - 4) & (t < tt)
        af = (t >= tt) & (t < tt + 4)
        if not b.any() or not af.any():
            return 0.0
        return float(abs(rms[af].mean() - rms[b].mean()) / (rms.mean() + 1e-9))

    print(f"\n  DETECTED sections ({len(a['sections'])}):")
    print(f"  {'#':>2s} {'start':>7s} {'end':>7s} {'len':>6s} {'label':10s} "
          f"{'loud':>5s} {'novelty':>8s} {'to beat':>8s} {'to bar':>8s}")
    for i, s in enumerate(a["sections"]):
        st, en = float(s["start_time"]), float(s["end_time"])
        v = win(st, en)
        nb = 1000 * (beats[np.argmin(np.abs(beats - st))] - st) if beats.size else 0
        nd = (1000 * (downbeats[np.argmin(np.abs(downbeats - st))] - st)
              if downbeats.size else 0)
        print(f"  {i:2d} {st:7.2f} {en:7.2f} {en-st:6.1f} {str(s.get('label'))[:10]:10s} "
              f"{(v[0] if v else 0):5.2f} {novelty(st):8.3f} {nb:+7.0f}ms {nd:+7.0f}ms")

    drops = []
    for key in ("energy_drops", "predrop_cutouts", "instrument_dropouts"):
        for ev in cues.get(key, []):
            if ev.get("drop_time") is not None:
                drops.append((float(ev["drop_time"]), key.replace("_", " ")))
    if drops:
        print(f"\n  drop candidates from cues.json ({len(drops)}):")
        print(f"  {'time':>7s} {'kind':20s} {'hush':>6s} {'lift':>6s} {'on beat':>9s}")
        for tt, kind in sorted(drops):
            pre = rms[(t >= tt - 2) & (t < tt)]
            base = rms[(t >= tt - 10) & (t < tt - 2)]
            post = rms[(t >= tt) & (t < tt + 1)]
            hush = float(pre.mean() / base.mean()) if pre.size and base.size else 0
            lift = float(post.max() / pre.mean()) if pre.size and post.size else 0
            off = 1000 * (beats[np.argmin(np.abs(beats - tt))] - tt) if beats.size else 0
            print(f"  {tt:7.2f} {kind:20s} {hush:6.2f} {lift:6.2f} {off:+8.0f}ms")

    # Suggested grid: bar lines whose energy step is big enough to read as a change.
    steps = []
    for k in range(1, n_bars):
        a0 = win(b0 + (k - 1) * bar_s, b0 + k * bar_s)
        a1 = win(b0 + k * bar_s, b0 + (k + 1) * bar_s)
        if a0 and a1 and abs(a1[0] - a0[0]) > 0.12:
            steps.append((k, b0 + k * bar_s, a1[0] - a0[0]))
    if steps:
        print(f"\n  SUGGESTED boundaries (bar lines with an energy step > 0.12):")
        for k, tt, dv in steps:
            print(f"    bar{k:3d} {tt:7.2f}s  {'up  ' if dv > 0 else 'down'} {dv:+.2f}")
        print("\n  paste-ready for SONG_SECTIONS (label them by ear):")
        print("    " + ", ".join(f"({tt:.2f}, \"?\")" for _, tt, _ in [(0, b0, 0)] + steps))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Gooey")
