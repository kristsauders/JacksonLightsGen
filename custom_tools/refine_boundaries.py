"""Put a medley's hand-read section times on the beat where the music turns.

    python custom_tools/refine_boundaries.py "<mix name>" [song filter ...] [--write]

Hand-reading a bar map gets a boundary to within a bar or so; it does not get
it onto the right beat. Two measured reasons:

* the bar maps used to lay a UNIFORM grid (first downbeat + k bars at the median
  beat), and madmom's 10 ms frames quantize that median -- by bar 40 of 12 to 12
  the grid sat 0.66 s off the real downbeat (now fixed in song_map/section_plan);
* the tracker's downbeat PHASE is not reliable through a DJ mix: inside greedy
  the nearest "downbeat" to the hand-read boundaries alternated +0.9 s / -0.9 s
  (two beats either way), so snapping to downbeats would have moved them onto
  the wrong beat.

So each boundary is tested against every beat of the whole-mix grid within
+-1.3 s: the score is how much the music changes across that beat (overall
level, bass, hats; two beats each side, log ratio), with a small bonus for a
tracked downbeat. The best beat wins. Song starts (the cut points, already on
downbeats) and anything listed in a song's "reveal" are never moved. Drops
stay locked to their section when they share a time.

Without --write it only prints old -> new with the score of each, so a person
can look before anything changes.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
ROOT = Path(__file__).resolve().parent
WINDOW = 1.3          # seconds either side of the hand-read time
DOWNBEAT_BONUS = 1.10


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    mix, filters = args[0], args[1:]
    G = SHOW / ".generation" / mix
    plan = json.loads((G / "segments.json").read_text())
    mb = json.loads((G / "mix_beats.json").read_text())
    beats = np.asarray(mb["beats"]); downs = set(np.round(mb["downbeats"], 3))
    slug = re.sub(r"[^a-z0-9]+", "_", mix.lower()).strip("_")
    dpath = ROOT / "medleys" / f"{slug}.py"
    spec = importlib.util.spec_from_file_location("medley_design", dpath)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    audio = next((SHOW / f"{mix}{ext}" for ext in (".flac", ".wav", ".mp3")
                  if (SHOW / f"{mix}{ext}").exists()))
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    # Each feature normalised by its whole-mix mean, compared LINEARLY over a
    # bar either side. A log ratio over two beats was tried first and was
    # wrong on exactly the moments that matter most: out of a silence, the
    # step from nothing into a half-bar vocal pickup scored higher than the
    # drop itself (Santa Claus Is Coming to Town: silence 20:41.7, pickup
    # 20:42.8, drop 20:43.60 -- the log ratio chose 20:42.66).
    feats = [(x / (x.mean() + 1e-9), w) for x, w in
             ((librosa.feature.rms(S=S)[0], 1.0),
              (S[f < 200].sum(axis=0), 1.0),
              (S[f > 4000].sum(axis=0), 0.5))]

    def score(c: float, beat: float) -> float:
        b = (t >= c - 4 * beat) & (t < c)
        a = (t >= c) & (t < c + 4 * beat)
        if not a.any() or not b.any():
            return 0.0
        return sum(w * abs(x[a].mean() - x[b].mean()) for x, w in feats)

    starts = {round(p["start"], 2) for p in plan}
    src = dpath.read_text()
    changes: list[tuple[str, str, str]] = []
    for name, d in mod.DESIGN.items():
        if filters and not any(k in name for k in filters):
            continue
        locked = set(starts)
        if "reveal" in d:
            locked |= {round(x, 2) for x in d["reveal"]["hits"] + [d["reveal"]["drop"]]}
        print(f"## {name}")
        for t0, lab in d.get("sections", []):
            if round(t0, 2) in locked:
                print(f"   {t0:8.2f} {lab:8s} locked")
                continue
            near = beats[np.abs(beats - t0) <= WINDOW]
            if near.size < 2:
                continue
            beat = float(np.median(np.diff(beats[np.abs(beats - t0) <= 6])))
            sc = [(score(c, beat) * (DOWNBEAT_BONUS if round(c, 3) in downs else 1.0), c)
                  for c in near]
            sc.sort(reverse=True)
            here = score(t0, beat)
            nearest = float(near[np.argmin(np.abs(near - t0))])
            # Trust the measurement only where it is decisive: a real turn
            # scores well above 1 and clearly above the beat as read. On a
            # weak boundary (a vocal phrase over a steady groove) the scores
            # are all alike and the best one is noise -- there the hand-read
            # time is kept and only put onto its nearest beat.
            confident = sc[0][0] >= 1.0 and sc[0][0] >= 1.25 * max(here, sc[1][0] * 0.9)
            best = sc[0][1] if confident else nearest
            why = "turn" if confident else "nearest beat"
            flag = "" if abs(best - t0) < 0.06 else f"  moved {best - t0:+.2f}s ({why})"
            print(f"   {t0:8.2f} {lab:8s} -> {best:8.2f}  best {sc[0][0]:.2f} at {sc[0][1]:.2f} "
                  f"(as read {here:.2f}; next {sc[1][0]:.2f} at {sc[1][1]:.2f}){flag}")
            if flag:
                changes.append((name, f"{t0:.2f}", f"{best:.2f}"))
    if write and changes:
        for name, old, new in changes:
            # Only inside this song's block, and only whole numbers: "361.57"
            # must not match inside "1361.57".
            i0 = src.index(f'"{name}"')
            i1 = src.find('\n    "', i0 + 1)
            i1 = len(src) if i1 < 0 else i1
            blk = re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d])", new, src[i0:i1])
            src = src[:i0] + blk + src[i1:]
        dpath.write_text(src)
        print(f"wrote {len(changes)} boundary moves to {dpath.name}")


if __name__ == "__main__":
    main()
