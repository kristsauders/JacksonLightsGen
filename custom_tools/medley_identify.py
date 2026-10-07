"""Steps 1-2 of a DJ mix (runbook section 9): the beat grid, and what is playing.

    python custom_tools/medley_identify.py "<mix>" beats
    uv run --python 3.12 --with shazamio python custom_tools/medley_identify.py "<mix>" shazam
    uv run --with openai-whisper python custom_tools/medley_identify.py "<mix>" lyrics

Reads SHOW/<mix>.flac (or .wav/.mp3); writes into SHOW/.generation/<mix>/.

beats  -- ONE madmom joint beat + downbeat pass over the whole mix
          (mix_beats.json). The per-song pipeline takes beats from librosa
          (one global tempo) and downbeats from madmom separately; neither
          survives a DJ tempo ramp and the two disagree with each other. A
          joint DBN pass follows gradual tempo changes and keeps every
          downbeat on a beat by construction. ~10 min for 29 min of audio.
shazam -- fingerprint a 10 s clip every 10 s (shazam_10s.json: t, title,
          artist, offset into the original). Only the fingerprint leaves the
          machine. Needs Python <= 3.12 (shazamio uses audioop). Remixes,
          instrumentals and unreleased edits come back None -- ask the owner.
          A linear fit of offset against mix time gives the DJ's speed change
          per song and where in the original the mix enters it.
lyrics -- Whisper "small" over the whole mix (lyrics_whisper.json). Places
          vocals for section-mapping, and catches what Shazam misses (it heard
          a "Last Christmas" vocal over the Deck the Halls remix that Shazam
          called sleep music).
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")


def audio_of(mix: str) -> Path:
    for ext in (".flac", ".wav", ".mp3"):
        if (SHOW / f"{mix}{ext}").exists():
            return SHOW / f"{mix}{ext}"
    raise SystemExit(f"no audio for {mix!r} in {SHOW}")


def beats(mix: str, out: Path) -> None:
    import numpy as np
    from madmom.features.downbeats import (DBNDownBeatTrackingProcessor,
                                           RNNDownBeatProcessor)
    act = RNNDownBeatProcessor()(str(audio_of(mix)))
    res = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100,
                                       min_bpm=55, max_bpm=190)(act)
    bt = [float(r[0]) for r in res]
    pos = [int(r[1]) for r in res]
    down = [b for b, q in zip(bt, pos) if q == 1]
    (out / "mix_beats.json").write_text(json.dumps({"beats": bt, "positions": pos,
                                                    "downbeats": down}))
    d = np.diff(bt)
    print(f"{len(bt)} beats, {len(down)} downbeats, median {60 / np.median(d):.1f} BPM")


def shazam(mix: str, out: Path, step: float = 10.0, clip_s: float = 10.0) -> None:
    from shazamio import Shazam
    src = audio_of(mix)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(src)], capture_output=True,
                               text=True).stdout.strip())

    async def run() -> None:
        sh, res, t = Shazam(), [], 1.0
        while t < dur - 5:
            clip = Path(tempfile.gettempdir()) / f"shz_{int(t):05d}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-t", str(clip_s),
                            "-i", str(src), "-ac", "1", "-ar", "16000", str(clip)], check=True)
            try:
                r = await sh.recognize(str(clip))
                tr = r.get("track") or {}
                hit = dict(t=t, title=tr.get("title"), artist=tr.get("subtitle"),
                           offset=(r.get("matches") or [{}])[0].get("offset"))
            except Exception as e:                     # noqa: BLE001
                hit = dict(t=t, title=None, artist=None, err=str(e)[:80])
            res.append(hit)
            os.remove(clip)
            print(f"{t:7.0f}s {int(t // 60):2d}:{int(t % 60):02d}  "
                  f"{hit.get('title')!s:45.45s} | {hit.get('artist')}", flush=True)
            (out / "shazam_10s.json").write_text(json.dumps(res, indent=1))
            await asyncio.sleep(2.5)                   # paced politely
            t += step

    asyncio.run(run())


def lyrics(mix: str, out: Path) -> None:
    import torch
    import whisper
    torch.set_num_threads(8)
    m = whisper.load_model("small")
    r = m.transcribe(str(audio_of(mix)), fp16=False, language="en",
                     condition_on_previous_text=False, verbose=False)
    segs = [dict(start=s["start"], end=s["end"], text=s["text"].strip(),
                 nsp=s.get("no_speech_prob", 0)) for s in r["segments"]]
    (out / "lyrics_whisper.json").write_text(json.dumps(segs, indent=1))
    print(f"{len(segs)} lyric segments")


if __name__ == "__main__":
    mix, what = sys.argv[1], sys.argv[2]
    out = SHOW / ".generation" / mix
    out.mkdir(parents=True, exist_ok=True)
    {"beats": beats, "shazam": shazam, "lyrics": lyrics}[what](mix, out)
