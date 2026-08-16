"""Run full audio analysis (beats, structure, spectrum, stem separation +
stem onsets/energy) on a song's mp3 and cache the result to disk as JSON.

This bypasses the MCP tool layer (which chokes on large outputs) and lets us
reuse the expensive analysis (demucs stem separation especially) across
multiple script runs while iterating on the generator.

Usage: python3 run_analysis.py [SongName]
  SongName must match a "<SongName>.mp3" in the show folder. Output and
  demucs stems are cached under the show folder, so re-runs after the first
  are fast even without re-doing stem separation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/src")
from xlights_mcp.audio.analyzer import full_analysis  # noqa: E402
from xlights_mcp.config import AudioConfig  # noqa: E402

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
SONG_NAME = sys.argv[1] if len(sys.argv) > 1 else "Gooey"
MP3_PATH = SHOW_PATH / f"{SONG_NAME}.mp3"
OUT_DIR = SHOW_PATH / ".generation" / SONG_NAME
OUT_PATH = OUT_DIR / "analysis.json"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Starting full analysis of {MP3_PATH}...", flush=True)
    analysis = full_analysis(MP3_PATH, AudioConfig())
    elapsed = time.time() - t0
    print(f"Analysis complete in {elapsed:.1f}s", flush=True)
    print(f"  duration={analysis.duration_seconds:.1f}s tempo={analysis.beats.tempo:.1f} "
          f"beats={len(analysis.beats.beat_times)} sections={len(analysis.sections)} "
          f"stems_available={analysis.stem_analysis.available}", flush=True)

    OUT_PATH.write_text(analysis.model_dump_json(indent=2))
    print(f"Wrote analysis to {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
