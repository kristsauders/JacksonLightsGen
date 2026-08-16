---
name: xlights-environment-fixes
description: "The Intel-Mac dependency conflict that silently broke demucs stem separation and madmom beat detection, and how it was fixed"
metadata: 
  node_type: memory
  type: project
  originSessionId: e63440c6-1e60-4008-b9b0-199368b74605
  modified: 2026-08-09T14:35:38.020Z
---

This Mac is **Intel (x86_64)**, not Apple Silicon. That single fact drives several dependency constraints in `/Users/kristsauders/xlights-mcp-server/pyproject.toml` — don't "helpfully" loosen them without re-checking wheel availability first, since PyPI stops shipping Intel-Mac wheels for a lot of ML packages well before Apple Silicon ones.

**What was broken (as of the first Gooey attempt):** the venv had numpy 2.4.6, but torch 2.2.2 (the newest torch with an Intel-Mac wheel — 2.3+ dropped it) was built against the numpy 1.x ABI. Every `demucs.separate` call was silently failing (`RuntimeError: Numpy is not available`), so stem separation never actually ran despite the tool reporting success paths elsewhere. Separately, madmom 0.16.1 (unmaintained since 2018) doesn't import on Python >=3.10 (`from collections import MutableSequence`), so downbeat detection was silently falling back to librosa's naive "every 4th beat" heuristic.

**Fix applied** (already committed to `pyproject.toml`/`uv.lock` as of 2026-08-09, not yet git-committed — check `git status` before assuming it's landed):
- `numpy>=1.24.0,<2.0.0` (top-level dependency)
- `torch>=2.1,<2.3` and `torchaudio>=2.1,<2.3` in both the `separation` and `lyrics` extras (2.2.2 is the last Intel-Mac build)
- `demucs>=4.1.0` for the 6-stem model and nicer `demucs.api` — this needs `sphn`, a Rust extension with no prebuilt wheel for this platform, so **Rust is now installed via `brew install rust`** (Krists explicitly approved this). If `sphn` ever fails to build again, check `cargo`/`rustc` are still on PATH (`/usr/local/opt/rust/bin`) before assuming a real regression.
- `setuptools<81` in the `beats` extra — setuptools removed `pkg_resources` (which madmom imports) starting v81.
- `madmom` sourced from git main via `[tool.uv.sources]` (PyPI's 0.16.1 doesn't run on Python 3.12), plus `beats.py`'s `DBNDownBeatTrackingProcessor` call needed an explicit `beats_per_bar=[3, 4]` — the git version made that arg required.

**After the fix:** `demucs.separate` and `madmom` both import and run cleanly; a full analysis+stem-separation pass on a 4-minute song takes ~4 minutes on CPU (only ~20s if stems are already cached).

**If this breaks again:** don't spend time trying to upgrade torch further — there is no newer Intel-Mac wheel to upgrade to. The fix space is: downgrade numpy, or replace demucs with something that doesn't need torch at all.
