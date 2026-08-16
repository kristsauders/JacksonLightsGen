"""Key and chord analysis, so colour can follow the music's harmony.

Until now palettes rotated on a fixed cycle with a no-repeat guard -- varied,
but arbitrary: the colour had no relationship to the song. This module supplies
the two harmonic facts that let colour become a musical decision:

* the song's KEY and its mode (major/minor), which sets the overall
  warm/cool character of the palette family;
* a CHORD timeline, whose changes are real musical boundaries -- often more
  meaningful than the energy-derived section boundaries the generator uses,
  and specifically a good moment to change a palette or swing a shader.

Uses madmom (already a dependency, installed from git for Python 3.12) rather
than adding anything new: `madmom.features.key.CNNKeyRecognitionProcessor` and
`madmom.features.chords.DeepChromaChordRecognitionProcessor`.

Both are slow (CNN inference over the whole track) and fully deterministic for
a given file, so results are cached to the song's `.generation` folder exactly
like the melody extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def detect_key(audio_path: Path) -> dict:
    """Return {'key': 'A minor', 'tonic': 'A', 'mode': 'minor', 'confidence': f}.

    madmom's key labels come as e.g. 'A minor' / 'C# major'.
    """
    from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

    proc = CNNKeyRecognitionProcessor()
    pred = proc(str(audio_path))
    label = key_prediction_to_label(pred)
    conf = float(np.max(pred)) if hasattr(pred, "__len__") else 0.0
    tonic, _, mode = label.partition(" ")
    return {"key": label, "tonic": tonic, "mode": mode or "major",
            "confidence": conf}


def detect_chords(audio_path: Path, min_dur: float = 0.6) -> list[dict]:
    """Chord segments as [{'start','end','label','root','quality'}].

    Segments shorter than `min_dur` are merged into their predecessor: a
    chord recogniser emits a lot of very short spurious changes, and a
    quarter-second "chord" is not a musical event anything should react to.
    """
    from madmom.features.chords import (CRFChordRecognitionProcessor,
                                         DeepChromaChordRecognitionProcessor)
    from madmom.audio.chroma import DeepChromaProcessor

    try:
        dcp = DeepChromaProcessor()
        decode = DeepChromaChordRecognitionProcessor()
        segs = decode(dcp(str(audio_path)))
    except Exception:
        return []

    out: list[dict] = []
    for s in segs:
        start, end, label = float(s[0]), float(s[1]), str(s[2])
        if out and (end - start) < min_dur and out[-1]["label"] == label:
            out[-1]["end"] = end
            continue
        if out and (end - start) < min_dur:
            out[-1]["end"] = end          # absorb the sliver, keep prior chord
            continue
        root, _, quality = label.partition(":")
        out.append({"start": start, "end": end, "label": label,
                    "root": root, "quality": quality or "maj"})
    return out


def chord_change_times(chords: list[dict], min_gap: float = 1.0) -> list[float]:
    """Times where the harmony actually moves, thinned so a fast turnaround
    doesn't produce a burst of change points."""
    times: list[float] = []
    for c in chords:
        if not times or c["start"] - times[-1] >= min_gap:
            times.append(c["start"])
    return times


def analyse_harmony(audio_path: Path, cache_path: Path | None = None) -> dict:
    """Key + chords + change times, cached to JSON."""
    if cache_path and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (OSError, ValueError):
            pass
    result: dict = {"key": None, "chords": [], "changes": []}
    try:
        result["key"] = detect_key(audio_path)
    except Exception as exc:
        result["key_error"] = str(exc)
    try:
        result["chords"] = detect_chords(audio_path)
        result["changes"] = chord_change_times(result["chords"])
    except Exception as exc:
        result["chord_error"] = str(exc)
    if cache_path:
        try:
            cache_path.write_text(json.dumps(result))
        except OSError:
            pass
    return result


# --- how harmony maps to colour -------------------------------------------
# Deliberately a SMALL, opinionated mapping rather than a lookup of all 24
# keys: the goal is that a song's colour family feels like it belongs to the
# song, not that every key gets a unique hue nobody could name. Mode does most
# of the work (minor -> cool/moody, major -> warm/bright), with the tonic
# nudging which member of that family leads.
MINOR_FAMILIES = [
    ["ice", "violet", "candy"],
    ["violet", "ice", "toxic"],
    ["ice", "candy", "violet"],
]
MAJOR_FAMILIES = [
    ["gold", "ember", "candy"],
    ["ember", "gold", "xmas"],
    ["gold", "candy", "ice"],
]
_TONICS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def palette_family_for_key(key_info: dict | None) -> list[str] | None:
    """Pick the song's palette family from its key, or None to fall back to
    the existing random-sample behaviour."""
    if not key_info or not key_info.get("tonic"):
        return None
    tonic = key_info["tonic"].replace("b", "#")
    idx = _TONICS.index(tonic) if tonic in _TONICS else 0
    fams = MINOR_FAMILIES if key_info.get("mode") == "minor" else MAJOR_FAMILIES
    return list(fams[idx % len(fams)])


if __name__ == "__main__":
    import sys
    show = Path("/Volumes/XLIGHTS/XLights 2026")
    song = sys.argv[1] if len(sys.argv) > 1 else "Gooey"
    res = analyse_harmony(show / f"{song}.mp3",
                          show / ".generation" / song / "harmony.json")
    print(f"{song}: key={res.get('key')}")
    ch = res.get("chords", [])
    print(f"  {len(ch)} chord segments, {len(res.get('changes', []))} change points")
    for c in ch[:8]:
        print(f"    {c['start']:6.1f}-{c['end']:6.1f}  {c['label']}")
    print(f"  palette family -> {palette_family_for_key(res.get('key'))}")
