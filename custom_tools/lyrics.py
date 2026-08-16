"""Word-level lyric timing, so the show can hit specific words.

Whisper (already installed as `openai-whisper`, previously unused by this
pipeline) transcribes with word-level timestamps. That unlocks a class of
detail nothing else here can reach: lighting a specific WORD -- the title
hook, a repeated phrase, the line everyone sings along to. It's the kind of
moment an audience notices and remembers, and it cannot be derived from
onsets, energy or harmony.

Transcription is run on the DEMUCS VOCALS STEM rather than the full mix.
Whisper is markedly more accurate on an isolated vocal, and the stem is
already produced and cached by the analysis step, so this costs nothing extra.

Slow (~1-3 min on CPU for a full track) and deterministic, so results cache to
the song's `.generation` folder like the melody and harmony data.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# Words that carry no lyrical identity -- never worth building a moment
# around, however often they repeat.
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "so", "to", "of", "in", "on",
    "at", "for", "with", "is", "it", "its", "i", "im", "you", "your", "me",
    "my", "we", "he", "she", "they", "that", "this", "was", "be", "been",
    "do", "dont", "just", "like", "got", "get", "all", "up", "out", "no",
    "not", "yeah", "oh", "uh", "na", "la", "ooh", "hey",
    # Pronouns/auxiliaries that slipped through the first pass -- the initial
    # hook result on a real song was "her" x5, which is exactly the failure
    # this list exists to prevent.
    "her", "him", "his", "hers", "them", "their", "us", "our", "were", "are",
    "am", "been", "being", "had", "has", "have", "hes", "shes", "theyre",
    "youre", "its", "what", "when", "who", "how", "why", "where", "cause",
    "know", "gonna", "wanna", "gotta", "aint", "cant", "wont", "there",
    "then", "than", "now", "one", "ones", "some", "any", "every", "will",
    "would", "could", "should", "did", "does", "from", "into", "about",
    "baby", "mmm", "woah", "whoa",
}


def transcribe_words(vocals_path: Path, model_name: str = "base") -> list[dict]:
    """[{'word','start','end','prob'}] for every recognised word.

    `base` rather than a larger model on purpose: we need TIMING and rough
    identity, not a publishable transcript, and base is several times faster
    on CPU. Sung vocals are hard for any Whisper model, so treat the text as
    approximate and lean on repetition (below) rather than any single word.
    """
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(str(vocals_path), word_timestamps=True,
                              fp16=False, verbose=False)
    words: list[dict] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = str(w.get("word", "")).strip().lower()
            text = "".join(ch for ch in text if ch.isalnum() or ch == "'")
            if not text:
                continue
            words.append({
                "word": text,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "prob": float(w.get("probability", 0.0)),
            })
    return words


def find_hook(words: list[dict], min_repeats: int = 6,
              min_conf: float = 0.6) -> dict | None:
    """The song's most-repeated meaningful word -- a decent proxy for the hook.

    Deliberately simple and robust rather than clever: Whisper's transcription
    of SUNG vocals is unreliable word-to-word, so anything depending on an
    exact phrase match would be fragile. A word that recurs many times with
    reasonable confidence is very likely part of the refrain even if some
    individual instances are mis-transcribed.
    """
    counts = Counter(w["word"] for w in words
                     if w["prob"] >= min_conf
                     and w["word"] not in _STOP and len(w["word"]) > 2)
    if not counts:
        return None
    word, n = counts.most_common(1)[0]
    if n < min_repeats:
        return None
    hits = [w for w in words if w["word"] == word and w["prob"] >= min_conf]
    return {"word": word, "count": n,
            "times": [{"start": h["start"], "end": h["end"]} for h in hits]}


def vocal_phrases(vocals_path: Path, words: list[dict] | None = None,
                  min_phrase: float = 0.8, min_gap: float = 0.55) -> list[dict]:
    """Sung phrases, derived from the vocal stem's ENERGY rather than from
    Whisper's word gaps.

    Measured why: on a real track 90% of consecutive Whisper words reported a
    gap of exactly 0.00s -- it emits near-contiguous timings on sung material,
    so gap-based splitting produced a single 56-SECOND "phrase". The word
    TIMINGS are usable, but Whisper cannot tell us where the singing stops.
    The isolated vocals stem can, directly and reliably: threshold its RMS
    envelope and every silence is a real one.

    Words (if supplied) are attached to whichever phrase contains them, so
    callers still get text alongside musically-correct boundaries.
    """
    import librosa
    import numpy as np

    if not vocals_path.exists():
        return []
    y, sr = librosa.load(str(vocals_path), sr=22050, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    if not len(rms):
        return []
    # Relative threshold: vocal stems vary a lot in level between songs.
    thresh = max(float(np.percentile(rms, 60)) * 0.6, float(rms.max()) * 0.06)
    active = rms >= thresh

    phrases: list[dict] = []
    start = None
    for i, on in enumerate(active):
        t = float(times[i])
        if on and start is None:
            start = t
        elif not on and start is not None:
            if t - start >= min_phrase:
                phrases.append({"start": start, "end": t, "words": []})
            start = None
    if start is not None and float(times[-1]) - start >= min_phrase:
        phrases.append({"start": start, "end": float(times[-1]), "words": []})

    # Merge phrases separated by less than min_gap -- a breath is not a
    # phrase boundary.
    merged: list[dict] = []
    for p in phrases:
        if merged and p["start"] - merged[-1]["end"] < min_gap:
            merged[-1]["end"] = p["end"]
        else:
            merged.append(p)

    for w in (words or []):
        for p in merged:
            if p["start"] - 0.2 <= w["start"] <= p["end"] + 0.2:
                p["words"].append(w["word"])
                break
    return merged


def analyse_lyrics(vocals_path: Path, cache_path: Path | None = None) -> dict:
    """Cache ONLY the transcription (the slow, fixed part); always recompute
    the derived hook/phrases so their heuristics can be re-tuned without
    paying for Whisper again."""
    out: dict = {"words": [], "hook": None, "phrases": []}
    if not vocals_path.exists():
        return out

    words: list[dict] | None = None
    if cache_path and cache_path.exists():
        try:
            words = json.loads(cache_path.read_text()).get("words")
        except (OSError, ValueError):
            words = None
    if words is None:
        # Don't pay for Whisper on an instrumental. A demucs vocals stem for a
        # track with no singing is near-silent rather than absent, so the
        # file existing proves nothing.
        try:
            import librosa
            import numpy as np
            _y, _ = librosa.load(str(vocals_path), sr=22050, mono=True)
            if float(np.abs(_y).mean()) < 1e-4:
                if cache_path:
                    try:
                        cache_path.write_text(json.dumps({"words": []}))
                    except OSError:
                        pass
                out["phrases"] = vocal_phrases(vocals_path, [])
                return out
        except Exception:
            pass
        try:
            words = transcribe_words(vocals_path)
        except Exception as exc:
            out["error"] = str(exc)
            return out
        if cache_path:
            try:
                cache_path.write_text(json.dumps({"words": words}))
            except OSError:
                pass

    out["words"] = words
    out["hook"] = find_hook(words)
    out["phrases"] = vocal_phrases(vocals_path, words)
    return out


if __name__ == "__main__":
    import sys
    show = Path("/Volumes/XLIGHTS/XLights 2026")
    song = sys.argv[1] if len(sys.argv) > 1 else "The WhatsApp Shakespeare"
    res = analyse_lyrics(show / "stems" / song / "vocals.wav",
                         show / ".generation" / song / "lyrics.json")
    if res.get("error"):
        print("error:", res["error"])
    print(f"{song}: {len(res['words'])} words, {len(res['phrases'])} phrases")
    h = res.get("hook")
    print(f"  hook: {h['word']!r} x{h['count']}" if h else "  hook: none found")
    for p in res["phrases"][:6]:
        print(f"    {p['start']:6.1f}-{p['end']:6.1f}  {' '.join(p['words'][:10])}")
