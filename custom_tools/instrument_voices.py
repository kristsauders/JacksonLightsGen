"""Separate a song into individually-addressable INSTRUMENT VOICES.

Motivation (direct feedback, 2026-08-10): the generated sequences were
rhythmically busy but not rhythmically *legible* --

    "there are lots of other flashes and patterns that I think might be going
     off other stems but it can just seem chaotic... I want a person watching
     to see some prop doing a rhythm and have a second to hear that oh it's
     doing that to the hi hats or to the piano chords."

The generator's rhythmic layer had been driven by whole-stem onset lists
(`drums`, `other`, ...). A whole drum stem is not ONE instrument -- it's a
kick AND a snare AND hats AND fills, all collapsed into a single undifferentiated
onset list. Driving a prop from that means the prop is representing "the drums
in general", which no viewer can lock onto. Splitting the drums stem by
frequency band recovers the individual instruments, which is what makes a prop
readable as "that one is the hi-hats."

Two other things this module fixes, both named in the same feedback:

1. **Timing tightness.** Onsets are SNAPPED to the sixteenth-note grid when
   they're already close to it (rather than the older approach of dropping
   off-grid events entirely). Detected onset times carry a few ms of jitter
   from the detector itself; snapping removes that jitter, so a prop's rhythm
   lines up exactly with the beat rather than sitting a hair off it.

2. **Rate discipline per voice.** "Don't use anything too rapid unless it's
   like a drum fill or transition." Each voice is rate-limited to a musically
   sensible minimum gap, EXCEPT inside detected fills where rapid is the
   point.

Voices produced (whichever are actually present in the mix):
    kick, snare, hat   -- band-split from the drums stem
    bass               -- bass stem
    chord              -- `other` stem (piano/synth/strings/guitar)
    vocal              -- vocals stem
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

# (low_hz, high_hz) for the three drum-kit voices. A kick's fundamental lives
# below ~150Hz; a snare's body sits in the low-mids; hats/cymbals are almost
# entirely high-frequency. These are deliberately non-overlapping so the same
# hit doesn't register as two different instruments.
DRUM_BANDS = {
    "kick":  (20, 150),
    "snare": (180, 450),
    "hat":   (6000, 11000),
}

# Minimum gap between kept onsets, as a fraction of a beat. A hi-hat pattern
# genuinely runs at eighths or sixteenths and that's fine (it reads as texture,
# not as strobing, because it's steady); a snare firing every sixteenth is
# almost always fill/ghost-note noise rather than the part a listener follows.
MIN_GAP_BEATS = {
    "kick":  0.45,   # up to ~2 per beat
    "snare": 0.9,    # roughly on-beat or slower -- backbeat, not ghost notes
    # Eighths, not sixteenths. At a typical 120-130 BPM, sixteenths are
    # ~8-9Hz -- far past the stated "2-4 flashes per second, faster should be
    # a movement pattern" ceiling. Eighths (~4.3Hz) sit right at the edge,
    # and `split_flashes_and_bursts` downstream converts any run denser than
    # 3.5Hz into a single movement effect anyway, so hats end up reading as
    # shimmer/motion rather than discrete strobing either way.
    "hat":   0.45,
    "bass":  0.45,
    "chord": 0.9,    # chord stabs -- slow enough to read as harmony
    "vocal": 0.9,
}


@dataclass
class Voice:
    """One instrument's rhythmic identity across the whole song."""

    name: str
    onsets: list[float] = field(default_factory=list)
    strengths: list[float] = field(default_factory=list)
    # How closely this voice's onsets sit on the sixteenth grid, 0-1. A voice
    # that scores low is either not really rhythmic or was mis-detected --
    # worth deprioritising for a prop assignment, since a prop can only read
    # as "representing" something that is itself legible.
    grid_alignment: float = 0.0
    # Median inter-onset interval in beats -- the voice's characteristic pace.
    median_ipi_beats: float = 0.0

    def __len__(self) -> int:
        return len(self.onsets)

    def in_window(self, start_s: float, end_s: float) -> list[float]:
        return [t for t in self.onsets if start_s <= t < end_s]

    def pattern_score_in(self, start_s: float, end_s: float, beats: list[float],
                         bar_beats: int = 4) -> float:
        """How REPETITIVE this voice's rhythm is over a window, 0-1.

        "Why can't we extract the rhythm from the synths in the quieter parts?
        There are clear repetitive rhythms in both. We don't need to follow
        complex melodies or anything, but any repetitive rhythm on any voice is
        worth showing."

        Density is the wrong gate for that. A synth playing a tight three-note
        figure once per bar through a breakdown has LOW density and a perfectly
        legible rhythm; a busy but unpatterned part has high density and reads
        as noise. What makes a rhythm showable is that it REPEATS, so that is
        what gets measured: onsets are quantised onto a sixteenth grid within
        each bar, and the score is the mean agreement between every pair of
        bars. 1.0 means every bar is identical.
        """
        idx = [i for i, b in enumerate(beats) if start_s <= b < end_s]
        if len(idx) < bar_beats * 2:
            return 0.0
        bars: list[np.ndarray] = []
        for k in range(0, len(idx) - bar_beats, bar_beats):
            b0 = beats[idx[k]]
            b1 = beats[idx[k] + bar_beats]
            span = b1 - b0
            if span <= 0:
                continue
            slots = np.zeros(bar_beats * 4, dtype=bool)
            for t in self.onsets:
                if b0 <= t < b1:
                    slots[min(len(slots) - 1,
                              int((t - b0) / span * len(slots)))] = True
            if slots.any():
                bars.append(slots)
        if len(bars) < 2:
            return 0.0
        scores = []
        for i in range(len(bars)):
            for j in range(i + 1, len(bars)):
                a, b = bars[i], bars[j]
                union = (a | b).sum()
                if union:
                    scores.append((a & b).sum() / union)
        return float(np.mean(scores)) if scores else 0.0

    def density_in(self, start_s: float, end_s: float) -> float:
        """Onsets per second within a window -- used to decide whether this
        voice is actually playing during a given section."""
        span = max(1e-6, end_s - start_s)
        return len(self.in_window(start_s, end_s)) / span


def _band_envelope(y: np.ndarray, sr: int, lo: float, hi: float,
                   hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return np.zeros(S.shape[1]), librosa.times_like(S[0], sr=sr, hop_length=hop)
    env = S[band].sum(axis=0)
    env = librosa.util.normalize(env)
    return env, librosa.times_like(env, sr=sr, hop_length=hop)


def _snap_and_thin(onsets: np.ndarray, strengths: np.ndarray, grid: np.ndarray,
                   snap_tol_s: float, min_gap_s: float,
                   protect_windows: list[tuple[float, float]] | None = None
                   ) -> tuple[list[float], list[float]]:
    """Snap near-grid onsets onto the grid, drop far-off-grid ones, then
    enforce a minimum gap -- keeping the STRONGEST onset in each cluster
    rather than the first, so thinning preserves the accented hits a listener
    actually follows instead of whichever happened to come first.

    `protect_windows` (drum fills) bypass the min-gap rule entirely: rapid is
    exactly what a fill is, and Krists explicitly wants fills highlighted.
    """
    protect_windows = protect_windows or []

    def protected(t: float) -> bool:
        return any(a <= t <= b for a, b in protect_windows)

    kept_t: list[float] = []
    kept_s: list[float] = []
    if len(grid) == 0:
        return kept_t, kept_s

    # Snap / reject against the grid.
    snapped: list[tuple[float, float, bool]] = []
    for t, s in zip(onsets, strengths):
        idx = int(np.searchsorted(grid, t))
        cands = grid[max(0, idx - 1):idx + 2]
        if len(cands) == 0:
            continue
        nearest = float(cands[int(np.argmin(np.abs(cands - t)))])
        if abs(nearest - t) <= snap_tol_s:
            snapped.append((nearest, float(s), protected(t)))
        elif protected(t):
            # Inside a fill, keep the real (unsnapped) time -- fills are
            # deliberately loose/rolling and snapping them to the grid would
            # flatten the very thing that makes them read as a fill.
            snapped.append((float(t), float(s), True))

    snapped.sort(key=lambda x: x[0])

    # Min-gap thinning, strongest-wins within each cluster.
    cluster: list[tuple[float, float, bool]] = []
    for item in snapped:
        if cluster and (item[0] - cluster[0][0]) < min_gap_s and not item[2]:
            cluster.append(item)
            continue
        if cluster:
            best = max(cluster, key=lambda x: x[1])
            kept_t.append(best[0])
            kept_s.append(best[1])
        cluster = [item]
    if cluster:
        best = max(cluster, key=lambda x: x[1])
        kept_t.append(best[0])
        kept_s.append(best[1])

    # Fill onsets bypass thinning, so re-add any that got clustered away.
    for t, s, prot in snapped:
        if prot and t not in kept_t:
            kept_t.append(t)
            kept_s.append(s)
    order = np.argsort(kept_t)
    return [kept_t[i] for i in order], [kept_s[i] for i in order]


def detect_drum_fills(stem_dir: Path, beats: list[float], beat_ms: float,
                      min_onsets: int = 4) -> list[tuple[float, float]]:
    """Find drum fills: bar-length windows with a burst of SNARE/TOM-band
    activity well above that band's own baseline.

    Direct feedback: "Drum fills are actually really great to highlight, but
    other random snare beats can be chaotic if they're not the song's focal
    point." So fills need finding explicitly -- they're the one place rapid
    activity is wanted, and everywhere else rapid activity is the problem.

    Measured first, then designed: counting onsets across the WHOLE drums
    stem does not work on modern produced tracks. On both test songs the
    per-bar total-onset count was nearly flat (Gooey median 8, max 13 -- only
    1.6x, never reaching any sensible "this is a fill" ratio), because a
    steady hi-hat pattern dominates the count and drowns out the thing that
    actually changes. Restricting to the 180-450Hz snare/tom band -- where
    fills actually live, and where hats contribute almost nothing -- gives
    real separation.

    Validated against musical structure rather than just eyeballed: on
    `12 to 12` this fires at 17.3s / 32.9s / 48.5s, spaced 15.6s apart, which
    at that song's 123 BPM is exactly 8 bars -- i.e. it independently
    rediscovered the song's phrase length, which is where fills belong.
    It also correctly returns ZERO for Gooey (programmed bass music with a
    genuinely constant drum pattern) rather than inventing fills to find.

    Returns (start_s, end_s) windows, merged where adjacent.
    """
    drums = stem_dir / "drums.wav"
    if not drums.exists() or len(beats) < 8:
        return []
    y, sr = librosa.load(str(drums), sr=22050, mono=True)
    lo, hi = DRUM_BANDS["snare"]
    env, _ = _band_envelope(y, sr, lo, hi)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                         hop_length=256, backtrack=False)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=256)
    if len(times) < min_onsets * 2:
        return []

    bar_s = (beat_ms / 1000.0) * 4.0
    song_end = float(beats[-1])
    windows: list[tuple[float, float, int]] = []
    t = float(beats[0])
    while t + bar_s <= song_end:
        n = int(((times >= t) & (times < t + bar_s)).sum())
        windows.append((t, t + bar_s, n))
        t += bar_s / 2.0
    if not windows:
        return []

    counts = np.array([w[2] for w in windows], dtype=float)
    active = counts[counts > 0]
    if len(active) == 0:
        return []
    baseline = float(np.median(active))
    p90 = float(np.percentile(counts, 90))
    # Require BOTH a high percentile within this song AND a real multiple of
    # the song's own baseline -- percentile alone would always return ~10% of
    # windows even for a song with no fills at all (Gooey), which is exactly
    # the failure mode of "find the top N%" thresholds on uniform data.
    threshold = max(p90, baseline * 1.8, float(min_onsets))

    fills: list[tuple[float, float]] = []
    for a, b, n in windows:
        if n >= threshold:
            if fills and a <= fills[-1][1]:
                fills[-1] = (fills[-1][0], max(fills[-1][1], b))
            else:
                fills.append((a, b))
    return fills


def detect_instrument_voices(stem_dir: Path, beats: list[float], beat_ms: float,
                             fills: list[tuple[float, float]] | None = None
                             ) -> dict[str, Voice]:
    """Build the per-instrument voice table for a song.

    `fills` (from detect_drum_fills) are passed through so rapid onsets inside
    a fill survive the per-voice rate limiting that everything else gets.
    """
    voices: dict[str, Voice] = {}
    if len(beats) < 8:
        return voices

    beat_s = beat_ms / 1000.0
    sixteenth = beat_s / 4.0
    grid = np.arange(float(beats[0]), float(beats[-1]) + sixteenth, sixteenth)
    # Snap tolerance: a bit under half a sixteenth, so an onset can only ever
    # snap to the grid point it's genuinely nearest to.
    snap_tol = sixteenth * 0.45

    def finish(name: str, onsets: list[float], strengths: list[float]) -> None:
        if len(onsets) < 6:
            return
        arr = np.asarray(onsets)
        # grid_alignment measured BEFORE snapping would be ideal, but post-snap
        # everything kept is on-grid by construction; instead score it as the
        # share of raw detections that were close enough to keep at all.
        ipis = np.diff(arr) / beat_s if len(arr) > 1 else np.array([0.0])
        v = Voice(name=name, onsets=list(arr), strengths=list(strengths))
        v.median_ipi_beats = float(np.median(ipis)) if len(ipis) else 0.0
        # This was declared and documented but never actually assigned, so it
        # read 0.00 for every voice in every song. Nothing consumed it, so it
        # degraded nothing -- but a field that always reads zero is worse than
        # no field, because it looks like a measurement. Scored here as the
        # share of onsets sitting within a sixteenth of the BEAT grid, which is
        # what "is this voice legible as rhythm" actually means.
        if len(arr):
            d = np.abs(arr[:, None] - np.asarray(beats)[None, :]).min(axis=1)
            v.grid_alignment = float((d <= beat_s / 4.0).mean())
        voices[name] = v

    # --- band-split drum voices ---
    drums_path = stem_dir / "drums.wav"
    if drums_path.exists():
        y, sr = librosa.load(str(drums_path), sr=22050, mono=True)
        for vname, (lo, hi) in DRUM_BANDS.items():
            env, _times = _band_envelope(y, sr, lo, hi)
            frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                                 hop_length=256, backtrack=False)
            if len(frames) == 0:
                continue
            t_raw = librosa.frames_to_time(frames, sr=sr, hop_length=256)
            s_raw = env[np.clip(frames, 0, len(env) - 1)]
            keep_t, keep_s = _snap_and_thin(
                t_raw, s_raw, grid, snap_tol,
                MIN_GAP_BEATS[vname] * beat_s, fills)
            finish(vname, keep_t, keep_s)

    # --- melodic / harmonic stem voices ---
    for vname, stem_file in (("bass", "bass.wav"), ("chord", "other.wav"),
                             ("vocal", "vocals.wav")):
        p = stem_dir / stem_file
        if not p.exists():
            continue
        y, sr = librosa.load(str(p), sr=22050, mono=True)
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                             hop_length=256, backtrack=False)
        if len(frames) == 0:
            continue
        t_raw = librosa.frames_to_time(frames, sr=sr, hop_length=256)
        s_raw = env[np.clip(frames, 0, len(env) - 1)]
        # Melodic stems get NO fill protection -- a fill is a drum event, and
        # letting a dense run of pitched onsets through unthinned is exactly
        # the "chaotic" texture this module exists to prevent.
        keep_t, keep_s = _snap_and_thin(
            t_raw, s_raw, grid, snap_tol, MIN_GAP_BEATS[vname] * beat_s, None)
        finish(vname, keep_t, keep_s)

    return voices


def extract_melody_notes(stem_dir: Path, sections, beat_ms: float,
                         min_conf: float = 0.72, max_notes: int = 200):
    """Pull a clear monophonic melody line out of the VOCALS stem as discrete
    pitched notes.

    Direct request: "I want to try extracting some melodies too when they're
    clear, and try showing them on something I have multiple of, so like if
    there are 8 orbs they can represent 8 notes."

    Uses the vocals stem specifically because it is the one reliably
    MONOPHONIC source in a demucs split -- an earlier note in this project
    recorded that pyin fails on polyphonic/synth material, which is true of
    the `other` stem (chords) but not of an isolated vocal line.
    Confidence-gated so unvoiced/breathy frames don't invent notes.

    Returns a list of {start, end, midi, degree} where `degree` is the note's
    rank within the song's own pitch range (0 = lowest used, N-1 = highest) --
    which is what lets a caller map "note height" onto "physical position
    across N props" without needing to know the key.
    """
    vocals = stem_dir / "vocals.wav"
    if not vocals.exists():
        return []
    y, sr = librosa.load(str(vocals), sr=22050, mono=True)
    if float(np.abs(y).mean()) < 1e-4:
        return []          # effectively an instrumental -- no vocal to track

    f0, voiced, voiced_prob = librosa.pyin(
        y, sr=sr, fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C6")), frame_length=2048)
    times = librosa.times_like(f0, sr=sr)

    ok = voiced & (voiced_prob >= min_conf) & np.isfinite(f0)
    if ok.sum() < 20:
        return []

    midi = np.full(len(f0), np.nan)
    midi[ok] = librosa.hz_to_midi(f0[ok])
    # Round to semitones and group consecutive equal-pitch frames into notes.
    semis = np.round(midi)

    notes = []
    i = 0
    min_len_s = (beat_ms / 1000.0) * 0.22
    while i < len(semis):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(semis) and ok[j + 1] and semis[j + 1] == semis[i]:
            j += 1
        start, end = float(times[i]), float(times[min(j, len(times) - 1)])
        if end - start >= min_len_s:
            notes.append({"start": start, "end": end, "midi": float(semis[i])})
        i = j + 1

    if not notes:
        return []
    if len(notes) > max_notes:
        # Keep the longest notes -- the melodic skeleton, not every passing tone.
        notes = sorted(notes, key=lambda n: -(n["end"] - n["start"]))[:max_notes]
        notes.sort(key=lambda n: n["start"])

    pitches = sorted({n["midi"] for n in notes})
    rank = {p: i for i, p in enumerate(pitches)}
    for n in notes:
        n["degree"] = rank[n["midi"]]
        n["n_degrees"] = len(pitches)
    return notes


if __name__ == "__main__":
    import json
    import sys

    song = sys.argv[1] if len(sys.argv) > 1 else "Gooey"
    show = Path("/Volumes/XLIGHTS/XLights 2026")
    a = json.loads((show / ".generation" / song / "analysis.json").read_text())
    beats = a["beats"]["beat_times"]
    beat_ms = 60000.0 / max(a["beats"]["tempo"], 60.0)
    stem_dir = show / "stems" / song

    fills = detect_drum_fills(stem_dir, beats, beat_ms)
    print(f"{song}: {len(fills)} drum fills")
    voices = detect_instrument_voices(stem_dir, beats, beat_ms, fills)
    for name, v in voices.items():
        print(f"  {name:6s} {len(v):5d} onsets  median IPI {v.median_ipi_beats:.2f} beats")
    mel = extract_melody_notes(stem_dir, a["sections"], beat_ms)
    print(f"  melody: {len(mel)} notes"
          + (f" spanning {mel[0]['n_degrees']} distinct pitches" if mel else ""))
