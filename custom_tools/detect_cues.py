"""Detect musical 'moments' in a song worth a signature lighting effect:

  - melody/arpeggio runs (repeating cyclical note patterns -> cross-model chase)
  - risers/sweeps (rising pitch or rising broadband noise -> vertical/horizontal sweep)
  - pre-drop implosions (silence-then-slam -> meteor implosion cue)
  - cymbal/hihat accents (bright high-frequency drum transients -> detail sparkle)

Reads the cached analysis.json (beats/sections/spectrum) plus the separated
stem .wav files (both produced by run_analysis.py), and writes cues.json for
the generator to consume.

Usage: python3 detect_cues.py [SongName]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
SONG_NAME = sys.argv[1] if len(sys.argv) > 1 else "Gooey"
GEN_DIR = SHOW_PATH / ".generation" / SONG_NAME
ANALYSIS_PATH = GEN_DIR / "analysis.json"
STEMS_DIR = SHOW_PATH / "stems" / SONG_NAME
MP3_PATH = SHOW_PATH / f"{SONG_NAME}.mp3"
OUT_PATH = GEN_DIR / "cues.json"

SR = 22050
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# htdemucs output lags the original mix by ~552 samples @ 22050Hz (measured
# via cross-correlation, consistent across a track). Every timestamp derived
# from a stem needs this subtracted before it's compared to/placed on the
# original mix's timeline. Same constant as analyzer.py's DEMUCS_OUTPUT_DELAY_SEC.
STEM_DELAY_SEC = 552 / SR


def _delay_correct(times: list[float]) -> list[float]:
    return [max(0.0, t - STEM_DELAY_SEC) for t in times]


def _correct_run_times(runs: list[dict]) -> list[dict]:
    for r in runs:
        r["start_time"] = max(0.0, r["start_time"] - STEM_DELAY_SEC)
        r["end_time"] = max(0.0, r["end_time"] - STEM_DELAY_SEC)
        r["onset_times"] = _delay_correct(r["onset_times"])
    return runs


def _correct_riser_times(risers: list[dict]) -> list[dict]:
    for r in risers:
        r["start_time"] = max(0.0, r["start_time"] - STEM_DELAY_SEC)
        r["end_time"] = max(0.0, r["end_time"] - STEM_DELAY_SEC)
    return risers


def load_stem(name: str) -> tuple[np.ndarray, int] | None:
    path = STEMS_DIR / f"{name}.wav"
    if not path.exists():
        return None
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    return y, sr


# ---------------------------------------------------------------------------
# 1. Melody / arpeggio detection
# ---------------------------------------------------------------------------

def detect_arpeggios(y: np.ndarray, sr: int, min_period=2, max_period=8, min_repeats=3,
                       min_onset_gap=0.09, dominance_ratio=1.25, mismatch_tolerance=1):
    """Chroma-track a stem (works on real polyphonic/synth material, unlike
    monophonic pyin), quantize onsets to a dominant pitch class, and find
    short cyclically-repeating note patterns (arpeggios) with fuzzy matching
    (a bit of tolerance for octave/passing-tone noise in real audio)."""
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    # Merge onsets that are too close together to be separate arpeggio notes
    merged = []
    for t in onset_times:
        if not merged or t - merged[-1] >= min_onset_gap:
            merged.append(t)
    onset_times = np.array(merged)
    if len(onset_times) < min_period * min_repeats:
        return []

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_times = librosa.times_like(chroma, sr=sr)

    notes = []  # (time, pitch_class or None)
    for t in onset_times:
        mask = (chroma_times >= t) & (chroma_times < t + 0.15)
        window = chroma[:, mask]
        if window.shape[1] == 0:
            notes.append((float(t), None))
            continue
        mean_bins = window.mean(axis=1)
        order = np.argsort(mean_bins)[::-1]
        top, second = mean_bins[order[0]], mean_bins[order[1]]
        if second > 0 and top / (second + 1e-9) < dominance_ratio:
            notes.append((float(t), None))  # ambiguous / not clearly tonal
            continue
        notes.append((float(t), int(order[0])))

    def pc_match(a, b):
        return a is not None and b is not None and a == b

    def cycle_close_enough(window_pcs, template_pcs):
        mismatches = sum(0 if pc_match(a, b) else 1 for a, b in zip(window_pcs, template_pcs))
        return mismatches <= mismatch_tolerance

    runs = []
    n = len(notes)
    i = 0
    while i < n:
        best_run = None
        for period in range(min_period, max_period + 1):
            if i + period * min_repeats > n:
                continue
            pcs = [notes[i + k][1] for k in range(period)]
            if pcs.count(None) > mismatch_tolerance:
                continue
            if len(set(p for p in pcs if p is not None)) < 2:
                continue  # not really an arpeggio if it's one repeated note
            reps = 0
            j = i
            while j + period <= n:
                window_pcs = [notes[j + k][1] for k in range(period)]
                if cycle_close_enough(window_pcs, pcs):
                    reps += 1
                    j += period
                else:
                    break
            if reps >= min_repeats:
                length = reps * period
                if best_run is None or length > best_run[1]:
                    best_run = (period, length, pcs)
        if best_run:
            period, length, pcs = best_run
            start_idx = i
            end_idx = i + length
            clean_pcs = [p if p is not None else pcs[k % len(pcs)] for k, p in enumerate(pcs)]
            runs.append({
                "start_time": notes[start_idx][0],
                "end_time": notes[end_idx - 1][0] + 0.15,
                "period": period,
                "pitch_classes": clean_pcs,
                "note_names": [NOTE_NAMES[pc] for pc in clean_pcs],
                "onset_times": [notes[k][0] for k in range(start_idx, end_idx)],
            })
            i = end_idx
        else:
            i += 1

    return runs


# ---------------------------------------------------------------------------
# 2. Riser / sweep detection
# ---------------------------------------------------------------------------

def detect_risers(y: np.ndarray, sr: int, min_duration=0.7, max_duration=8.0,
                    rise_ratio=1.35, smooth_sec=0.15):
    """Find segments where spectral centroid rises steadily and substantially
    (relative rise, not an absolute Hz/sec threshold -- more robust across
    different mixes) -- classic 'riser' sound design. Classify tonal
    (narrowband) vs broadband (noise) by mean bandwidth."""
    hop = 512
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    times = librosa.times_like(centroid, sr=sr, hop_length=hop)

    win = max(3, int(sr / hop * smooth_sec))
    kernel = np.ones(win) / win
    centroid_s = np.convolve(centroid, kernel, mode="same")

    # Walk forward tracking local minima; a "riser" is a run from a local low
    # to a later point at least `rise_ratio`x higher, monotonic-ish (allow
    # small dips) and within a plausible duration window.
    risers = []
    n = len(centroid_s)
    i = 0
    while i < n - 1:
        base = centroid_s[i]
        base_i = i
        j = i + 1
        peak = base
        peak_i = i
        dip_budget = int(0.3 * sr / hop)  # allow brief dips without resetting
        dip_count = 0
        while j < n:
            if centroid_s[j] >= peak:
                peak = centroid_s[j]
                peak_i = j
                dip_count = 0
            else:
                dip_count += 1
                if dip_count > dip_budget and centroid_s[j] < base * 1.05:
                    break
            if peak >= base * rise_ratio and (times[j] - times[base_i]) <= max_duration:
                j += 1
                continue
            if (times[j] - times[base_i]) > max_duration:
                break
            j += 1
        dur = times[peak_i] - times[base_i]
        if peak >= base * rise_ratio and min_duration <= dur <= max_duration:
            seg_bw = float(np.mean(bandwidth[base_i:peak_i + 1]))
            risers.append({
                "start_time": float(times[base_i]),
                "end_time": float(times[peak_i]),
                "duration": float(dur),
                "centroid_rise_hz": float(peak - base),
                "rise_ratio": float(peak / (base + 1e-6)),
                "mean_bandwidth_hz": seg_bw,
                "kind": "broadband" if seg_bw > 2200 else "tonal",
            })
            i = peak_i + 1
        else:
            i += 1

    return risers


# ---------------------------------------------------------------------------
# 3. Pre-drop implosion detection (silence-then-slam)
# ---------------------------------------------------------------------------

def detect_predrop_cutouts(rms: np.ndarray, rms_times: np.ndarray,
                             quiet_frac=0.12, loud_frac=0.55,
                             min_quiet=0.1, max_quiet=2.5, lookahead=3.0,
                             merge_gap_s=3.0):
    """Find brief near-silence dips immediately followed by a big energy jump."""
    max_rms = float(np.max(rms)) + 1e-9
    norm = rms / max_rms

    events = []
    i = 0
    n = len(norm)
    while i < n:
        if norm[i] < quiet_frac:
            start_i = i
            while i < n and norm[i] < quiet_frac:
                i += 1
            end_i = i
            dur = rms_times[end_i - 1] - rms_times[start_i] if end_i > start_i else 0
            if min_quiet <= dur <= max_quiet:
                # Look ahead for the big jump -- the FIRST moment crossing
                # loud_frac, not the single loudest instant anywhere in the
                # window. A drop is characterized by the quiet->loud
                # transition; picking the loudest point in a 3s window can
                # land on a later, marginally-louder beat instead of the
                # actual drop onset, reported directly as "drop is several
                # beats late" -- confirmed against the specific case this
                # was tuned against.
                look_end_t = rms_times[end_i - 1] + lookahead
                look_mask = (rms_times >= rms_times[end_i - 1]) & (rms_times <= look_end_t)
                look_times = rms_times[look_mask]
                look_norm = norm[look_mask]
                crossing = np.argmax(look_norm >= loud_frac) if np.any(look_norm >= loud_frac) else -1
                if crossing >= 0 and look_norm[crossing]:
                    drop_time = float(look_times[crossing])
                    events.append({
                        "cutout_start": float(rms_times[start_i]),
                        "cutout_end": float(rms_times[end_i - 1]),
                        "drop_time": drop_time,
                        "drop_energy": float(look_norm[crossing]),
                    })
        else:
            i += 1

    # Merge near-duplicate candidates (e.g. several brief quiet blips in a
    # stuttering buildup, each independently satisfying the criteria above)
    # -- without this, multiple implosions can get scheduled for what's
    # really one musical drop, and their overlapping dark-hold/hit windows
    # compound into an extended blackout that reads as "the drop is late".
    events.sort(key=lambda ev: ev["drop_time"])
    merged: list[dict] = []
    for ev in events:
        if merged and ev["drop_time"] - merged[-1]["drop_time"] < merge_gap_s:
            if ev["drop_energy"] > merged[-1]["drop_energy"]:
                merged[-1] = ev
        else:
            merged.append(ev)
    return merged


def detect_energy_drops(rms: np.ndarray, rms_times: np.ndarray,
                        window_s=4.0, ratio=1.55, min_after=0.55,
                        merge_gap_s=8.0):
    """Find drops as a SUSTAINED STEP UP in level, with no cutout required.

    `detect_predrop_cutouts` above only fires when a near-silence (<12% of
    peak) precedes the jump. That is one specific EDM convention, and songs
    that build at normal loudness straight into the drop produce **zero**
    candidates -- which means zero drop staging, no blackout, no hit, nothing.
    Measured on World Is Mine, whose two biggest moments Krists identified by
    ear at 1:35 and 3:39: the level goes 37% -> 101% of peak across one second
    at 95.7s and 42% -> 73% at 218.8s, both unmistakable, and both invisible
    to a cutout-based detector because the build sits at 30-45% rather than
    going quiet. "The main drops are at 1:35 and 3:39 and they don't look like
    drops at all" was exactly this.

    The test here is deliberately about SUSTAIN, not transient: mean level over
    the next `window_s` seconds must be `ratio`x the mean over the previous
    `window_s`, and must itself be loud in absolute terms (`min_after` of
    peak). A single loud beat inside an already-loud passage fails both, so
    this doesn't fire on every crash cymbal.

    The returned `drop_time` is the FIRST frame crossing the post-jump average,
    not the loudest instant -- the same correction already learned for cutouts,
    where taking the peak landed the hit several beats late.
    """
    max_rms = float(np.max(rms)) + 1e-9
    norm = rms / max_rms
    if len(rms_times) < 4:
        return []
    dt = float(np.median(np.diff(rms_times)))
    w = max(2, int(window_s / dt))

    events = []
    for i in range(w, len(norm) - w):
        before = float(norm[i - w:i].mean())
        after = float(norm[i:i + w].mean())
        if before < 1e-6 or after < min_after:
            continue
        if after / before < ratio:
            continue
        # Walk back to the first frame of the rise: the earliest point in the
        # last half-window that is already above the post-jump mean. Without
        # this the reported time is wherever the sliding window happened to
        # cross, which can sit a beat or two into the drop.
        back = max(0, i - w // 2)
        seg = norm[back:i + 1]
        hits = np.nonzero(seg >= after)[0]
        onset_i = back + int(hits[0]) if len(hits) else i
        events.append({
            "drop_time": float(rms_times[onset_i]),
            "drop_energy": after,
            "lift": float(after / before),
            "tier": "major",
        })

    # A SECOND, SMALLER TIER. "Maybe we're not detecting some smaller drops
    # that dip only a little and aren't necessarily big drops but are still
    # important moments to accentuate." The thresholds above are deliberately
    # strict -- 1.55x over four seconds, landing above 55% of peak -- and they
    # find two moments in the whole of Gooey. A song has more turns than that.
    #
    # These are returned tagged rather than mixed in, so the generator can give
    # them a proportionate response: a breath before, a change of look on the
    # beat, but not the full blackout-and-hit staging a major drop gets.
    for i in range(w, len(norm) - w):
        before = float(norm[i - w:i].mean())
        after = float(norm[i:i + w].mean())
        if before < 1e-6 or after < 0.38:
            continue
        lift = after / before
        if not (1.22 <= lift < ratio):
            continue
        back = max(0, i - w // 2)
        seg = norm[back:i + 1]
        hits = np.nonzero(seg >= after)[0]
        onset_i = back + int(hits[0]) if len(hits) else i
        events.append({
            "drop_time": float(rms_times[onset_i]),
            "drop_energy": after,
            "lift": float(lift),
            "tier": "minor",
        })

    # One drop per musical moment. The sliding window fires on many consecutive
    # frames across the same rise, and a chorus can also step up again a few
    # bars in; an 8s merge keeps the strongest of each cluster.
    # Merge within each tier, and never let a minor sit near a major -- the
    # major IS that moment and a second marker beside it is the "other changes
    # happening right before or after that distract from the main moment".
    events.sort(key=lambda ev: (ev["drop_time"], ev["tier"] != "major"))
    merged: list[dict] = []
    for ev in events:
        near = next((m for m in merged
                     if abs(m["drop_time"] - ev["drop_time"]) < merge_gap_s), None)
        if near is not None:
            if near["tier"] == ev["tier"] and ev["lift"] > near["lift"]:
                near.update(ev)
            elif near["tier"] == "minor" and ev["tier"] == "major":
                near.update(ev)
            continue
        merged.append(dict(ev))
    merged.sort(key=lambda ev: ev["drop_time"])
    return merged


def detect_instrument_dropouts(y: np.ndarray, sr: int, min_dur=0.8, max_dur=9.0,
                                collapse_frac=0.28, recover_frac=0.55,
                                lookahead=10.0):
    """Stretches where the INSTRUMENTS drop out but the bass keeps going.

    Neither existing detector can see this. `detect_predrop_cutouts` needs the
    whole mix to go near-silent, and `detect_energy_drops` watches broadband
    level -- but in this gesture the low end continues at full strength, so
    overall RMS barely moves. Measured on ENIGMA, where Krists described it
    exactly: "From 2:48 to 2:53 every instrument drops out and it's just the
    bass, and then the big drop is at 2:53." Mid+high band energy collapses
    from ~0.40 to 0.02-0.06 for roughly two and a half seconds while the low
    band holds. Invisible to RMS; obvious once you split the bands.

    This is the single most powerful set-up gesture in electronic music -- the
    whole point is that the drop lands into a hole -- so missing it costs the
    biggest moment in the song its impact.

    Returns dicts with the dropout window and the recovery (drop) time.
    """
    hop = 512

    def band(lo, hi):
        sos = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
        return librosa.feature.rms(y=sosfiltfilt(sos, y), hop_length=hop)[0]

    mid = band(200, 2000)
    high = band(4000, min(10000, sr / 2 - 100))
    n = min(len(mid), len(high))
    body = (mid[:n] + high[:n]) / 2.0
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)
    # SMOOTH BEFORE THRESHOLDING. The kick bleeds into the 200-2000Hz band on
    # every hit, so the raw envelope pops above threshold once a beat and
    # shatters a genuine 2-second hole into 0.7s fragments -- measured on
    # ENIGMA, where the real dropout (168.5-170.6s) was being reported as a
    # single 0.7s run and then rejected. A ~350ms moving average bridges the
    # per-beat spikes while preserving a gap this long.
    win = max(3, int(round(0.35 * sr / hop)))
    body = np.convolve(body, np.ones(win) / win, mode="same")
    ref = float(np.percentile(body, 85)) + 1e-9
    norm = body / ref

    dt0 = float(np.median(np.diff(times))) if len(times) > 1 else 0.02
    events = []
    i = 0
    while i < n:
        if norm[i] < collapse_frac:
            j = i
            while j < n and norm[j] < collapse_frac:
                j += 1
            dur = times[min(j, n - 1)] - times[i]
            # The music has to have been PLAYING before the hole. A quiet
            # intro is mostly hole, so any dip inside it registers as a
            # dropout -- ENIGMA's intro produced one at 7.3-8.7s whose
            # "drop" was just the vocal entering at 10s, staged as a full
            # implosion. "When the vocals start at :11 it's treated like a
            # drop but it's really not." Require real body immediately
            # before, which is what makes a hole a hole.
            pre = norm[max(0, i - int(2.0 / max(dt0, 1e-6))):i]
            if len(pre) and float(pre.mean()) < 0.45:
                i = j
                continue
            if min_dur <= dur <= max_dur:
                # The drop is the first strong RETURN of the instruments.
                k_end = times[min(j, n - 1)] + lookahead
                m = (times >= times[min(j, n - 1)]) & (times <= k_end)
                if m.any():
                    seg, seg_t = norm[m], times[m]
                    over = np.nonzero(seg >= recover_frac)[0]
                    if len(over):
                        first_back = float(seg_t[over[0]])
                        # TWO-STAGE RETURN: the drums often come back first,
                        # over a still-quiet bass, and the REAL drop lands a
                        # bar or two later when everything else arrives.
                        # Krists described exactly this on ENIGMA: "at 2:51
                        # after the dropout the drums come in with a quiet bass
                        # but it's like a fake drop, it's a bar or two later at
                        # 2:53 that the real drop happens when all the
                        # instruments suddenly come back in."
                        #
                        # Measured there: the partial return plateaus at
                        # 0.52-0.68 for about three seconds, then steps to
                        # 1.09-1.47 -- roughly DOUBLE -- and stays. Staging the
                        # big moment on the first crossing spends it on the
                        # fake drop and leaves the real one unmarked.
                        #
                        # So: measure the plateau the first return settles
                        # into, then look for a sustained step well above it.
                        real_drop, false_drop = first_back, None
                        pw = (seg_t >= first_back) & (seg_t <= first_back + 2.5)
                        if pw.sum() >= 3:
                            plateau = float(np.median(seg[pw]))
                            look = (seg_t > first_back + 1.0)
                            if look.any() and plateau > 1e-6:
                                lt, lv = seg_t[look], seg[look]
                                # Sustained: the following ~1.2s must hold up,
                                # so a single loud transient can't trigger it.
                                dt_ = float(np.median(np.diff(times)))
                                hold = max(2, int(1.2 / dt_))
                                for k in range(len(lv) - hold):
                                    if lv[k:k + hold].mean() >= plateau * 1.5:
                                        if lt[k] - first_back >= 1.2:
                                            false_drop, real_drop = first_back, float(lt[k])
                                        break
                        ev = {
                            "dropout_start": float(times[i]),
                            "dropout_end": float(times[min(j, n - 1)]),
                            "drop_time": real_drop,
                            "depth": float(norm[i:j].mean()),
                        }
                        if false_drop is not None:
                            ev["false_drop"] = false_drop
                        events.append(ev)
            i = j
        else:
            i += 1
    # One per moment.
    merged = []
    for ev in events:
        if merged and ev["drop_time"] - merged[-1]["drop_time"] < 6.0:
            continue
        merged.append(ev)
    return merged


# ---------------------------------------------------------------------------
# 4. Cymbal / hihat accent detection (bright, brief, high-frequency drum hits)
# ---------------------------------------------------------------------------

def detect_fast_risers(y: np.ndarray, sr: int, min_dur=0.12, max_dur=0.9,
                        rise_ratio=1.6, min_abs_rise=900.0, min_gap=0.5):
    """Short, steep upward spectral sweeps -- the 'synth zips upward quickly'
    gesture, distinct from the long builds detect_risers() finds. Uses a finer
    hop and much lighter smoothing so sub-second events survive."""
    hop = 256
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    times = librosa.times_like(centroid, sr=sr, hop_length=hop)
    win = max(3, int(sr / hop * 0.03))
    c = np.convolve(centroid, np.ones(win) / win, mode="same")

    events = []
    n = len(c)
    i = 0
    while i < n - 1:
        base, base_i = c[i], i
        peak, peak_i = base, i
        j = i + 1
        while j < n and (times[j] - times[base_i]) <= max_dur:
            if c[j] > peak:
                peak, peak_i = c[j], j
            elif c[j] < base * 0.9:
                break
            j += 1
        dur = times[peak_i] - times[base_i]
        if (peak >= base * rise_ratio and (peak - base) >= min_abs_rise
                and min_dur <= dur <= max_dur):
            events.append({
                "start_time": float(times[base_i]),
                "end_time": float(times[peak_i]),
                "duration": float(dur),
                "rise_ratio": float(peak / (base + 1e-6)),
                "centroid_rise_hz": float(peak - base),
            })
            i = peak_i + 1
        else:
            i += 1

    # enforce a minimum spacing so one gesture isn't reported many times
    out = []
    for e in events:
        if not out or e["start_time"] - out[-1]["start_time"] >= min_gap:
            out.append(e)
        elif e["rise_ratio"] > out[-1]["rise_ratio"]:
            out[-1] = e
    return out


def detect_synth_stabs(y: np.ndarray, sr: int, centroid_jump=1.25,
                        amp_jump=1.3, min_gap=0.14):
    """Synth stabs where the filter opens: an onset that is simultaneously an
    amplitude attack AND a jump in brightness (spectral centroid). Those are
    the moments worth a fast synchronized fade-in on the lights."""
    hop = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=True, units="time")
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.times_like(centroid, sr=sr, hop_length=hop)

    stabs = []
    for t in onsets:
        pre = (times >= t - 0.12) & (times < t)
        post = (times >= t) & (times < t + 0.09)
        if not pre.any() or not post.any():
            continue
        c_pre, c_post = float(np.median(centroid[pre])), float(np.median(centroid[post]))
        r_pre, r_post = float(np.median(rms[pre])), float(np.median(rms[post]))
        if c_pre <= 0 or r_pre <= 0:
            continue
        if c_post >= c_pre * centroid_jump and r_post >= r_pre * amp_jump:
            stabs.append({
                "time": float(t),
                "brightness_ratio": c_post / c_pre,
                "amp_ratio": r_post / r_pre,
            })

    out = []
    for s in stabs:
        if not out or s["time"] - out[-1]["time"] >= min_gap:
            out.append(s)
        elif s["amp_ratio"] > out[-1]["amp_ratio"]:
            out[-1] = s
    return out


def detect_cymbal_accents(y: np.ndarray, sr: int, cutoff_hz=6000.0):
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    y_hp = sosfiltfilt(sos, y)
    onset_frames = librosa.onset.onset_detect(
        y=y_hp.astype(np.float32), sr=sr, backtrack=True, units="frames",
        delta=0.15,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return [float(t) for t in onset_times]


# ---------------------------------------------------------------------------


def veto_vocal_led(events: list, envs: dict, label: str) -> list:
    """Drop "drops" that are really just the VOCAL coming back in.

    "At :42 it cuts out and then does a 'drop' at :46 but it's just the vocal
    coming back in... the vocal does not constitute a drop."

    Both detectors work off a broadband envelope, so they cannot tell WHICH
    instrument refilled the hole. Measured at Gooey's false drop: the vocal
    returns 13.7x its in-hole level while the bass comes back at 0.39x -- the
    band is not back, the singer is. At ENIGMA's real drop the ordering is
    reversed (drums 32.5x, bass 2.2x, vocals only 3.2x), which is why the test
    is RELATIVE: reject when the vocal outruns the whole rhythm section.

    Deliberately a veto and not a score. An earlier attempt to boost drops by
    corroboration fixed one song and broke ENIGMA's double drop; a veto can
    only ever remove a moment that measurably isn't one.

    Guard: if this would leave a song with NO structural moments at all, the
    strongest candidate is kept. A song whose drops are genuinely vocal-led
    (a ballad) should read as understated, not as having no shape whatever.
    """
    if not events or len(envs) < 3:
        return events

    def _med(e, t, a, b):
        m = (t >= a) & (t < b)
        return float(np.median(e[m])) if m.any() else 0.0

    kept, vetoed = [], []
    for d in events:
        dt = d.get("drop_time")
        h0 = d.get("dropout_start", d.get("cutout_start"))
        h1 = d.get("dropout_end", d.get("cutout_end"))
        if dt is None or h0 is None:
            kept.append(d)
            continue
        r = {}
        for n, (e, t) in envs.items():
            hole = _med(e, t, h0, h1)
            r[n] = _med(e, t, dt, dt + 1.5) / max(hole, 1e-6)
        rhythm = max(r.get("drums", 0.0), r.get("bass", 0.0))
        if r.get("vocals", 0.0) > 2.0 * rhythm:
            d["_rhythm_ratio"] = rhythm
            vetoed.append(d)
        else:
            kept.append(d)
    if vetoed:
        print(f"  {label}: vetoed {len(vetoed)} vocal-led (not real drops): "
              + ", ".join(f"{d['drop_time']:.1f}s" for d in vetoed))
    return kept, vetoed


def main():
    analysis = json.loads(ANALYSIS_PATH.read_text())
    rms = np.array(analysis["spectrum"]["rms_energy"])
    rms_times = np.array(analysis["spectrum"]["rms_times"])

    print("Loading stems...", flush=True)
    other = load_stem("other")
    drums = load_stem("drums")
    bass = load_stem("bass")

    result = {"arpeggios": [], "risers": [], "predrop_cutouts": [], "cymbal_accents": []}

    if other is not None:
        y, sr = other
        print("Detecting arpeggios in 'other' stem...", flush=True)
        result["arpeggios"] = _correct_run_times(detect_arpeggios(y, sr))
        print(f"  found {len(result['arpeggios'])} arpeggio runs")

        print("Detecting risers in 'other' stem...", flush=True)
        result["risers"] = _correct_riser_times(detect_risers(y, sr))
        print(f"  found {len(result['risers'])} risers")
    else:
        print("No 'other' stem found -- falling back to full mix for arpeggios/risers", flush=True)
        y, sr = librosa.load(str(MP3_PATH), sr=SR, mono=True)
        result["arpeggios"] = detect_arpeggios(y, sr)
        result["risers"] = detect_risers(y, sr)

    if bass is not None:
        y, sr = bass
        print("Detecting bassline arpeggios...", flush=True)
        result["bass_arpeggios"] = _correct_run_times(detect_arpeggios(y, sr, min_repeats=4))
        print(f"  found {len(result['bass_arpeggios'])} bassline arpeggio runs")

    print("Detecting pre-drop cutouts from full-mix RMS...", flush=True)
    result["predrop_cutouts"] = detect_predrop_cutouts(rms, rms_times)
    print(f"  found {len(result['predrop_cutouts'])} cutouts")

    print("Detecting instrument dropouts (bass-only holes before a drop)...", flush=True)
    _mix_y, _mix_sr = librosa.load(str(MP3_PATH), sr=SR, mono=True)
    result["instrument_dropouts"] = detect_instrument_dropouts(_mix_y, _mix_sr)
    print(f"  found {len(result['instrument_dropouts'])} dropouts"
          + (": " + ", ".join(f"{d['dropout_start']:.1f}-{d['dropout_end']:.1f}s"
                              f"->drop {d['drop_time']:.1f}s"
                              for d in result["instrument_dropouts"])
             if result["instrument_dropouts"] else ""))

    # --- vocal-led veto, applied to BOTH hole detectors -------------------
    # Central rather than inside each detector: they find the same class of
    # moment from the same broadband envelope, and neither can see which
    # instrument came back. See veto_vocal_led().
    _stem_env = {}
    for _n in ("drums", "bass", "vocals"):
        _f = STEMS_DIR / f"{_n}.wav"
        if _f.exists():
            _y, _s = librosa.load(str(_f), sr=SR, mono=True)
            _e = librosa.feature.rms(y=_y, hop_length=512)[0]
            _stem_env[_n] = (_e, librosa.frames_to_time(np.arange(len(_e)),
                                                        sr=_s, hop_length=512))
    _all_vetoed = []
    for _key, _label in (("instrument_dropouts", "dropouts"),
                         ("predrop_cutouts", "cutouts")):
        _res = veto_vocal_led(result.get(_key, []), _stem_env, _label)
        if isinstance(_res, tuple):
            result[_key], _v = _res
            _all_vetoed.extend(_v)
    # A song whose moments are ALL vocal-led still needs some shape: put the
    # single strongest one back rather than leaving it with nothing.
    if _all_vetoed and not result.get("instrument_dropouts") \
            and not result.get("predrop_cutouts"):
        _best = max(_all_vetoed, key=lambda d: d.get("_rhythm_ratio", 0.0))
        _best.pop("_rhythm_ratio", None)
        _k = ("instrument_dropouts" if "dropout_start" in _best
              else "predrop_cutouts")
        result[_k] = [_best]
        print(f"  all moments were vocal-led; kept the strongest "
              f"({_best['drop_time']:.1f}s) so the song still has a shape")

    print("Detecting energy-step drops (no cutout required)...", flush=True)
    result["energy_drops"] = detect_energy_drops(rms, rms_times)
    print(f"  found {len(result['energy_drops'])} energy drops"
          + (": " + ", ".join(f"{d['drop_time']:.1f}s({d['lift']:.1f}x)"
                              for d in result["energy_drops"])
             if result["energy_drops"] else ""))

    if drums is not None:
        y, sr = drums
        print("Detecting cymbal/hihat accents in 'drums' stem...", flush=True)
        result["cymbal_accents"] = _delay_correct(detect_cymbal_accents(y, sr))
        print(f"  found {len(result['cymbal_accents'])} accents")

    print("Detecting risers in full mix (catches broadband FX risers stems split apart)...", flush=True)
    mix_y, mix_sr = librosa.load(str(MP3_PATH), sr=SR, mono=True)
    result["mix_risers"] = detect_risers(mix_y, mix_sr)  # no stem delay -- this IS the mix
    print(f"  found {len(result['mix_risers'])} risers in full mix")

    print("Detecting fast upward sweeps (short steep synth zips)...", flush=True)
    fast = detect_fast_risers(mix_y, mix_sr)
    if other is not None:
        fast += _correct_riser_times(detect_fast_risers(*other))
    fast.sort(key=lambda e: e["start_time"])
    deduped = []
    for e in fast:
        if not deduped or e["start_time"] - deduped[-1]["start_time"] >= 0.5:
            deduped.append(e)
        elif e["rise_ratio"] > deduped[-1]["rise_ratio"]:
            deduped[-1] = e
    result["fast_risers"] = deduped
    print(f"  found {len(deduped)} fast upward sweeps")

    if other is not None:
        print("Detecting synth stabs (filter-open transients in 'other' stem)...", flush=True)
        stabs = detect_synth_stabs(*other)
        for s in stabs:
            s["time"] = max(0.0, s["time"] - STEM_DELAY_SEC)
        result["synth_stabs"] = stabs
        print(f"  found {len(stabs)} synth stabs")

    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Wrote cues to {OUT_PATH}")


if __name__ == "__main__":
    main()
