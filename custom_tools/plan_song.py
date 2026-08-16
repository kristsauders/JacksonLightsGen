"""Everything needed to make CHOICES about a song, on one screen.

    python custom_tools/plan_song.py "ENIGMA"

The generator is a rule system: it reads the analysis and applies the same
rules to every song. That is why fixes generalise and why nothing is bespoke.
This tool exists for the other half of the job -- Krists asked directly
whether I am "weighing in rather than purely automating it," and the honest
answer was no. Planning requires seeing a song's shape at a glance, and until
now that meant six ad-hoc scripts per song.

What it prints, per section:
  * measured LOUDNESS rank (not the detector's label -- labels are wrong often
    enough to matter: on ENIGMA two "chorus" sections were a bass breakdown and
    a near-beatless passage, and following the labels would have produced the
    wrong show)
  * which VOICES are actually playing, and for each, whether its rhythm
    REPEATS (see Voice.pattern_score_in) -- a sparse-but-repeating figure is
    showable and a dense-but-unpatterned one is noise
  * the structural moments: drops, instrument dropouts (with fake/real split),
    risers

Then a short list of JUDGEMENT FLAGS -- places where the rules are likely to
get it wrong and a human call is needed. These are the things worth reading
before generating, not after.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from instrument_voices import detect_instrument_voices  # noqa: E402

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
VOICE_ORDER = ["kick", "snare", "hat", "bass", "chord", "vocal"]


def plan(song: str) -> dict:
    gen = SHOW_PATH / ".generation" / song
    a = json.loads((gen / "analysis.json").read_text())
    cues = json.loads((gen / "cues.json").read_text()) if (gen / "cues.json").exists() else {}
    beats = a["beats"]["beat_times"]
    tempo = a["beats"]["tempo"]
    secs = a["sections"]

    rms = np.array(a["spectrum"]["rms_energy"])
    rt = np.array(a["spectrum"]["rms_times"])
    peak = float(np.percentile(rms, 99)) + 1e-9
    loud = []
    for s in secs:
        m = (rt >= s["start_time"]) & (rt < s["end_time"])
        loud.append(float(rms[m].mean() / peak) if m.any() else 0.0)
    order = sorted(range(len(secs)), key=lambda i: loud[i])
    rank = {}
    for r, i in enumerate(order):
        rank[i] = r / max(1, len(order) - 1)

    voices = detect_instrument_voices(SHOW_PATH / "stems" / song, beats, tempo)

    rows = []
    for i, s in enumerate(secs):
        vs = {}
        for n in VOICE_ORDER:
            v = voices.get(n)
            if v is None:
                continue
            vs[n] = (v.density_in(s["start_time"], s["end_time"]),
                     v.pattern_score_in(s["start_time"], s["end_time"], beats))
        rows.append({"i": i, "s": s, "loud": loud[i], "rank": rank[i], "voices": vs})

    # --- judgement flags ---------------------------------------------------
    flags = []
    for r in rows:
        s, i, vs = r["s"], r["i"], r["voices"]
        dur = s["end_time"] - s["start_time"]
        kick_d = vs.get("kick", (0, 0))[0]
        # A "chorus" with no kick is not a chorus.
        if s["label"] == "chorus" and kick_d < 0.35 and dur > 8:
            flags.append(f"§{i} labelled CHORUS but kick density {kick_d:.2f} "
                         f"-- likely a breakdown; don't give it chorus activity")
        if s["label"] in ("verse", "bridge") and r["rank"] > 0.8:
            flags.append(f"§{i} labelled {s['label'].upper()} but is among the "
                         f"loudest sections -- likely a chorus")
        # Sections carried by one voice only.
        playing = [n for n, (d, _p) in vs.items() if d >= 0.6]
        if dur > 8 and len(playing) <= 2:
            pat = [f"{n}({vs[n][1]:.2f})" for n in playing if vs[n][1] >= 0.25]
            flags.append(f"§{i} sparse -- only {playing or 'nothing'} playing"
                         + (f"; repeating figures: {', '.join(pat)}" if pat else
                            "; NO repeating figure, risks looking empty"))
        if dur > 8 and not vs:
            flags.append(f"§{i} has NO detected voices at all ({dur:.0f}s)")

    # Climax: loudest run of adjacent sections.
    if rows:
        top = max(rows, key=lambda r: r["loud"])
        adj = [r for r in rows if abs(r["i"] - top["i"]) <= 1 and r["rank"] > 0.85]
        if len(adj) > 1:
            flags.append(f"climax spans §{adj[0]['i']}-§{adj[-1]['i']} "
                         f"({adj[0]['s']['start_time']:.0f}-{adj[-1]['s']['end_time']:.0f}s)"
                         f" -- treat as ONE peak, don't let the palette split it")

    return {"song": song, "a": a, "cues": cues, "rows": rows, "flags": flags,
            "tempo": tempo, "voices": voices}


def render(p: dict) -> None:
    a, rows, cues = p["a"], p["rows"], p["cues"]
    print(f"\n{'=' * 78}\n{p['song']}   {a['duration_seconds']:.0f}s   "
          f"{p['tempo']:.1f} BPM   {len(rows)} sections")
    kv = p["voices"].get("kick")
    if kv is not None:
        mode = "quarter" if kv.median_ipi_beats < 1.75 else "GROOVE (kick-led)"
        print(f"  kick median spacing {kv.median_ipi_beats:.1f} beats -> pulse mode: {mode}")
    hdr = " ".join(f"{n[:5]:>11}" for n in VOICE_ORDER)
    print(f"\n{'#':>3} {'time':>13} {'label':10} {'loud':>5} {hdr}")
    for r in rows:
        s = r["s"]
        cells = []
        for n in VOICE_ORDER:
            if n not in r["voices"]:
                cells.append(f"{'-':>11}")
                continue
            d, pat = r["voices"][n]
            mark = "*" if pat >= 0.25 and d < 1.0 else " "
            cells.append(f"{d:4.1f}/{pat:.2f}{mark}")
        print(f"{r['i']:>3} {s['start_time']:6.1f}-{s['end_time']:6.1f} "
              f"{s['label'][:10]:10} {r['loud']:5.2f} " + " ".join(cells))
    print("  (density / repetition;  * = sparse but repeating -> worth showing)")

    for key, label in [("instrument_dropouts", "DROPOUT"),
                       ("energy_drops", "energy drop"),
                       ("predrop_cutouts", "cutout")]:
        for d in cues.get(key, [])[:8]:
            t = d.get("drop_time", 0)
            extra = ""
            if d.get("false_drop"):
                extra = f"  (fake drop {d['false_drop']:.1f}s -> REAL {t:.1f}s)"
            if key == "instrument_dropouts":
                extra = (f"  hole {d['dropout_start']:.1f}-{d['dropout_end']:.1f}s" + extra)
            print(f"  {label:12} @ {t // 60:.0f}:{t % 60:04.1f}{extra}")

    if p["flags"]:
        print("\n  JUDGEMENT FLAGS:")
        for f in p["flags"]:
            print(f"    - {f}")


def main() -> None:
    songs = sys.argv[1:] or ["ENIGMA"]
    for s in songs:
        render(plan(s))


if __name__ == "__main__":
    main()
