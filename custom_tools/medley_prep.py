"""Prepare each song of a medley for generation.

    python custom_tools/medley_prep.py "<mix name>" beats     # after analysis
    python custom_tools/medley_prep.py "<mix name>" design    # write medley.json files

beats -- replace every song's beat grid with the WHOLE-MIX grid. The per-song
analysis takes beats from librosa's beat_track (one global tempo) and downbeats
from madmom, independently: on a slice of a DJ mix librosa read Bluey at
172.3 BPM (it is 127.7), and across a tempo ramp a single tempo is simply wrong.
The whole-mix pass (mix_beats.json) is one madmom joint beat+downbeat run, so
it follows ramps and keeps every downbeat on a beat by construction. Each song
gets the slice of it inside its padded audio window, shifted to song time.
The original is kept once as analysis.librosa_beats.json.

design -- write each song's medley.json from custom_tools/medleys/<mix>.py:
its keep window, whether it is interior (no end-of-song behaviour), and its
design (sections, drops, palettes, shader pool, caps, reveal, outro). Design
times are written in MIX time, which is how a person reads them off the track,
and converted to song time here.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
ROOT = Path(__file__).resolve().parent


def beat_tempo(b: np.ndarray) -> float:
    """The song's main tempo: the MEAN of the gaps near the median gap.

    Not 60/median: madmom reports on a 10 ms frame, so the median gap is
    quantized (480 ms read for a ~486 ms beat on 12 to 12 -- 125.0 BPM for a
    ~123.8 BPM song). Not a line fit either: Be With You's slow organ intro
    pulled that to 113.5 for a 127.7 BPM song. Averaging only the gaps within
    8% of the median keeps the median's robustness and loses the quantization.
    """
    d = np.diff(b)
    m = float(np.median(d))
    sel = d[np.abs(d - m) < 0.08 * m]
    return float(round(60.0 / float(sel.mean() if sel.size else m), 2))


def load_plan(mix: str) -> tuple[Path, list[dict]]:
    G = SHOW / ".generation" / mix
    return G, json.loads((G / "segments.json").read_text())


def beats(mix: str) -> None:
    G, plan = load_plan(mix)
    mb = json.loads((G / "mix_beats.json").read_text())
    bt = np.array(mb["beats"]); db = np.array(mb["downbeats"])
    for p in plan:
        aj = SHOW / ".generation" / p["song"] / "analysis.json"
        if not aj.exists():
            print(f"  {p['name']:34s} not analysed yet"); continue
        raw = json.loads(aj.read_text())
        bak = aj.with_name("analysis.librosa_beats.json")
        if not bak.exists():
            bak.write_text(aj.read_text())
        a0, a1 = p["audio_start"], p["audio_end"]
        b = (bt[(bt >= a0) & (bt < a1)] - a0).round(4)
        d = (db[(db >= a0) & (db < a1)] - a0).round(4)
        old = raw["beats"].get("tempo")
        raw["beats"]["beat_times"] = b.tolist()
        raw["beats"]["downbeat_times"] = d.tolist()
        raw["beats"]["tempo"] = beat_tempo(b) if len(b) > 4 else old
        aj.write_text(json.dumps(raw))
        print(f"  {p['name']:34s} {len(b):4d} beats {len(d):3d} downbeats  "
              f"tempo {old:6.1f} -> {raw['beats']['tempo']:6.1f}")


def design(mix: str) -> None:
    G, plan = load_plan(mix)
    slug = re.sub(r"[^a-z0-9]+", "_", mix.lower()).strip("_")
    spec = importlib.util.spec_from_file_location("medley_design", ROOT / "medleys" / f"{slug}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    D = mod.DESIGN
    last = plan[-1]["name"]
    for p in plan:
        d = dict(D.get(p["name"], {}))
        off = p["audio_start"]
        conv = lambda x: round(float(x) - off, 3)            # mix time -> song time
        out = {"keep": [conv(p["start"]), conv(p["end"])],
               "interior": p["name"] != last}
        if "sections" in d:
            out["sections"] = [[conv(t), lab] for t, lab in d["sections"]]
            # The slice starts with padding from the previous song, which the
            # merge trims. Give it a section of its own: the generator expects
            # sections to start at 0, and keeping the song's real start as a
            # separate boundary means it still gets a look change.
            if out["sections"] and out["sections"][0][0] > 0.5:
                out["sections"].insert(0, [0.0, "transition"])
        if "drops" in d:
            out["drops"] = [conv(t) for t in d["drops"]]
        if "climax" in d:
            # A climax is named by its section's start (so moving a boundary
            # moves it too) and placed a second inside: the generator snaps
            # boundaries to beats, and a marker exactly on one can fall into
            # the section before -- on Be With You the BUILD then took 100%
            # and the reveal chorus 85%.
            starts = [t for t, _ in d.get("sections", [])]
            out["climax"] = [conv(t + 1.0 if any(abs(t - s) < 0.05 for s in starts) else t)
                             for t in d["climax"]]
        if "ambient" in d:
            out["ambient"] = [[conv(a), conv(b)] for a, b in d["ambient"]]
        if "section_look" in d:
            out["section_look"] = {str(conv(t)): v for t, v in d["section_look"].items()}
        if "caps" in d:
            out["caps"] = [[conv(a), conv(b), lo, hi] for a, b, lo, hi in d["caps"]]
        if "reveal" in d:
            r = d["reveal"]
            out["reveal"] = {"hits": [conv(x) for x in r["hits"]], "drop": conv(r["drop"]),
                             "colours": r.get("colours")}
        if "outro" in d:
            o = d["outro"]
            out["outro"] = {"no_flash_from": conv(o["no_flash_from"]),
                            "fade": [conv(x) for x in o["fade"]]}
        if "scenes" in d:
            # Every time-valued field of a scene, converted the same way.
            def _sc(v, key):
                if key in ("window", "build", "fade", "hits", "chords", "strokes"):
                    return [conv(x) for x in v]
                if key == "spans":
                    return [[conv(x) for x in sp] for sp in v]
                if key == "swells":
                    return [[conv(x) for x in sw] for sw in v]
                if key in ("drop", "end", "peak", "frame_from", "roof_from",
                           "arches_from", "build2", "rise_from", "still_from"):
                    return conv(v)
                return v
            out["scenes"] = [{k: _sc(v, k) for k, v in sc.items()} for sc in d["scenes"]]
        for k in ("palettes", "palette_family", "shader_pool", "groove", "pulse_mode"):
            if k in d:
                out[k] = d[k]
        gd = SHOW / ".generation" / p["song"]
        gd.mkdir(parents=True, exist_ok=True)
        (gd / "medley.json").write_text(json.dumps(out, indent=1))
        print(f"  {p['name']:34s} keep {out['keep'][0]:7.2f}-{out['keep'][1]:7.2f}  "
              f"{len(out.get('sections', []))} sections, {len(out.get('drops', []))} drops"
              f"{', REVEAL' if 'reveal' in out else ''}{', OUTRO' if 'outro' in out else ''}"
              f"{', caps' if 'caps' in out else ''}")


if __name__ == "__main__":
    {"beats": beats, "design": design}[sys.argv[2]](sys.argv[1])
