"""Stitch per-song sequences into ONE sequence on a continuous DJ mix.

    python custom_tools/merge_medley.py "<mix name>" --out "<show>/<mix name> (vN).xsq"

Why this exists: the generator assumes one tempo per song -- dozens of passes
use a single median beat length -- so a half-hour mix whose songs blend with
tempo ramps cannot be generated as one "song". Instead every song is cut out
with padding into its neighbours, sequenced on its own terms, and stitched back
here. `.generation/<mix>/segments.json` (written by the segmenting step) holds,
per song: the keep window in MIX time (start, end), the padded audio window it
was generated from (audio_start), and the generated `xsq`.

Merging rules, each for a measured reason:

* Each segment keeps only effects inside its own [start, end) window, clipped at
  the edges. Windows are disjoint and contiguous, so no two segments ever paint
  the same moment -- which is what makes merging layers BY POSITION safe even
  though the writer compacts empty layers differently per segment.
* Times are shifted by audio_start (segment time 0 = audio_start in the mix) and
  rounded to the 25 ms frame.
* The ColorPalettes and EffectDB tables are positional lookups (`palette="N"`,
  `ref="N"`), so both are rebuilt globally with de-duplication and every
  reference is remapped.
* Three timing tracks are added so 29 minutes are navigable in xLights: Songs,
  Sections (each song's planned sections, in mix time) and Bars (from the
  whole-mix madmom grid, which follows tempo ramps).
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
FRAME = 25


def snap(ms: float) -> int:
    return int(round(ms / FRAME)) * FRAME


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mix")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = SHOW / ".generation" / a.mix
    plan = json.loads((G / "segments.json").read_text())
    beats = json.loads((G / "mix_beats.json").read_text())
    duration = max(p["end"] for p in plan)

    template = ET.parse(plan[0]["xsq"])
    root = template.getroot()

    palettes: list[str] = []
    pal_ix: dict[str, int] = {}
    effdb: list[str] = []
    eff_ix: dict[str, int] = {}
    elements: "OrderedDict[str, dict]" = OrderedDict()
    display: "OrderedDict[str, ET.Element]" = OrderedDict()
    stats = []

    def gpal(s: str) -> int:
        if s not in pal_ix:
            pal_ix[s] = len(palettes); palettes.append(s)
        return pal_ix[s]

    def geff(s: str) -> int:
        if s not in eff_ix:
            eff_ix[s] = len(effdb); effdb.append(s)
        return eff_ix[s]

    for p in plan:
        tree = ET.parse(p["xsq"]); r = tree.getroot()
        lp = [(c.text or "") for c in r.find("ColorPalettes")]
        le = [(c.text or "") for c in r.find("EffectDB")]
        off = p["audio_start"] * 1000.0
        w0, w1 = snap(p["start"] * 1000.0), snap(p["end"] * 1000.0)
        kept = clipped = dropped = 0
        for de in r.find("DisplayElements"):
            if de.get("type") == "model" and de.get("name") not in display:
                display[de.get("name")] = de
        for el in r.find("ElementEffects"):
            if el.get("type") != "model":
                continue
            rec = elements.setdefault(el.get("name"), {"attrib": dict(el.attrib), "layers": []})
            for li, layer in enumerate(el.findall("EffectLayer")):
                while len(rec["layers"]) <= li:
                    rec["layers"].append([])
                for ef in layer.findall("Effect"):
                    s = snap(int(ef.get("startTime")) + off)
                    e = snap(int(ef.get("endTime")) + off)
                    if e <= w0 or s >= w1:
                        dropped += 1
                        continue
                    if s < w0 or e > w1:
                        clipped += 1
                    s, e = max(s, w0), min(e, w1)
                    if e - s < FRAME:
                        dropped += 1
                        continue
                    at = dict(ef.attrib)
                    at["startTime"], at["endTime"] = str(s), str(e)
                    if "ref" in at:
                        at["ref"] = str(geff(le[int(at["ref"])]))
                    if "palette" in at:
                        at["palette"] = str(gpal(lp[int(at["palette"])]))
                    rec["layers"][li].append(at)
                    kept += 1
        stats.append((p["name"], kept, clipped, dropped))

    # ---- timing tracks ---------------------------------------------------
    def timing(name: str, marks: list[tuple[int, int, str]]) -> ET.Element:
        el = ET.Element("Element", {"type": "timing", "name": name})
        lay = ET.SubElement(el, "EffectLayer")
        for s, e, lab in marks:
            if e - s >= FRAME:
                ET.SubElement(lay, "Effect", {"label": lab, "startTime": str(s), "endTime": str(e)})
        return el

    songs = [(snap(p["start"] * 1000), snap(p["end"] * 1000),
              f"{p['name'][3:]} - {p['artist']}") for p in plan]
    secs = []
    for p in plan:
        an = json.loads((SHOW / ".generation" / p["song"] / "analysis.json").read_text())
        for s in an["sections"]:
            ms0 = snap((s["start_time"] + p["audio_start"]) * 1000)
            ms1 = snap((s["end_time"] + p["audio_start"]) * 1000)
            w0, w1 = snap(p["start"] * 1000), snap(p["end"] * 1000)
            ms0, ms1 = max(ms0, w0), min(ms1, w1)
            if ms1 - ms0 >= 1000:
                secs.append((ms0, ms1, str(s.get("label") or "")))
    secs.sort()
    bt, pos = beats["beats"], beats["positions"]
    bars, bar_no, cur = [], 0, None
    for t, q in zip(bt, pos):
        if q == 1:
            if cur is not None:
                bars.append((cur[0], snap(t * 1000), cur[1]))
            bar_no += 1
            cur = (snap(t * 1000), str(bar_no))
    if cur is not None:
        bars.append((cur[0], snap(duration * 1000), cur[1]))

    # ---- write -----------------------------------------------------------
    head = root.find("head")
    head.find("song").text = a.mix
    head.find("mediaFile").text = str(SHOW / f"{a.mix}.mp3")
    head.find("sequenceDuration").text = f"{duration:.3f}"

    cp = root.find("ColorPalettes"); cp.clear()
    for s in palettes:
        ET.SubElement(cp, "ColorPalette").text = s
    db = root.find("EffectDB"); db.clear()
    for s in effdb:
        ET.SubElement(db, "Effect").text = s

    dl = root.find("DisplayElements"); dl.clear()
    for nm in ("Songs", "Sections", "Bars"):
        ET.SubElement(dl, "Element", {"collapsed": "0", "type": "timing", "name": nm,
                                      "visible": "1", "views": "", "active": "1" if nm == "Songs" else "0"})
    for nm, de in display.items():
        dl.append(de)

    ee = root.find("ElementEffects"); ee.clear()
    ee.append(timing("Songs", songs)); ee.append(timing("Sections", secs)); ee.append(timing("Bars", bars))
    overlaps = 0
    for nm, rec in elements.items():
        el = ET.SubElement(ee, "Element", rec["attrib"])
        for lay in rec["layers"]:
            lay.sort(key=lambda x: int(x["startTime"]))
            L = ET.SubElement(el, "EffectLayer")
            prev_end = -1
            for at in lay:
                if int(at["startTime"]) < prev_end:
                    overlaps += 1
                prev_end = max(prev_end, int(at["endTime"]))
                ET.SubElement(L, "Effect", at)

    ET.indent(template, space="  ")
    template.write(a.out, encoding="UTF-8", xml_declaration=True)
    tot = sum(s[1] for s in stats)
    print(f"merged {len(plan)} songs -> {a.out}")
    for nm, k, c, d in stats:
        print(f"  {nm:34s} kept {k:6d}  clipped {c:4d}  outside window {d:6d}")
    print(f"  total {tot} effects, {len(palettes)} palettes, {len(effdb)} effect settings, "
          f"{len(elements)} elements; same-layer overlaps {overlaps}; "
          f"timing: {len(songs)} songs, {len(secs)} sections, {len(bars)} bars")


if __name__ == "__main__":
    main()
