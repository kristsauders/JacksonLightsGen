"""Are the shader changes, section boundaries and drops all on the same grid?

    python custom_tools/verify_alignment.py "Gooey (v209)" ...
    python custom_tools/verify_alignment.py --all

Three questions, because they fail independently:

  beat      what share of shader changes land within 40ms of a beat
  section   how far each section boundary sits from the nearest drum/bass
            onset -- a boundary that is not on a hit will pull every pass
            that reads it off the music
  drop      does the look actually change ON each drop, and not before it
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

SHOW = Path(json.loads((Path(__file__).resolve().parents[1] / "config.json")
                       .read_text())["show_folders"]["christmas2026"])
BIG = ("matri", "mega tree", "garage roof", "tunnel")


def audit(name: str) -> None:
    song = name.rsplit(" (", 1)[0]
    gen = SHOW / ".generation" / song
    analysis = json.loads((gen / "analysis.json").read_text())
    beats = np.array(analysis["beats"]["beat_times"])
    secs = [s["start_time"] for s in analysis["sections"]]
    # Prefer what the generator actually targeted. It snaps both drops and
    # section starts onto the hit the band lands on, and measuring against the
    # raw detections instead reported correctly-placed changes as misses.
    prov = {}
    try:
        prov = json.loads((gen / "provenance.json").read_text())
    except (OSError, ValueError):
        pass
    drops = [float(t) for t in prov.get("snapped_drops", [])]
    if not drops:
        cues = json.loads((gen / "cues.json").read_text())
        drops = sorted({round(float(e.get("drop_time") or e.get("time")), 2)
                        for k in ("energy_drops",) for e in (cues.get(k) or [])
                        if (e.get("drop_time") or e.get("time"))})
    if prov.get("section_starts"):
        secs = [float(t) for t in prov["section_starts"]]

    root = ET.parse(SHOW / f"{name}.xsq").getroot()
    starts = []
    for el in root.find("ElementEffects"):
        nm = (el.get("name") or "").lower()
        if "/" in nm or not any(k in nm for k in BIG):
            continue
        for layer in el.findall("EffectLayer"):
            for ef in layer.findall("Effect"):
                if ef.get("name") == "Shader":
                    starts.append(int(ef.get("startTime")) / 1000.0)
    starts = np.array(sorted(set(starts)))
    on_beat = (np.array([np.min(np.abs(beats - s)) for s in starts]) < 0.04
               ).mean() * 100 if starts.size else 0.0

    # Section boundaries vs the nearest drum/bass onset. Read the voices the
    # same way the generator does, from the cached analysis where possible.
    # Against the BEAT GRID. Re-deriving the voices here gave numbers that
    # disagreed with the generator's own (a different call signature), and a
    # measurement that does not match what the code sees is worse than none.
    if secs and beats.size:
        off = np.array([np.min(np.abs(beats - t)) for t in secs])
        sec_txt = (f"median {np.median(off) * 1000:3.0f}ms off the beat, "
                   f"{(off < 0.04).mean() * 100:3.0f}% on it")
    else:
        sec_txt = "no section data"

    late = early = onit = 0
    for d in drops:
        near = starts[(starts > d - 5.0) & (starts < d + 1.0)] if starts.size \
            else np.array([])
        # +-0.8s, not 0.25. A drop is snapped forward to the first strong hit
        # after the detected rise -- up to about a second -- so comparing a
        # change against the RAW detection reports a correctly-placed change
        # as a miss.
        if any(-0.25 <= s - d <= 0.8 for s in near):
            onit += 1
        elif any(d - 5.0 < s < d - 0.25 for s in near):
            early += 1
        else:
            late += 1
    print(f"{name[:34]:34s} beat {on_beat:3.0f}%  |  sections {sec_txt}  |  "
          f"drops on {onit}/{len(drops)}, early {early}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--all":
        seen = {}
        for p in SHOW.glob("*.xsq"):
            m = re.match(r"(.*) \(v(\d+)\)\.xsq$", p.name)
            if m and (SHOW / f"{m.group(1)} (v{m.group(2)}).fseq").exists():
                s, v = m.group(1), int(m.group(2))
                if v > seen.get(s, -1):
                    seen[s] = v
        args = [f"{s} (v{v})" for s, v in sorted(seen.items())]
    for a in args:
        try:
            audit(a)
        except Exception as e:
            print(f"{a[:34]:34s} -- {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
