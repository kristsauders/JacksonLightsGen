"""Verify the generated sequence against the measured human-sequence spec."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/src")
from xlights_mcp.xlights.show import load_show_config  # noqa: E402

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
SEQ = SHOW / (sys.argv[1] if len(sys.argv) > 1 else "Gooey (v6).xsq")

sc = load_show_config(SHOW)
group_names = {g.name for g in sc.model_groups}
real_models = {m.name for m in sc.models}
submodels_by_parent = {m.name: {s.name for s in m.submodels} for m in sc.models}


def resolves(name: str) -> bool:
    if name in group_names or name in real_models:
        return True
    if "/" in name:
        parent, sub = name.split("/", 1)
        return sub in submodels_by_parent.get(parent, ())
    return False

root = ET.parse(SEQ).getroot()
print(f"XML OK: {SEQ.name}")
db = [e.text or "" for e in root.find("EffectDB")]
dur = float(root.find("head").findtext("sequenceDuration", "0"))

ee = root.find("ElementEffects")
ticks = int(dur) + 1
occ = [0] * ticks
kinds = Counter()
eff_types = Counter()
durs = []
buffer_styles = Counter()
targets = {}

for elem in ee:
    if elem.get("type") != "model":
        continue
    name = elem.get("name", "")
    n = 0
    for layer in elem:
        for e in layer:
            st, en = int(e.get("startTime", 0)), int(e.get("endTime", 0))
            eff_types[e.get("name", "")] += 1
            durs.append(en - st)
            n += 1
            ref = e.get("ref")
            if ref is not None:
                s = db[int(ref)]
                for part in s.split(","):
                    if part.startswith("B_CHOICE_BufferStyle="):
                        buffer_styles[part.split("=", 1)[1]] += 1
            for t in range(max(0, st // 1000), min(ticks, en // 1000 + 1)):
                occ[t] += 1
    if n:
        targets[name] = n
        if name in group_names:
            kinds["GROUP"] += 1
        elif name in real_models:
            kinds["model"] += 1
        elif resolves(name):
            kinds["submodel"] += 1
        else:
            kinds["STALE"] += 1

print(f"\ntargets with effects: {len(targets)}  ({dict(kinds)})   [human: 45-57]")
print(f"total placements: {sum(targets.values())}   [human: 2438-8487]")
ds = sorted(durs)
print(f"durations ms: p10={ds[len(ds)//10]} median={ds[len(ds)//2]} p90={ds[9*len(ds)//10]}"
      f"   [human median 250-825]")
short = sum(1 for d in durs if d <= 1000) / len(durs) * 100
print(f"fraction <=1s: {short:.0f}%   [human ~70%]")
print(f"\nbuffer styles used: {dict(buffer_styles)}")
print(f"effect types: {eff_types.most_common(12)}")

n_de = len([x for x in root.find("DisplayElements") if x.get("type") == "model"])
mean_occ = sum(occ) / len(occ)
print(f"\nmean simultaneous lit targets: {mean_occ:.1f}  peak {max(occ)}")
# NOTE: this "lit fraction" counts display ELEMENTS carrying an effect, which
# is not the same as how much of the display is actually lit -- a group element
# expands to many models at render time, so the two diverge badly. Do not treat
# the 11-26% figure as a density target: measuring the real rendered output
# (custom_tools/verify_render.py) shows Krists's own sequences light ~50% of
# pixels, while a generated sequence scoring 45% here lit only 27% of pixels.
# Chasing this number down once already made sequences too sparse. For real
# density, render and measure; this line is only useful for spotting big
# relative swings between two generated versions.
print(f"mean lit fraction: {mean_occ/max(n_de,1)*100:.0f}%   "
      f"[elements-with-effects, NOT rendered brightness -- see verify_render.py]")

step = max(1, ticks // 24)
print("\noccupancy over time:")
row = [f"{i:>3}s:{occ[i]:>3}" for i in range(0, ticks, step)]
for i in range(0, len(row), 12):
    print("   " + "  ".join(row[i:i+12]))

# curtain check
print("\n--- garage curtain targeting ---")
for n in ["Curtain Garage", "Curtain Garage 1", "Curtain Garage-2", "Curtain Garage-3",
          "Curtain Front", "Curtain Front Left", "Curtain North", "Curtain South"]:
    print(f"  {n:26s} {'TARGETED x'+str(targets[n]) if n in targets else '(not targeted)'}")

print("\n--- STALE targets (would render as orphans) ---")
stale = [n for n in targets if not resolves(n)]
print("  none" if not stale else f"  {stale}")
