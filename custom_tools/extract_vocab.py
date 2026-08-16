"""Extract a VALIDATED effect-settings vocabulary from the user's real,
human-made sequences. Copying exact settings strings avoids guessing xLights
parameter keys (a wrong key silently no-ops -- see the Chase_Rotations bug).
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
SEQS = [
    "Biometrix - Silent Night.xsq",
    "01 - Cold Heart (PNAU Remix) 2025 40fps.xsq",
    "Blinding Lights 2025 40fps.xsq",
    "Chappell Roan - Pink Pony Club 2025 40fps.xsq",
]
WANT = ["Wave", "Bars", "Fan", "Ripple", "Shimmer", "Curtain", "Pinwheel",
        "Galaxy", "Warp", "Meteors", "Shockwave", "Twinkle", "On", "Off",
        "Color Wash", "Spirals", "Morph", "Plasma", "Butterfly", "Tendril",
        "Strobe", "Lightning", "Snowflakes", "Marquee"]

out = defaultdict(Counter)

for seq in SEQS:
    p = SHOW / seq
    if not p.exists():
        continue
    root = ET.parse(p).getroot()
    db = [e.text or "" for e in root.find("EffectDB")]
    ee = root.find("ElementEffects")
    for elem in ee:
        if elem.get("type") != "model":
            continue
        for layer in elem:
            for e in layer:
                nm = e.get("name", "")
                if nm not in WANT:
                    continue
                ref = e.get("ref")
                if ref is None:
                    continue
                s = db[int(ref)]
                out[nm][s] += 1

result = {}
for nm in WANT:
    if nm not in out:
        continue
    top = out[nm].most_common(4)
    result[nm] = [{"count": c, "settings": s} for s, c in top]
    print(f"\n{'='*70}\n{nm}  ({sum(out[nm].values())} uses, {len(out[nm])} variants)\n{'='*70}")
    for s, c in top:
        print(f"  x{c}")
        print(f"    {s[:400]}")

Path("/private/tmp/claude-501/-Volumes-XLIGHTS-XLights-2026/e63440c6-1e60-4008-b9b0-199368b74605/scratchpad/vocab.json").write_text(json.dumps(result, indent=2))
print("\n\nwrote vocab.json")
