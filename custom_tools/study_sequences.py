"""Empirically study finished, human-made sequences to learn the patterns
we should be reproducing: group vs model targeting, staging/density over
time, effect vocabulary per prop type, and how sections differ.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/src")
from xlights_mcp.xlights.show import load_show_config  # noqa: E402

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")

SEQS = [
    "Biometrix - Silent Night.xsq",           # bass music -- closest to Gooey
    "01 - Cold Heart (PNAU Remix) 2025 40fps.xsq",
    "Blinding Lights 2025 40fps.xsq",
    "Chappell Roan - Pink Pony Club 2025 40fps.xsq",
    "Coldplay - A Sky Full Of Stars.xsq",
]


def main():
    sc = load_show_config(SHOW)
    real_models = {m.name for m in sc.models}
    group_names = {g.name for g in sc.model_groups}
    model_display = {m.name: m.display_as for m in sc.models}

    for seq in SEQS:
        p = SHOW / seq
        if not p.exists():
            print(f"!! missing {seq}")
            continue
        print("=" * 78)
        print(seq)
        print("=" * 78)

        tree = ET.parse(p)
        root = tree.getroot()
        head = root.find("head")
        dur = float(head.findtext("sequenceDuration", "0") or 0)
        print(f"duration {dur:.1f}s   timing {head.findtext('sequenceTiming')}")

        db = [e.text or "" for e in (root.find("EffectDB") or [])]
        ee = root.find("ElementEffects")
        if ee is None:
            print("no ElementEffects")
            continue

        as_group = []
        as_model = []
        unknown = []
        # effect coverage: total lit-time per element
        total_effects = 0
        eff_by_target_kind = Counter()
        eff_types = Counter()
        layer_counts = []
        durations = []
        # timeline occupancy: which elements have an effect at each 1s tick
        ticks = int(dur) + 1
        occupancy = [0] * ticks
        elements_with_effects = 0

        per_element = {}

        for elem in ee:
            name = elem.get("name", "")
            if elem.get("type") != "model":
                continue
            n_eff = 0
            layers = 0
            spans = []
            for layer in elem:
                layers += 1
                for e in layer:
                    st = int(e.get("startTime", "0"))
                    en = int(e.get("endTime", "0"))
                    nm = e.get("name", "")
                    eff_types[nm] += 1
                    durations.append(en - st)
                    spans.append((st, en))
                    n_eff += 1
            if n_eff == 0:
                continue
            elements_with_effects += 1
            total_effects += n_eff
            layer_counts.append(layers)
            per_element[name] = (n_eff, layers, spans)

            if name in group_names:
                as_group.append(name)
                eff_by_target_kind["group"] += n_eff
            elif name in real_models:
                as_model.append(name)
                eff_by_target_kind["model"] += n_eff
            else:
                unknown.append(name)
                eff_by_target_kind["unknown"] += n_eff

            for (st, en) in spans:
                for t in range(max(0, st // 1000), min(ticks, en // 1000 + 1)):
                    occupancy[t] += 1

        print(f"elements with effects: {elements_with_effects}"
              f"  (groups {len(as_group)}, models {len(as_model)}, unknown {len(unknown)})")
        print(f"total effect placements: {total_effects}")
        print(f"placements by target kind: {dict(eff_by_target_kind)}")
        print(f"layers per element: min {min(layer_counts)} max {max(layer_counts)} "
              f"mean {sum(layer_counts)/len(layer_counts):.1f}")
        ds = sorted(durations)
        print(f"effect duration ms: p10 {ds[len(ds)//10]}  median {ds[len(ds)//2]}  "
              f"p90 {ds[9*len(ds)//10]}  max {ds[-1]}")
        print(f"top effects: {eff_types.most_common(14)}")

        print(f"GROUP targets used ({len(as_group)}): {sorted(as_group)[:40]}")
        print(f"MODEL targets used ({len(as_model)}): {sorted(as_model)[:25]}"
              f"{' ...' if len(as_model) > 25 else ''}")
        if unknown:
            print(f"UNKNOWN targets ({len(unknown)}): {sorted(unknown)[:20]}")

        # staging: occupancy over time (how many elements lit simultaneously)
        step = max(1, ticks // 40)
        bar = []
        peak = max(occupancy) or 1
        for i in range(0, ticks, step):
            v = occupancy[i]
            bar.append(f"{i:>3}s:{v:>3}")
        print("occupancy (elements lit) sampled:")
        print("   " + "  ".join(bar[:20]))
        print("   " + "  ".join(bar[20:40]))
        print(f"peak simultaneous elements: {peak}   "
              f"mean: {sum(occupancy)/len(occupancy):.1f}")

        # how many DISPLAY elements exist total (for density ratio)
        de = root.find("DisplayElements")
        n_de = len([x for x in de if x.get("type") == "model"]) if de is not None else 0
        print(f"DisplayElements model entries: {n_de}  -> "
              f"mean lit fraction {sum(occupancy)/len(occupancy)/max(n_de,1)*100:.0f}%")
        print()


if __name__ == "__main__":
    main()
