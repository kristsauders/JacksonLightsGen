"""Deep dive on Biometrix - Silent Night (bass music, closest to Gooey):
how group elements are declared, how staging works per section, what the
SingleStrand-heavy vocabulary actually looks like, and how Off is used.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/src")
from xlights_mcp.xlights.show import load_show_config  # noqa: E402

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
SEQ = SHOW / "Biometrix - Silent Night.xsq"


def main():
    sc = load_show_config(SHOW)
    group_names = {g.name for g in sc.model_groups}

    tree = ET.parse(SEQ)
    root = tree.getroot()
    db = [e.text or "" for e in root.find("EffectDB")]

    # --- 1. How are group elements declared in DisplayElements? ---
    de = root.find("DisplayElements")
    print("=== DisplayElements: first 12 entries (raw attrs) ===")
    for x in list(de)[:12]:
        print("   ", x.tag, dict(x.attrib), "| IS_GROUP" if x.get("name") in group_names else "")
    n_group_de = sum(1 for x in de if x.get("name") in group_names)
    print(f"\n{n_group_de} of {len(list(de))} DisplayElements entries are model GROUPS")

    # --- 2. Timeline: what is active per 10s window ---
    ee = root.find("ElementEffects")
    dur_s = int(float(root.find("head").findtext("sequenceDuration", "0")))
    win = 10
    per_win = defaultdict(Counter)      # window -> effect name counter
    per_win_targets = defaultdict(set)  # window -> element names
    off_events = []

    for elem in ee:
        if elem.get("type") != "model":
            continue
        name = elem.get("name", "")
        for layer in elem:
            for e in layer:
                st = int(e.get("startTime", "0"))
                en = int(e.get("endTime", "0"))
                nm = e.get("name", "")
                w = st // (win * 1000)
                per_win[w][nm] += 1
                per_win_targets[w].add(name)
                if nm == "Off":
                    off_events.append((st, en, name))

    print("\n=== Per-10s window: #targets and dominant effects ===")
    for w in range(0, dur_s // win + 1):
        c = per_win[w]
        if not c:
            print(f"  {w*win:>4}s   (silent/no new effects)")
            continue
        top = ", ".join(f"{k}x{v}" for k, v in c.most_common(4))
        print(f"  {w*win:>4}s   targets={len(per_win_targets[w]):>3}  {top}")

    print(f"\n=== Explicit 'Off' effects: {len(off_events)} ===")
    for st, en, name in off_events[:15]:
        print(f"   {st/1000:>7.2f}s - {en/1000:>7.2f}s   on {name}")

    # --- 3. SingleStrand settings variety (the dominant effect) ---
    ss_settings = Counter()
    for elem in ee:
        if elem.get("type") != "model":
            continue
        for layer in elem:
            for e in layer:
                if e.get("name") != "SingleStrand":
                    continue
                ref = e.get("ref")
                if ref is None:
                    continue
                s = db[int(ref)]
                keep = [p for p in s.split(",") if any(
                    k in p for k in ("Chase_Type1", "Chase_Rotations", "Number_Chases",
                                      "Fade_Type", "SSEFFECT_TYPE", "BufferStyle",
                                      "Chase_Size", "Group_All"))]
                ss_settings[" | ".join(keep)] += 1
    print(f"\n=== SingleStrand config variants ({len(ss_settings)} distinct) ===")
    for s, n in ss_settings.most_common(10):
        print(f"  x{n:>4}  {s}")

    # --- 4. Which targets get the most effects (the 'workhorses') ---
    per_target = Counter()
    per_target_layers = {}
    for elem in ee:
        if elem.get("type") != "model":
            continue
        name = elem.get("name", "")
        n = sum(len(list(layer)) for layer in elem)
        if n:
            per_target[name] = n
            per_target_layers[name] = len(list(elem))
    print("\n=== Top 20 targets by effect count ===")
    for name, n in per_target.most_common(20):
        kind = "GROUP" if name in group_names else "model"
        print(f"  {n:>5}  [{kind}] {name}  (layers={per_target_layers[name]})")

    # --- 5. Duration histogram ---
    durs = []
    for elem in ee:
        if elem.get("type") != "model":
            continue
        for layer in elem:
            for e in layer:
                durs.append(int(e.get("endTime", "0")) - int(e.get("startTime", "0")))
    buckets = Counter()
    for d in durs:
        if d <= 150: buckets["<=150ms"] += 1
        elif d <= 400: buckets["150-400ms"] += 1
        elif d <= 1000: buckets["400ms-1s"] += 1
        elif d <= 3000: buckets["1-3s"] += 1
        elif d <= 10000: buckets["3-10s"] += 1
        else: buckets[">10s"] += 1
    print(f"\n=== Effect duration distribution (n={len(durs)}) ===")
    for k in ["<=150ms", "150-400ms", "400ms-1s", "1-3s", "3-10s", ">10s"]:
        v = buckets.get(k, 0)
        print(f"  {k:>10}: {v:>5}  ({v/len(durs)*100:>4.1f}%)  {'#'*int(v/len(durs)*50)}")


if __name__ == "__main__":
    main()
