"""Extract full model layout (positions, groups, chase-ready sequences) from
the xLights show XML, so the sequence generator can reason about physical
adjacency without needing a screenshot.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/src")
from xlights_mcp.xlights.show import load_show_config  # noqa: E402

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
OUT_PATH = SHOW_PATH / ".generation" / "layout.json"


def parse_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    show_config = load_show_config(SHOW_PATH)
    canonical = {m.name: m for m in show_config.models}

    tree = ET.parse(SHOW_PATH / "xlights_rgbeffects.xml")
    root = tree.getroot()
    models_elem = root.find("models")

    models = []
    for m in models_elem.findall("model"):
        name = m.get("name", "")
        if not name or name not in canonical:
            continue
        c = canonical[name]
        models.append({
            "name": name,
            "display_as": c.display_as,
            "controller": c.controller,
            "x": parse_float(m.get("WorldPosX")),
            "y": parse_float(m.get("WorldPosY")),
            "z": parse_float(m.get("WorldPosZ")),
            "dir": m.get("Dir", ""),
            "string_type": c.string_type,
            "face_definitions": c.face_definitions,
        })

    print(f"Parsed {len(models)} models with positions")

    groups_out = {g.name: g.members for g in show_config.model_groups}
    print(f"Found {len(groups_out)} xLights-defined model groups")

    # --- Group into families by stripping trailing number (handles "Orb-1",
    # "Flood-1", "Tunnel Arches 1", "Star 1 12in" style names) ---
    def base_key(name: str) -> str | None:
        # Strip a trailing "<number>" or "-<number>" token
        m = re.match(r"^(.*?)[\s\-]*(\d+)$", name)
        if m:
            return m.group(1).strip()
        return None

    families: dict[str, list[dict]] = defaultdict(list)
    for model in models:
        key = base_key(model["name"])
        if key:
            families[key].append(model)

    # Only keep families with 2+ members (actual chase-able sequences)
    families = {k: v for k, v in families.items() if len(v) >= 2}

    chase_sequences = {}
    for key, members in families.items():
        # Determine dominant spatial axis (the one with greatest spread) to
        # sort the physical chase order along.
        xs = [m["x"] for m in members]
        ys = [m["y"] for m in members]
        zs = [m["z"] for m in members]
        spreads = {"x": max(xs) - min(xs), "y": max(ys) - min(ys), "z": max(zs) - min(zs)}
        axis = max(spreads, key=spreads.get)
        ordered = sorted(members, key=lambda m: m[axis])
        chase_sequences[key] = {
            "axis": axis,
            "spread": spreads[axis],
            "count": len(ordered),
            "display_as": ordered[0]["display_as"],
            "order": [m["name"] for m in ordered],
        }

    # --- Whole-display sweep order (all models, left-to-right by global X) ---
    whole_display_lr = [m["name"] for m in sorted(models, key=lambda m: m["x"])]
    whole_display_ud = [m["name"] for m in sorted(models, key=lambda m: -m["y"])]  # top->bottom

    # --- Cluster models by naming prefix (front/north/south/garage/tunnel) for
    # side-specific sweeps ---
    side_keywords = ["Front", "North", "South", "Garage", "Tunnel", "Sun Room", "Dining Room",
                      "Kids Room", "Hallway", "Lincoln"]
    sides: dict[str, list[dict]] = defaultdict(list)
    for model in models:
        matched = False
        for kw in side_keywords:
            if kw.lower() in model["name"].lower():
                sides[kw].append(model)
                matched = True
                break
        if not matched:
            sides["Other"].append(model)

    side_sweeps = {}
    for side, members in sides.items():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda m: m["x"])
        side_sweeps[side] = [m["name"] for m in ordered]

    out = {
        "models": models,
        "xlights_model_groups": groups_out,
        "chase_sequences": chase_sequences,
        "whole_display_left_to_right": whole_display_lr,
        "whole_display_top_to_bottom": whole_display_ud,
        "side_sweeps": side_sweeps,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote layout to {OUT_PATH}")

    print("\nChase-ready families:")
    for key, info in sorted(chase_sequences.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  {key!r}: {info['count']} members, axis={info['axis']}, spread={info['spread']:.0f}")


if __name__ == "__main__":
    main()
