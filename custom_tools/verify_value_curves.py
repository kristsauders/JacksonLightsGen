"""Prove that a brightness value curve actually renders.

WHY THIS EXISTS
---------------
A malformed xLights value curve does not raise, does not warn, and leaves a
.xsq that inspects perfectly. It just renders BLACK. For a long time every
brightness curve this generator wrote was malformed in exactly that way
(`Type=Flat` with `Values=` and no `P1`), so roughly a third of the effects in
every sequence rendered as nothing -- while the .xsq showed them present and
correct. Round after round of fixes measured inert in the .fseq for that one
reason.

Nothing about that is visible without rendering, so it needs a test. This
builds a tiny sequence in which the SAME effect is repeated with different
curve encodings, renders it, and reports the measured brightness of each.
A form that measures 0.0 is broken no matter how plausible it looks.

    python custom_tools/verify_value_curves.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import channel_map  # noqa: E402
from fseq_reader import read_fseq  # noqa: E402
from xlights_mcp.xlights.palettes import ColorPalette  # noqa: E402
from xlights_mcp.xlights.show import load_show_config  # noqa: E402
from xlights_mcp.xlights.xsq_writer import (  # noqa: E402
    EffectPlacement,
    SequenceSpec,
    write_xsq,
)

SHOW = Path(json.loads((Path(__file__).resolve().parents[1] / "config.json")
                       .read_text())["show_folders"]["christmas2026"])
NAME = "ZZ Value Curve Check"
HEAD = "C_VALUECURVE_Brightness=Active=TRUE|Id=ID_VALUECURVE_Brightness|"

# Each entry: label, extra_parts, and whether it is expected to render.
CASES: list[tuple[str, list[str], bool]] = [
    ("no curve (control)", [], True),
    ("Flat + Values (BROKEN)",
     [HEAD + "Type=Flat|Min=0.00|Max=400.00|RV=TRUE|Values=0.00:0.16;1.00:0.16"],
     False),
    ("Flat + P1",
     [HEAD + "Type=Flat|Min=0.00|Max=400.00|P1=64.00|RV=TRUE"], True),
    ("Custom + Values",
     [HEAD + "Type=Custom|Min=0.00|Max=400.00|RV=TRUE|Values=0.00:0.16;1.00:0.16"],
     True),
    ("Custom ramp",
     [HEAD + "Type=Custom|Min=0.00|Max=400.00|RV=TRUE|Values=0.00:0.40;1.00:0.05"],
     True),
]


def main() -> None:
    sc = load_show_config(SHOW)
    group = next((g.name for g in sc.model_groups
                  if g.name == "Corner Verticals"), None)
    if group is None:
        print("no 'Corner Verticals' group in this show; edit `group` above")
        return
    effects, palettes = [], []
    for i, (_lbl, parts, _ok) in enumerate(CASES):
        pal = ColorPalette(colors=["#FF6A00", "#FFA033", "#FFD9A0"],
                           active_colors=[1, 2, 3], extra_parts=parts)
        palettes.append(pal)
        effects.append(EffectPlacement(
            model_name=group, layer=0, effect_name="Color Wash",
            start_time_ms=2000 + i * 4000, end_time_ms=5000 + i * 4000,
            settings={"B_CHOICE_BufferStyle": "Per Model Default",
                      "E_CHECKBOX_ColorWash_HFade": "1"},
            palette=pal))
    write_xsq(SequenceSpec(
        song_title=NAME, artist="", album="",
        media_file=str(next(SHOW.glob("*.mp3"))),
        duration_ms=4000 * len(CASES) + 4000, timing_ms=25,
        palettes=palettes, effects=effects, timing_tracks=[]),
        sc, SHOW / f"{NAME}.xsq")

    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "verify_render.py"),
                    NAME], check=False, capture_output=True)

    cm = channel_map.build()["models"]
    members = [m for m in
               json.loads((SHOW / ".generation/layout.json").read_text())
               ["xlights_model_groups"][group] if m in cm]
    fseq = read_fseq(SHOW / f"{NAME}.fseq")
    bad = 0
    for i, (lbl, _p, expect_lit) in enumerate(CASES):
        frame = np.frombuffer(fseq.frame(int((3.5 + i * 4) / 0.025)),
                              dtype=np.uint8).astype(float)
        seg = np.concatenate([frame[cm[m]["first"] - 1:cm[m]["last"]]
                              for m in members])
        lit = seg.mean() > 1.0
        ok = "ok" if lit == expect_lit else "UNEXPECTED"
        if lit != expect_lit:
            bad += 1
        print(f"  {lbl:26s} mean {seg.mean():6.1f}  "
              f"{'renders' if lit else 'BLACK':8s}  {ok}")
    for suffix in (".xsq", ".fseq"):
        (SHOW / f"{NAME}{suffix}").unlink(missing_ok=True)
    print("\nAll curve forms behaved as expected." if not bad
          else f"\n{bad} case(s) behaved differently -- xLights may have changed.")


if __name__ == "__main__":
    main()
