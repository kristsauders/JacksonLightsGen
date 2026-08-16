"""Does a greyscale shader's VALUE map across the palette, or just scale one colour?

    python custom_tools/shader_palette_test.py "Oil Film" "Aurora Curtain" ...

This matters because it decides how to write a shader. If xLights maps a
shader's output value along the palette gradient, then a shader written as a
full-range value RAMP produces bands of DIFFERENT COLOURS sliding across the
house -- real colour movement, which is what Krists asked for. If instead the
palette only tints, then the same ramp is one colour breathing, and colour
variety has to come from somewhere else entirely.

Sampling a real song's render could not answer this: shaders there sit under
other layers and each carries a different palette, so hue counts came back
between 1 and 6 with no way to attribute the difference. This isolates it --
one shader, one layer, nothing else on the prop, and the SAME deliberately
hue-spread palette for every shader tested. Any difference in hue count is
then the shader's doing.
"""

from __future__ import annotations

import colorsys
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import channel_map  # noqa: E402
from fseq_reader import read_fseq  # noqa: E402
from isf_shader import shader_settings  # noqa: E402
from shader_smoke_test import SHADER_DIR, SHOW_PATH, SLOT_MS, TEST_MODEL, XLIGHTS  # noqa: E402
from xlights_mcp.xlights.palettes import ColorPalette  # noqa: E402
from xlights_mcp.xlights.show import load_show_config  # noqa: E402
from xlights_mcp.xlights.xsq_writer import (EffectPlacement,  # noqa: E402
                                            SequenceSpec, write_xsq)

NAME = "_shader_palette"
# Three colours ~120 degrees apart, all fully saturated. Deliberately spread
# so that ANY gradient mapping is unmissable in the hue histogram -- a subtle
# palette (pink/orange/white) is exactly what made the earlier reading
# ambiguous.
#
# `active_colors` holds palette SLOT NUMBERS, not a per-slot on/off flag.
# Writing [1, 1, 1] marks slot 1 active three times and leaves slots 2 and 3
# off -- which made every shader here report a single hue, because xLights
# feeds colour inputs from the ACTIVE colours in order and c2/c3 fell back to
# white. That looked exactly like a shader bug and cost a bisect to find.
SPREAD = ColorPalette(colors=["#0044FF", "#00FF66", "#FF9900"],
                      active_colors=[1, 2, 3])


def main() -> None:
    wanted = sys.argv[1:]
    shaders = [p for p in sorted(SHADER_DIR.glob("*.fs")) if p.stem in wanted]
    media = next(SHOW_PATH.glob("*.mp3"))
    out = SHOW_PATH / f"{NAME}.xsq"
    effects, used = [], []
    for i, fs in enumerate(shaders):
        s = i * SLOT_MS
        effects.append(EffectPlacement(
            model_name=TEST_MODEL, layer=0, effect_name="Shader",
            start_time_ms=s, end_time_ms=s + SLOT_MS - 200,
            settings=shader_settings(fs, speed=400), palette=SPREAD))
        used.append(fs)
    write_xsq(SequenceSpec(song_title=NAME, media_file=str(media),
                           duration_ms=len(used) * SLOT_MS + 2000, timing_ms=25,
                           palettes=[SPREAD], effects=effects),
              load_show_config(SHOW_PATH), out)
    subprocess.run([str(XLIGHTS), "-s", str(SHOW_PATH), "-r", str(out)],
                   capture_output=True)

    f = read_fseq(SHOW_PATH / f"{NAME}.fseq", frame_stride=1)
    rg = channel_map.build()["models"][TEST_MODEL]
    lo, hi = rg["first"], rg["last"] + 1
    print(f"palette {SPREAD.colors} on {TEST_MODEL}\n")
    print(f"{'shader':22} {'hues':>5} {'sat':>5}  reading")
    for i, fs in enumerate(used):
        idx = min(int((i * SLOT_MS + 3000) / f.step_time_ms), len(f) - 1)
        px = np.frombuffer(f.frame(idx), dtype=np.uint8)[lo:hi].astype(int)
        px = px[:len(px) // 3 * 3].reshape(-1, 3)
        lit = px[px.max(axis=1) > 25]
        if len(lit) < 50:
            print(f"{fs.stem[:22]:22} {'-':>5} {'-':>5}  too dark to read")
            continue
        hsv = np.array([colorsys.rgb_to_hsv(*(p / 255.0)) for p in lit[::5]])
        # Count only well-saturated pixels: near-white/near-black pixels have
        # a meaningless hue and would inflate the count either way.
        sat = hsv[:, 1]
        hues = hsv[sat > 0.35, 0] * 360.0
        nb = len(set((hues // 30).astype(int))) if len(hues) else 0
        reading = ("value maps ACROSS the palette" if nb >= 3 else
                   "one colour, brightness only" if nb <= 1 else "partial")
        print(f"{fs.stem[:22]:22} {nb:5} {sat.mean():5.2f}  {reading}")


if __name__ == "__main__":
    main()
