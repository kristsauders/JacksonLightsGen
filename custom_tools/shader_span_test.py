"""Which shaders survive being stretched across the WHOLE HOUSE?

    python custom_tools/shader_span_test.py            # all usable shaders
    python custom_tools/shader_span_test.py Wisps ...  # just these

The whole-house bed puts one shader on `ALL Display` with `Per Preview`, so a
single canvas is stretched over every prop at once. A shader with any kind of
focal point then puts its centre in the empty air between props, and the props
themselves catch only whatever is happening at the edges of the frame:

    "There's a shader on all the matrices that seems to be not circular or
     radial but it seems to have a center that's off the matrices again and so
     you just see the tendril looking outer edges on the matrices."

`is_centred()` was a hand-maintained list of stems, which by construction only
knows about shaders someone remembered to add. Wisps is a stock shader and was
never on it. So measure the actual failure instead of enumerating causes: run
each shader spanned across the house and look at how the light lands on the
PROPS.

  spread  coefficient of variation of per-prop mean brightness. Low means
          every prop gets a comparable share of the picture; high means the
          picture is concentrated somewhere and most props are off to one side
          of it.
  dark%   share of props sitting below 20% of the brightest prop -- the
          literal "you only see the outer edges" symptom, and the number that
          actually decides the verdict.

Results cache to .generation/shader_span.json, which generate_song.py reads to
build the house-bed pool. Re-run it after adding shaders.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import channel_map  # noqa: E402
from fseq_reader import read_fseq  # noqa: E402
from isf_shader import shader_settings, usable_shaders  # noqa: E402
from xlights_mcp.xlights.palettes import ColorPalette  # noqa: E402
from xlights_mcp.xlights.show import load_show_config  # noqa: E402
from xlights_mcp.xlights.xsq_writer import (EffectPlacement,  # noqa: E402
                                            SequenceSpec, write_xsq)

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
SHADER_DIR = SHOW_PATH / "Shaders"
XLIGHTS = Path("/Applications/xLights.app/Contents/MacOS/xLights")
CACHE = SHOW_PATH / ".generation" / "shader_span.json"
NAME = "_shader_span"
HOUSE_GROUP = "ALL Display"
SLOT_MS = 6000
SPAN = {"B_CHOICE_BufferStyle": "Per Preview"}
WHITE = ColorPalette(colors=["#FFFFFF"], active_colors=[1])
# A prop this dim is showing essentially nothing of the picture.
DARK_FRAC = 0.20
# Above this share of dark props the shader is not fit to span the house.
MAX_DARK = 0.45


def main() -> None:
    wanted = sys.argv[1:]
    shaders = usable_shaders(SHADER_DIR)
    if wanted:
        shaders = [p for p in shaders if p.stem in wanted]

    effects, used = [], []
    for i, fs in enumerate(shaders):
        s = i * SLOT_MS
        effects.append(EffectPlacement(
            model_name=HOUSE_GROUP, layer=0, effect_name="Shader",
            start_time_ms=s, end_time_ms=s + SLOT_MS - 200,
            settings={**shader_settings(fs, speed=300), **SPAN},
            palette=WHITE))
        used.append(fs)

    out = SHOW_PATH / f"{NAME}.xsq"
    print(f"spanning {len(used)} shaders across {HOUSE_GROUP} ...")
    write_xsq(SequenceSpec(song_title=NAME,
                           media_file=str(next(SHOW_PATH.glob("*.mp3"))),
                           duration_ms=len(used) * SLOT_MS + 2000, timing_ms=25,
                           palettes=[WHITE], effects=effects),
              load_show_config(SHOW_PATH), out)
    subprocess.run([str(XLIGHTS), "-s", str(SHOW_PATH), "-r", str(out)],
                   capture_output=True)

    f = read_fseq(SHOW_PATH / f"{NAME}.fseq", frame_stride=1)
    cm = channel_map.build()["models"]
    # Props big enough for "how much of the picture landed here" to mean
    # something; a 3-node prop is noise either way.
    props = {n: (v["first"], v["last"] + 1) for n, v in cm.items()
             if v["count"] >= 150}

    verdict = {}
    print(f"\n{'shader':24} {'spread':>7} {'dark%':>6} {'level':>6}  verdict   ({len(props)} props)")
    for i, fs in enumerate(used):
        means = []
        for off in (1500, 3000, 4500):
            idx = min(int((i * SLOT_MS + off) / f.step_time_ms), len(f) - 1)
            frame = np.frombuffer(f.frame(idx), dtype=np.uint8).astype(np.float32)
            means.append([frame[a:b].mean() for a, b in props.values()])
        m = np.array(means).mean(axis=0)
        peak = m.max()
        if peak < 1.0:
            spread, dark = 0.0, 1.0
        else:
            spread = float(m.std() / max(m.mean(), 1e-6))
            dark = float((m < peak * DARK_FRAC).mean())
        # How BRIGHT the shader is when spread over everything. The house bed
        # runs under the whole show continuously, so a bright one lifts the
        # floor of the entire display: it pushes the lit fraction up and, worse,
        # compresses the run-up dip, because a dip is measured as a RATIO
        # against the surrounding light and a constant bright base leaves less
        # to take away.
        level = float(m.mean() / 255.0)
        ok = dark <= MAX_DARK
        verdict[fs.stem] = {"spread": round(spread, 3),
                            "dark_frac": round(dark, 3),
                            "level": round(level, 4), "span_ok": ok}
        print(f"{fs.stem[:24]:24} {spread:7.2f} {dark*100:5.0f}% {level:6.3f}  "
              + ("even enough to span" if ok else "CONCENTRATED -- don't span"))

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(verdict, indent=1, sort_keys=True))
    n_ok = sum(1 for v in verdict.values() if v["span_ok"])
    print(f"\n{n_ok}/{len(verdict)} can span the house -> {CACHE}")


if __name__ == "__main__":
    main()
