"""Convert this show's custom greyscale shaders to real palette colour.

    python custom_tools/colorize_shaders.py            # dry run, shows a diff
    python custom_tools/colorize_shaders.py --apply

WHY THIS EXISTS -- the assumption that was wrong for 32 shaders:

`isf_shader.py` documented ISF `color` inputs as "handled by the effect's
colour palette", and every custom shader was written to emit greyscale
(`gl_FragColor = vec4(vec3(v), v)`) in the belief that xLights would then
recolour it from the assigned palette. Measured on a real render: it does
not. A custom shader alone on a prop renders GREY -- mean saturation 0.00 on
Smoke Drift, Drift Cells, Rain on Glass and Cable Knit in Gooey. Where colour
did appear, it came from other layers or from a group element rendering over
the same model, which is why sampling songs gave an unattributable 1-to-6
hues and looked like the shaders were fine.

What xLights ACTUALLY does, established with two probe shaders:
  * an ISF input of TYPE "color" is fed the palette's colour automatically,
    with NO setting emitted -- the ISF DEFAULT is overridden
  * multiple colour inputs take palette slots 1, 2, 3 IN DECLARATION ORDER
  * a slot beyond the palette's active colours comes through white

So the route to palette colour -- and to the colour MOVEMENT Krists asked
for -- is to declare colour inputs and use them. This rewrites each shader to:

  * declare c1/c2/c3 (all 12 mood palettes carry exactly 3 active colours)
  * add a `palcol(t)` helper mapping 0..1 across those three
  * emit `vec4(palcol(a), a)` instead of `vec4(vec3(a), a)`

Alpha is unchanged, so layering, blending and every brightness value curve
behave exactly as before -- this changes hue only. A shader written as a
full-range value ramp (Oil Film) now slides through all three palette
colours; a shader that is mostly on/off picks up a two-tone reading.

The originals are copied to Shaders/_greyscale_backup/ first, so reverting is
a file copy.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SHADER_DIR = Path("/Volumes/XLIGHTS/XLights 2026/Shaders")
BACKUP_DIR = SHADER_DIR / "_greyscale_backup"

COLOR_INPUTS = """,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }"""

# The ramp is deliberately not linear-in-three-equal-thirds: c2 owns the wide
# middle, so the mid-tones that dominate most of these textures read as a
# colour of their own rather than as a brief crossfade between the ends.
#
# `palpos` is the part that took a measurement to get right. Mapping colour
# from the shader's own value alone LOOKS correct and renders as a single
# hue, because alpha carries that same value: everything at the dark end of
# the ramp is transparent, so only the top of the palette is ever on screen.
# Measured on eight converted shaders -- saturation rose to 0.25-0.98 but the
# hue count stayed at 1.
#
# So colour is driven mostly by a slow diagonal drift and only partly by the
# shader's value. The pattern, its motion and its brightness are untouched;
# what travels across it is colour. TIME is global, so the drift is in phase
# on every prop at once and the whole house shifts colour together rather
# than each canvas doing its own thing.
HELPER = """
// c1/c2/c3 are fed from the xLights palette (slots 1-3, in declaration
// order) -- see colorize_shaders.py for how that was established.
vec3 palcol(float t) {
    t = clamp(t, 0.0, 1.0);
    return mix(mix(c1.rgb, c2.rgb, smoothstep(0.0, 0.55, t)),
               c3.rgb, smoothstep(0.55, 1.0, t));
}
// Where in the palette this pixel sits: a slow diagonal drift, nudged by the
// shader's own value so bright detail still leans toward the palette's top
// end. Weighted toward the drift because value alone is invisible -- alpha
// hides exactly the part of the ramp that value would have shown.
float palpos(vec2 uv, float v) {
    float drift = 0.5 + 0.5 * sin((uv.x * 1.6 + uv.y * 1.1) * 2.1 - TIME * 0.11);
    return clamp(mix(drift, v, 0.35), 0.0, 1.0);
}
"""

TAIL = re.compile(r'gl_FragColor\s*=\s*vec4\(\s*vec3\(([^;]*?)\)\s*,\s*([^;]*?)\);')


def convert(src: str) -> str | None:
    """Return the colourized source, or None if it doesn't need converting."""
    if '"TYPE": "color"' in src:
        return None
    m = TAIL.search(src)
    if not m:
        return None

    # 1. declare the colour inputs at the end of the INPUTS array
    header_end = src.index("*/")
    header, body = src[:header_end], src[header_end:]
    close = header.rindex("]")
    header = header[:close] + COLOR_INPUTS + "\n  " + header[close:]

    # 2. helper immediately before main()
    body = body.replace("void main()", HELPER.lstrip("\n") + "\nvoid main()", 1)

    # 3. colour the output, binding the alpha expression so it is evaluated once
    # Bind the alpha expression once, under a name that cannot collide with
    # the shader's own locals. `a` seemed obvious and broke three shaders
    # outright -- Interference, Kaleido Bloom and Slow Dawn all already had a
    # local `a` (an angle, in each case), and GLSL redefinition is a compile
    # error, which xLights renders as a dead or frozen panel with no message.
    alpha = m.group(2).strip()
    body = TAIL.sub(f"float _pv = {alpha};\n    "
                    "gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);",
                    body, count=1)
    return header + body


def main() -> None:
    apply = "--apply" in sys.argv
    import generate_song as g  # noqa: E402  (slow import, only needed here)
    stems = set(g.NEW_SHADER_STEMS)
    files = [p for p in sorted(SHADER_DIR.glob("*.fs"))
             if p.stem.lower().replace(" ", "_") in stems]

    if apply:
        BACKUP_DIR.mkdir(exist_ok=True)
    done, skipped = [], []
    for p in files:
        out = convert(p.read_text())
        if out is None:
            skipped.append(p.stem)
            continue
        if apply:
            if not (BACKUP_DIR / p.name).exists():
                shutil.copy2(p, BACKUP_DIR / p.name)
            p.write_text(out)
        done.append(p.stem)

    print(f"{'converted' if apply else 'would convert'}: {len(done)}")
    if skipped:
        print(f"skipped (already colour, or nonstandard tail): {skipped}")
    if not apply:
        print("\nre-run with --apply to write. originals are backed up to "
              f"{BACKUP_DIR.name}/")


if __name__ == "__main__":
    main()
