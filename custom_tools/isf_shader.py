"""Build correct xLights `Shader` effect settings from a .fs file's own ISF header.

Why: xLights names every shader parameter after the shader's OWN declared ISF
input, as `E_<CONTROL>_SHADERXYZZY_<inputName>`. Those key names are therefore
different for every shader file. Copying a known-good settings string from one
sequence and pointing it at a different .fs (which is what round 8 did --
lifting `uContRot`/`uColMode`/`uIntensity`/`uOffsetX/Y`/`uRotate`, which belong
to `Hex 3D Spiral.fs` / `LogTransSpiralSmall.fs`, onto `Wisps.fs` whose inputs
are `lines`/`amp`/`glow`/...) produces settings that silently no-op, because a
wrong `E_` key is ignored rather than erroring.

xLights omits any parameter left at its ISF default (verified: Silent Night's
shader effects carry only the inputs that were actually changed, and omit
`uC1`/`uC2`/`uTwist`/`uOutline`/`uReverse` entirely), so emitting explicit
defaults is safe and self-documenting.

Control-type mapping, derived from real xLights settings strings:
    ISF "float"   -> E_SLIDER_SHADERXYZZY_<name>     (integer, value * 100)
    ISF "bool"    -> E_CHECKBOX_SHADERXYZZY_<name>   (0/1)
    ISF "long"    -> E_CHOICE_SHADERXYZZY_<name>     (the option LABEL string)
    ISF "point2D" -> E_SLIDER_SHADERXYZZY_<name>X and ...<name>Y
    ISF "color"   -> fed from the effect's palette, not a setting. Verified on
                     a real render 2026-08-16: xLights assigns the palette's
                     ACTIVE colours to colour inputs in declaration order
                     (slot 1 -> first colour input, and so on), overriding the
                     ISF default; a colour input past the last active colour
                     comes through white. Floats declared ahead of them do not
                     shift the mapping. This is the ONLY way a shader gets
                     palette colour -- emitting greyscale does not, whatever
                     this file used to imply.
    ISF "image"   -> needs an input image; such shaders are unusable here

The float scaling (x100) is inferred from `Hex 3D Spiral.fs`'s `uIntensity`
(ISF range 0..3) appearing as `E_SLIDER_SHADERXYZZY_uIntensity=171` in a real
sequence, i.e. 1.71 stored as 171.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class ShaderUnusable(Exception):
    """Raised for shaders this generator must not place (e.g. ones requiring
    an input image, which we have no way to supply)."""


def parse_isf_header(fs_path: str | Path) -> dict:
    """Return the ISF JSON header dict of a .fs file."""
    src = Path(fs_path).read_text(errors="replace")
    m = re.search(r"/\*\s*(\{.*?\})\s*\*/", src, re.S)
    if not m:
        raise ShaderUnusable(f"{Path(fs_path).name}: no ISF header block")
    return json.loads(m.group(1))


# Param-name substrings that control PACE/MOTION rather than shape/density.
# Jittered over a wider span than other floats -- direct feedback: "adding
# more movement... changing the parameters." A shader whose rotation/zoom/
# scroll rate is jittered per song visibly moves differently even when the
# underlying pattern and palette are the same, which is the single biggest
# lever for "don't want them all to look the same" since every other float
# (density/thickness/radius-type params) mostly changes the pattern's
# geometry, not its felt energy.
_MOTION_NAME_HINTS = ("rate", "speed", "rotat", "zoom", "scroll", "cycle",
                      "freq", "loop")


def shader_settings(fs_path: str | Path, speed: int = 400,
                    overrides: dict[str, float] | None = None,
                    jitter_rng: object | None = None) -> dict[str, str]:
    """Build the full settings dict for an xLights Shader effect on `fs_path`.

    `overrides` maps an ISF input NAME to a raw (unscaled) value, for the few
    cases where the shader's own default is not what we want visually --
    ALWAYS wins over jitter (a safety pin like InnerDimensionalMatrix's
    `flash` override must never be randomized back toward its risky ISF
    default).

    `jitter_rng`, when given a `random.Random`, varies every OTHER float
    input around its ISF default (scaled to the input's own MIN/MAX span, so
    a default of exactly 0 -- common for offset/scroll-type params -- still
    gets real variation instead of jitter*0). Without a jitter_rng, behavior
    is unchanged: every parameter sits at its ISF default, identically,
    every time this shader is used anywhere. That was the actual reason
    otherwise-palette-varied shaders still read as repetitive across
    different songs -- confirmed by direct pixel sampling that the palette
    itself already varies correctly; nothing else ever did.
    """
    fs_path = Path(fs_path)
    header = parse_isf_header(fs_path)
    inputs = header.get("INPUTS", [])
    overrides = overrides or {}

    for inp in inputs:
        if inp.get("TYPE") == "image":
            raise ShaderUnusable(
                f"{fs_path.name}: requires an input image ('{inp.get('NAME')}')")

    if jitter_rng is not None:
        # Vary DOWNWARD much more than upward. Direct feedback across two
        # songs: "there are a few shaders that are ironically too fast, some
        # on the matrices and some on the whole house... it loses its effect."
        # A shader that is too slow still reads as texture; one that is too
        # fast reads as strobing on a big prop, so the risk is asymmetric and
        # the range should be too. Was 0.55-1.6.
        speed = int(round(speed * jitter_rng.uniform(0.45, 1.05)))

    settings: dict[str, str] = {
        "E_0FILEPICKERCTRL_IFS": str(fs_path),
        "E_SLIDER_Shader_Speed": str(speed),
        "E_TEXTCTRL_Shader_LeadIn": "0",
        "E_TEXTCTRL_Shader_Offset_X": "0",
        "E_TEXTCTRL_Shader_Offset_Y": "0",
        "E_TEXTCTRL_Shader_Zoom": "0",
    }

    for inp in inputs:
        name = inp.get("NAME")
        typ = inp.get("TYPE")
        is_override = name in overrides
        default = overrides.get(name, inp.get("DEFAULT"))
        if name is None or default is None:
            continue
        if typ == "float" and jitter_rng is not None and not is_override:
            lo, hi = inp.get("MIN"), inp.get("MAX")
            if lo is not None and hi is not None and hi > lo:
                lo, hi = float(lo), float(hi)
                span = hi - lo
                wide = any(h in name.lower() for h in _MOTION_NAME_HINTS)
                spread = span * (0.45 if wide else 0.22)
                # Sample INSIDE a margin-inset range, never clamp to it.
                #
                # The first version sampled uniformly around the default and
                # then clamped to [lo, hi]. Whenever a default sits near an
                # edge of its own range -- which is common -- that piles
                # probability mass onto the exact boundary value. Measured on
                # Wisps.fs, whose defaults are glow=-6 of [-40,0], zoom=11 of
                # [0,100], amp=0.1 of [0,1]: real songs came out with zoom=0,
                # amp=0, and (Gooey) glow=0, i.e. maximum glow with no wave
                # amplitude -- a blown-out white wash. That is the "multiple
                # matrices are just white" report, and it was self-inflicted
                # by the jitter, not a broken shader.
                #
                # Keeping a 6% margin off both ends means an extreme is now
                # merely possible rather than a probability spike, and no
                # parameter can ever land exactly on a degenerate endpoint.
                margin = span * 0.06
                a = max(lo + margin, float(default) - spread)
                b = min(hi - margin, float(default) + spread)
                if b <= a:                      # range too tight to jitter
                    a = b = min(max(float(default), lo + margin), hi - margin)
                default = jitter_rng.uniform(a, b)
        if typ == "float":
            settings[f"E_SLIDER_SHADERXYZZY_{name}"] = str(int(round(float(default) * 100)))
        elif typ == "bool":
            settings[f"E_CHECKBOX_SHADERXYZZY_{name}"] = "1" if default else "0"
        elif typ == "long":
            labels = inp.get("LABELS") or []
            values = inp.get("VALUES") or []
            label = None
            if labels:
                if default in values:
                    label = labels[values.index(default)]
                else:
                    label = labels[0]
            if label is not None:
                settings[f"E_CHOICE_SHADERXYZZY_{name}"] = str(label)
        elif typ == "point2D":
            try:
                dx, dy = float(default[0]), float(default[1])
            except (TypeError, IndexError, ValueError):
                dx = dy = 0.0
            settings[f"E_SLIDER_SHADERXYZZY_{name}X"] = str(int(round(dx * 100)))
            settings[f"E_SLIDER_SHADERXYZZY_{name}Y"] = str(int(round(dy * 100)))
        # "color" inputs are driven by the effect's palette, not a setting.

    return settings


OWN_SHADER_CREDIT = "written for the Auders 2026 show"


def has_uncontrolled_color(fs_path: str | Path) -> bool:
    """True if the shader may put colour on the house that the palette does
    not govern.

    **The original reasoning here was wrong and is worth stating plainly**,
    because it silently cost this library its colour. This used to claim a
    `color` TYPE input meant the shader's colour was "baked in, invisible to
    and unchanged by whatever xLights palette we assign it". Measured with a
    probe shader on a real render (2026-08-16), the opposite is true:
    xLights feeds ISF `color` inputs FROM THE PALETTE automatically -- slots
    1, 2, 3 in declaration order, overriding the ISF default, with no
    setting emitted. A colour input is the palette's route IN, not a way
    around it.

    What was real is the measurement that prompted the rule: Zoom Loop.fs
    once measured ~39% of lit pixels as saturated red when shared across 7
    canvases at once, and Circle Tunnel.fs has the same shape. A stock
    shader can take a palette colour AND mix in colour of its own, and there
    is no way to tell statically which it does. So stock shaders declaring
    colour inputs stay out of the pool, on the measurement rather than the
    theory.

    Shaders written for this show are exempt: their colour comes only from
    c1/c2/c3, which are palette slots, so they are palette-governed by
    construction. The exemption keys off the CREDIT line so new ones written
    the same way qualify without editing any list."""
    try:
        header = parse_isf_header(fs_path)
    except (ShaderUnusable, json.JSONDecodeError):
        return False
    if OWN_SHADER_CREDIT in header.get("CREDIT", ""):
        return False
    inputs = header.get("INPUTS", [])
    has_color = any(i.get("TYPE") == "color" for i in inputs)
    has_colmode = any(i.get("NAME") == "uColMode" for i in inputs)
    return has_color and not has_colmode


def usable_shaders(shader_dir: str | Path) -> list[Path]:
    """Every .fs in `shader_dir` that this generator can safely place.

    Skips macOS AppleDouble `._` sidecar files (the XLIGHTS volume is full of
    them and xLights' own --shadertranslate reports every one as a parse
    failure), any shader needing an input image, and any shader with an
    uncontrolled colour input (see `has_uncontrolled_color`) -- safe alone,
    risky once shared across canvases, so kept out of the default pool.
    """
    out = []
    for p in sorted(Path(shader_dir).glob("*.fs")):
        if p.name.startswith("._"):
            continue
        try:
            shader_settings(p)
        except (ShaderUnusable, json.JSONDecodeError):
            continue
        if has_uncontrolled_color(p):
            continue
        out.append(p)
    return out


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/XLIGHTS/XLights 2026/Shaders"
    for p in usable_shaders(d):
        s = shader_settings(p)
        params = [k for k in s if "SHADERXYZZY" in k]
        print(f"{p.name}: {len(params)} params -> {', '.join(sorted(params)[:5])}")
