"""Render every shader on a real matrix and MEASURE what came out.

    python custom_tools/shader_smoke_test.py                 # all shaders
    python custom_tools/shader_smoke_test.py "Ink Bloom" ...  # just these

Writing a shader is easy; knowing it works is not. Three separate failure
modes have shipped past a careful read of the source:

  * a GLSL compile error (a Python-style `#` comment, a ternary on the float
    returned by step()) -- xLights renders the prop BLACK and says nothing
  * a shader that compiles but sits at one value -- renders as a flat wash
    with no motion, indistinguishable in the .xsq from a good one
  * a shader that pins near full white -- reads as a blown-out flat panel on
    the house and drags the whole song's lit fraction up

`xLights --shadertranslate` looked like the answer but scans a location this
show folder doesn't populate ("0 .fs found"). So do what has worked every
other time here: render it and read the pixels back.

Each shader gets its own time slot on one large matrix. Then, per slot:

    lit    mean fraction of channels above threshold -- 0.00 means it did not
           compile (or is genuinely black); above ~0.90 means it is a wash
    var    spatial standard deviation within a frame -- near 0 means a flat
           panel with no texture, whatever the source claims to draw
    motion mean absolute change between frames 0.5s apart -- near 0 means a
           still image; the `rate` input is doing nothing
    flick  largest frame-to-frame swing in WHOLE-PANEL mean brightness. This
           is the standing no-strobe rule made measurable. It has to be the
           panel MEAN, not per-channel change: a shader with bands sliding
           fast across it has a large per-channel delta and is perfectly
           calm to look at, while a shader that dims and brightens the whole
           canvas 25ms at a time is a strobe even if every pixel moves a
           little. Oil Film measures 0.038 per-channel and 0.010 panel-level;
           only the second number is the one that hurts anyone.

A shader passes only if all four are in range. That is the whole point: the
verdict comes from the render, never from reading the code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import channel_map  # noqa: E402
from fseq_reader import read_fseq  # noqa: E402
from isf_shader import ShaderUnusable, shader_settings  # noqa: E402
from xlights_mcp.xlights.palettes import ColorPalette  # noqa: E402
from xlights_mcp.xlights.show import load_show_config  # noqa: E402
from xlights_mcp.xlights.xsq_writer import (EffectPlacement,  # noqa: E402
                                            SequenceSpec, write_xsq)

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
SHADER_DIR = SHOW_PATH / "Shaders"
XLIGHTS = Path("/Applications/xLights.app/Contents/MacOS/xLights")
NAME = "_shader_smoke"
# Garage Roof is the largest canvas in the show (18576 channels) and is a
# plain 2D matrix, so a shader lands on it undistorted -- unlike Fence Matrix
# 1, which is rotated front-to-back.
TEST_MODEL = "Garage Roof"
SLOT_MS = 8000
# White so the measurement sees the shader's own greyscale output rather than
# a palette's colour choices. Real use recolours from the song palette.
WHITE = ColorPalette(colors=["#FFFFFF"], active_colors=[1])


def build(shaders: list[Path], out: Path, media: Path) -> list[Path]:
    effects, used = [], []
    for i, fs in enumerate(shaders):
        try:
            settings = shader_settings(fs, speed=400)
        except (ShaderUnusable, ValueError) as e:
            print(f"  skip {fs.stem}: {e}")
            continue
        s = i * SLOT_MS
        effects.append(EffectPlacement(
            model_name=TEST_MODEL, layer=0, effect_name="Shader",
            start_time_ms=s, end_time_ms=s + SLOT_MS - 200,
            settings=settings, palette=WHITE))
        used.append(fs)
    spec = SequenceSpec(
        song_title=NAME, media_file=str(media),
        duration_ms=len(used) * SLOT_MS + 2000, timing_ms=25,
        palettes=[WHITE], effects=effects, timing_tracks=[])
    write_xsq(spec, load_show_config(SHOW_PATH), out)
    return used


def measure(fseq: Path, used: list[Path]) -> list[dict]:
    f = read_fseq(fseq, frame_stride=1)
    rng = channel_map.build()["models"][TEST_MODEL]
    lo, hi = rng["first"], rng["last"] + 1

    def frame_at(ms: float) -> np.ndarray:
        idx = min(int(ms / f.step_time_ms), len(f) - 1)
        return np.frombuffer(f.frame(idx), dtype=np.uint8)[lo:hi].astype(np.float32)

    rows = []
    for i, fs in enumerate(used):
        base = i * SLOT_MS
        # Sample across the slot, avoiding the fade at either end.
        samples = [frame_at(base + off) for off in (1200, 2700, 4200, 5700)]
        lit = float(np.mean([(s > 12).mean() for s in samples]))
        var = float(np.mean([s.std() for s in samples])) / 255.0
        motion = float(np.mean([np.abs(frame_at(base + o + 500) - frame_at(base + o)).mean()
                                for o in (1200, 2700, 4200)])) / 255.0
        # 1.5s of consecutive frames, tracking whole-panel mean brightness.
        levels = [frame_at(base + 2000 + k * f.step_time_ms).mean()
                  for k in range(int(1500 / f.step_time_ms))]
        flick = max(abs(levels[k + 1] - levels[k])
                    for k in range(len(levels) - 1)) / 255.0
        why = []
        if lit < 0.02:
            why.append("BLACK (compile failure?)")
        elif lit > 0.93:
            why.append("washed out")
        if var < 0.03:
            why.append("flat, no texture")
        if motion < 0.004:
            why.append("still, rate does nothing")
        if flick > 0.06:
            why.append("STROBES the whole panel")
        rows.append({"name": fs.stem, "lit": lit, "var": var,
                     "motion": motion, "flick": flick, "why": why})
    return rows


def main() -> None:
    wanted = sys.argv[1:]
    shaders = sorted(SHADER_DIR.glob("*.fs"))
    if wanted:
        shaders = [p for p in shaders if p.stem in wanted]
        missing = set(wanted) - {p.stem for p in shaders}
        if missing:
            print(f"not found: {sorted(missing)}")
    media = next(SHOW_PATH.glob("*.mp3"))
    out = SHOW_PATH / f"{NAME}.xsq"
    print(f"building {len(shaders)} shaders x {SLOT_MS/1000:.0f}s on {TEST_MODEL} ...")
    used = build(shaders, out, media)
    print("rendering ...")
    subprocess.run([str(XLIGHTS), "-s", str(SHOW_PATH), "-r", str(out)],
                   capture_output=True)
    fseq = SHOW_PATH / f"{NAME}.fseq"
    if not fseq.exists():
        print("render produced no fseq")
        return
    rows = measure(fseq, used)
    print(f"\n{'shader':22} {'lit':>6} {'var':>6} {'motion':>7} {'flick':>7}  verdict")
    bad = 0
    for r in sorted(rows, key=lambda r: (not r["why"], r["name"])):
        verdict = "ok" if not r["why"] else "; ".join(r["why"])
        bad += 1 if r["why"] else 0
        print(f"{r['name'][:22]:22} {r['lit']:6.2f} {r['var']:6.3f} "
              f"{r['motion']:7.4f} {r['flick']:7.4f}  {verdict}")
    print(f"\n{len(rows) - bad}/{len(rows)} pass")


if __name__ == "__main__":
    main()
