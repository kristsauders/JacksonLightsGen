"""Group-first, sparsely-staged xLights sequence generator.

This is a ground-up rewrite of the earlier per-model generator, rebuilt to
match the patterns measured from Krists's real finished sequences (see the
`xlights-human-sequence-patterns` memory for the full measured spec):

  * Effects target a CAST of ~50 model GROUPS / whole-area container models,
    not all 235 individual models. A group element renders as one canvas, so
    a chase spans the physical group instead of duplicating per member.
  * Buffer style is chosen deliberately: `Per Preview` to span a group as one
    canvas, `As Pixel` to step prop-to-prop across scattered props, never
    `Per Model *` (which is what caused "duplicated instead of spanning").
  * Staging is SPARSE and dynamic: 15-55% of the cast lit at a time depending
    on section energy, with roles rotating so different props lead in
    different sections. Human sequences average 11-26% lit.
  * Effects are SHORT: most 150ms-1s, chases at rotations=1, placed on beats.
    Long effects with high rotation counts read as fast blur, not rhythm.
  * Repeated section types escalate rather than repeating identically.
  * Signature musical cues (arpeggio chases, fast upward sweeps, synth stabs,
    pre-drop implosions, cymbal detail) are layered on top.

Usage: python3 generate_song.py [SongName] [--out "Name (v6).xsq"]
Requires run_analysis.py + detect_cues.py to have been run for the song.
"""
from __future__ import annotations

import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/src")

from xlights_mcp.audio.analyzer import SongAnalysis
from xlights_mcp.xlights.palettes import ColorPalette
from xlights_mcp.xlights.show import load_show_config
from xlights_mcp.xlights.xsq_writer import EffectPlacement, SequenceSpec, write_xsq

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
SONG_NAME = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "Gooey"
GEN_DIR = SHOW_PATH / ".generation" / SONG_NAME


def _peek_tempo(default: float = 120.0) -> float:
    """Tempo, read from the first bytes of analysis.json.

    Needed at MODULE level (register_shaders runs at import time) while the
    full SongAnalysis is only parsed inside main(); analysis.json is several
    megabytes, so re-parsing it just for one number is not worth it. `tempo`
    is the second field of the file, so a short head read always finds it."""
    try:
        head = (GEN_DIR / "analysis.json").read_bytes()[:2048].decode(
            "utf-8", "ignore")
        m = re.search(r'"tempo"\s*:\s*([0-9.]+)', head)
        if m:
            return float(m.group(1))
    except OSError:
        pass
    return default


SONG_TEMPO = _peek_tempo()
MP3_PATH = SHOW_PATH / f"{SONG_NAME}.mp3"
# Auto-increment rather than a hardcoded default: the default sat at "(v8)"
# for months because every run passed --out, so the one run that forgot
# silently overwrote a v8 that was eight versions old. The next free number
# is always safe and always the newest.
def _next_version() -> int:
    import re as _re
    n = 0
    for _f in SHOW_PATH.glob(f"{SONG_NAME} (v*).xsq"):
        _m = _re.search(r"\(v(\d+)\)\.xsq$", _f.name)
        if _m:
            n = max(n, int(_m.group(1)))
    return n + 1


OUT_NAME = f"{SONG_NAME} (v{_next_version()}).xsq"
for i, a in enumerate(sys.argv):
    if a == "--out" and i + 1 < len(sys.argv):
        OUT_NAME = sys.argv[i + 1]
OUTPUT_PATH = SHOW_PATH / OUT_NAME
CUESHEET_PATH = GEN_DIR / "cuesheet.md"

# Layers. Keeping roles on distinct layers stops them fighting for the same slot.
# xLights render frame period for this show, in ms (40fps). Every rhythmic
# placement is snapped to this grid -- see add() for the measurement that
# motivated it.
FRAME_MS = 25

L_BASE = 0      # outline / texture -- the quiet presence layer
L_MOTION = 1    # feature motion (chases, waves, curtains)
L_PULSE = 2     # rhythmic hits
L_SIG = 3       # signature musical cues

# ---------------------------------------------------------------------------
# Validated effect settings. Every key below was copied from a real finished
# sequence's EffectDB -- do NOT invent E_/B_ keys, xLights silently ignores
# unknown ones (this is how the Chase_Rotations bug hid for so long).
# ---------------------------------------------------------------------------

SPAN = {"B_CHOICE_BufferStyle": "Per Preview"}       # group -> one canvas
PERPROP = {"B_CHOICE_BufferStyle": "As Pixel"}       # each model = one pixel
# Every member runs the effect INDEPENDENTLY, inside its own geometry.
# Essential for groups whose members are discrete standalone objects rather
# than parts of one surface -- direct feedback on the stars: "they have group
# effects that don't look good, they should mostly have effects that render
# per prop, otherwise the left right across the whole group of stars just
# looks like flashing and strobing." A group-spanning chase over 12 physically
# separated stars can only express itself by lighting them one after another,
# which at rhythm speed is indistinguishable from strobing.
PERMODEL = {"B_CHOICE_BufferStyle": "Per Model Default"}
# For a tunnel/tree modelled as one multi-string CUSTOM model, the default
# buffer lays the strings out end-to-end, so a 2D pattern renders as a smear
# of lines instead of the real image. Stacking the strings reconstitutes the
# actual surface.
#
# Direct feedback: "when applying cone effects to the tunnel matrix, I think
# we need to change the render mode to stacked or something like that if I
# remember right, otherwise you just see lines instead of the true pattern.
# The shaders etc work fine but not many of the built in effects." Confirmed
# against his own sequences rather than guessed -- Pink Pony Club drives the
# Tunnel with Morph/Wave/Spirals/On/Color Wash ALL on `Vertical Stack` (69,
# 64, 30, 20, 9 uses). Shaders are exempt because they render their own
# coordinate space and already look right, which matches what he observed.
VSTACK = {"B_CHOICE_BufferStyle": "Vertical Stack"}
# Treat each member as ONE STRAND and run the effect along it.
#
# This is the idiom for physically vertical props, and it took three failed
# attempts to find because the failure mode was silent. The "Vertical Strip *"
# props are `DisplayAs="Single Line"` -- 1D strands. A `Bars` effect with
# `direction=up` needs a 2D buffer to have an "up" at all, so on a 1D strand
# it degenerates and produces no vertical travel, which is precisely the
# repeated report of "still no up down movement at all on verticals."
# A chase along the strand, by contrast, physically IS up-and-down, because
# the strand itself is mounted vertically.
#
# Recovered from Gooey v5 -- the version Krists remembered as having "great
# movement on all the window outlines and verticals": it drove each
# `Vertical Strip *` MODEL with `SingleStrand` + `Per Model Single Line` +
# "Bounce from Left", 253 times each. `Per Model Single Line` also appears
# 50x across his own finished sequences. Looking at the version he liked was
# far faster than reasoning about geometry.
PMSL = {"B_CHOICE_BufferStyle": "Per Model Single Line"}

# LEARNED FROM KRISTS'S OWN SEQUENCES (the 2025 40fps set), by measuring what
# he actually puts on outlines, flakes, mini trees and the window matrices.
# Across 17 of his sequences the SingleStrand vocabulary on outline groups is:
#
#   chase type   Bounce from Left 2693, Left-Right 1536, Right-Left 1301,
#                From Middle 447, To Middle 398, Dual Chase 109
#   buffer       Per Model Per Preview 2624, none 1607, Per Model Default 717,
#                Overlay - Scaled 399, Per Model Single Line 395
#   fade         From Head 3419, None 3062, Head and Tail 24
#
# Head and Tail was this generator's DEFAULT and therefore 100% of its chases,
# against 0.4% of his. A head fade leaves a trail behind the dot instead of
# tapering both ends, which is what makes a chase read as travelling rather
# than as a lit segment sliding.
#   rotations    1.0 (4694), 0.4 (1302), 4 (71)
#   FX           Fireworks 1D 3321, none 3097
#
# Mine, for comparison, on the same prop families: Left-Right/Right-Left only,
# Per Model Default on 96%, Head and Tail on 100%, and no SingleStrand FX at
# all. Three of his five most-used settings did not exist in this generator.
PMPP = {"B_CHOICE_BufferStyle": "Per Model Per Preview"}
OVERLAY = {"B_CHOICE_BufferStyle": "Overlay - Scaled"}


def _chase(direction: str, fade: str = "From Head", span: bool = True,
           chases: str = "1", rotations: str = "1.0", fx: str | None = None,
           buffer: dict | None = None) -> dict:
    s = {
        "E_CHECKBOX_Chase_Group_All": "0",
        "E_CHOICE_Chase_Type1": direction,
        "E_CHOICE_Fade_Type": fade,
        "E_CHOICE_SingleStrand_Colors": "Palette",
        "E_NOTEBOOK_SSEFFECT_TYPE": "Chase",
        # `chases` > 1 puts several dots on the strand at once. On a long run
        # (the C9 gutter line) a SINGLE dot has to cross the entire length
        # within the effect's duration, which at rhythm speed is far too fast
        # to look good -- direct feedback: "it should move side to side but
        # not across the entire line because it's a long line and it just
        # doesn't look good going that fast. Multiple spots or dots or lines
        # can all move side to side instead of a single one going all the way
        # across." With N dots evenly spaced, each only travels 1/N of the
        # line to hand over to the next, so the same musical timing produces
        # a much calmer apparent speed.
        "E_SLIDER_Number_Chases": chases,
        # Rotations is the chase's SPEED: how many times it crosses within its
        # own span. Krists uses 1.0 for a plain pass and 0.4 for a drift that
        # never reaches the end -- a slow reveal rather than a sweep -- and 4
        # for a genuine flurry. Only 1.0 existed here.
        "E_TEXTCTRL_Chase_Rotations": rotations,
    }
    if fx:
        # SingleStrand's own FX layer, on more than half of his chases. It puts
        # sparkle INSIDE the chase, so one effect carries both the travel and
        # the detail, where this generator would have stacked two.
        s["E_CHOICE_SingleStrand_FX"] = fx
    if buffer is not None:
        s.update(buffer)
    elif span:
        s.update(SPAN)
    return s


# RETIRED. The xLights "Wave" effect -- the jagged line that travels side to
# side across a matrix -- is no longer used anywhere. Krists flagged it as
# over-repetitive twice, then: "I don't like the jagged line wave effect we
# still keep using on the matrices sometimes that goes side to side. Let's just
# remove that one completely." Kept only so the recipe isn't lost.
def _wave(direction: str) -> dict:
    return {
        "B_CHOICE_BufferStyle": "Default", "B_CHOICE_BufferTransform": "None",
        "E_CHECKBOX_Mirror_Wave": "0", "E_CHOICE_Fill_Colors": "Palette",
        "E_CHOICE_Wave_Direction": direction, "E_CHOICE_Wave_Type": "Triangle",
        "E_SLIDER_Number_Waves": "1202", "E_SLIDER_Thickness_Percentage": "40",
        "E_SLIDER_Wave_Height": "50", "E_SLIDER_Wave_YOffset": "0",
        "E_TEXTCTRL_Wave_Speed": "50.00", "T_CHOICE_LayerMethod": "1 reveals 2",
    }


def _bars(direction: str, cycles: str = "1", buffer: dict | None = None,
          bars: str = "1", highlight: str = "0") -> dict:
    # NOTE the default buffer here is PERPROP ("As Pixel"), which makes each
    # MODEL a single pixel -- so the bar travels ACROSS the group's members,
    # not along each member's own geometry. That's right for a cross-prop
    # sweep, and completely wrong for "move up and down inside each vertical
    # strip": on a group of vertical strips, As Pixel + direction=up produces
    # motion across the strips, i.e. side to side. That baked-in default is
    # why vertical props still had no vertical movement after the previous
    # round nominally gave them `sweep_up`/`sweep_down`. Pass `buffer` to
    # override -- see `bars_up_in`/`bars_down_in`.
    return {
        **(buffer if buffer is not None else PERPROP),
        "E_CHECKBOX_Bars_3D": "0", "E_CHECKBOX_Bars_Gradient": "0",
        "E_CHECKBOX_Bars_Highlight": highlight, "E_CHOICE_Bars_Direction": direction,
        "E_SLIDER_Bars_BarCount": bars, "E_TEXTCTRL_Bars_Cycles": cycles,
    }


# Bars, retired at Krists's request after it turned up "everywhere" and read as
# stuttering when it changed orientation. Mapped to the nearest non-bar gesture
# rather than deleted outright, so the passes that reached for a directional
# sweep still get one.
_BARS_SUBSTITUTE = {
    "sweep_right": "chase_lr", "sweep_left": "chase_rl",
    "sweep_up": "strand_up", "sweep_down": "strand_down",
    "bars_up_in": "strand_up", "bars_down_in": "strand_down",
    "bars_alt_up": "chase_bounce_pp", "bars_alt_down": "chase_bounce_pp",
    "bars_expand": "chase_mid_out", "bars_compress": "chase_mid_in",
    "bars_hexpand": "chase_mid_out_pp", "bars_hcompress": "chase_mid_in_pp",
    "bars_three": "shockwave", "bars_two_up": "strand_up",
    # SPIRALS GOES TOO, and it was the real offender. On a matrix it draws
    # diagonal bands, and the Saw Tooth rotation curve added last round swept
    # those bands back and forth -- so what Krists was describing as "diagonal
    # bars changing orientation", "stuttering when they change orientation"
    # and "for just one second" was one effect: a 1.44s Spirals on the full
    # matrix at This Version of You 0:58, with a reversing rotation on top.
    # He had already asked for "less built-in effects like the spiral and
    # wave" in an earlier round.
    "spirals": "chase_mid_out",
}


EFFECTS: dict[str, tuple[str, dict]] = {
    # --- feature motion (span the group as one canvas) ---
    "chase_lr":       ("SingleStrand", _chase("Left-Right")),
    "chase_rl":       ("SingleStrand", _chase("Right-Left")),
    "chase_mid_out":  ("SingleStrand", _chase("From Middle")),
    "chase_mid_in":   ("SingleStrand", _chase("To Middle")),
    "chase_bounce":   ("SingleStrand", _chase("Bounce from Left")),
    # --- the human vocabulary, added 2026-08-29 -------------------------------
    # Same gestures Krists reaches for, with the settings he actually uses:
    # per-model-per-preview buffers so each prop performs the chase in its own
    # geometry while keeping its place on the facade, head-fades instead of
    # head-and-tail, and the slow 0.4-rotation drift that reads as a reveal.
    "chase_bounce_pp": ("SingleStrand", _chase("Bounce from Left", fade="From Head",
                                               buffer=PMPP)),
    "chase_bounce_drift": ("SingleStrand", _chase("Bounce from Left", fade="From Head",
                                                  rotations="0.4", buffer=PMPP)),
    "chase_lr_pp":    ("SingleStrand", _chase("Left-Right", fade="From Head",
                                              buffer=PMPP)),
    "chase_rl_pp":    ("SingleStrand", _chase("Right-Left", fade="From Head",
                                              buffer=PMPP)),
    "chase_mid_out_pp": ("SingleStrand", _chase("From Middle", fade="From Head",
                                                buffer=PMPP)),
    "chase_mid_in_pp": ("SingleStrand", _chase("To Middle", fade="From Head",
                                               buffer=PMPP)),
    "chase_dual":     ("SingleStrand", _chase("Dual Chase", fade="From Head",
                                              buffer=PMPP)),
    # Sparkle carried INSIDE the chase rather than stacked on top of it.
    "chase_spark":    ("SingleStrand", _chase("Bounce from Left", fade="None",
                                              fx="Fireworks 1D", buffer=PMPP)),
    "chase_spark_lr": ("SingleStrand", _chase("Left-Right", fade="From Head",
                                              fx="Fireworks 1D", buffer=PMPP)),
    # A flurry: four crossings inside one span, for a fill or a drum roll.
    "chase_flurry":   ("SingleStrand", _chase("Bounce from Left", fade="From Head",
                                              rotations="4", buffer=PMPP)),
    # Multi-dot variants for long runs (C9 gutter line, long outlines) -- see
    # the `chases` note in _chase().
    "chase_dots_lr":  ("SingleStrand", _chase("Left-Right", chases="4")),
    "chase_dots_rl":  ("SingleStrand", _chase("Right-Left", chases="4")),
    # Two-dot variants for SHORTER runs. "The chases are better for the c9s
    # since they cover large gutters but the chases on the window outlines and
    # verticals etc don't need that many because they're smaller." Dot count
    # has to scale with the physical length of the run, not be one constant.
    "chase_pair_lr":  ("SingleStrand", _chase("Left-Right", chases="2")),
    "chase_pair_rl":  ("SingleStrand", _chase("Right-Left", chases="2")),
    # --- cross-prop stepping (each model = 1 pixel) ---
    "sweep_right":    ("Bars", _bars("Right")),
    "sweep_left":     ("Bars", _bars("Left")),
    "sweep_up":       ("Bars", _bars("up")),
    "sweep_down":     ("Bars", _bars("down")),
    # Vertical travel INSIDE each member, for verticals / vertical outlines /
    # small trees. Per Model instead of As Pixel is the whole difference --
    # see the note in _bars().
    "bars_up_in":     ("Bars", _bars("up", buffer=PERMODEL)),
    "bars_down_in":   ("Bars", _bars("down", buffer=PERMODEL)),
    # --- rhythmic bars, from Krists's own sequences -------------------------
    # He uses Alternate Up/Down (37 each), which REVERSES the bar's direction
    # on every cycle -- so with the cycle rate locked to the beat, the bar
    # changes direction on the beat. That is the "moved fast and then slowed
    # down on the next beat, or the speed changed back and forth" effect, and
    # it needs both halves: the alternating direction AND a beat-derived cycle
    # count. This generator had neither.
    "bars_alt_up":    ("Bars", _bars("Alternate Up", buffer=PERMODEL)),
    "bars_alt_down":  ("Bars", _bars("Alternate Down", buffer=PERMODEL)),
    # ORIENTATION, not just direction: expand/compress open from the middle
    # outward, H-expand/H-compress do the same on the other axis. 313 uses
    # between them in his sequences against none here.
    "bars_expand":    ("Bars", _bars("expand", buffer=PERMODEL)),
    "bars_compress":  ("Bars", _bars("compress", buffer=PERMODEL)),
    "bars_hexpand":   ("Bars", _bars("H-expand", buffer=PERMODEL)),
    "bars_hcompress": ("Bars", _bars("H-compress", buffer=PERMODEL)),
    # Several bars at once, and the highlighted leading edge he uses on the
    # up/down sweeps -- a brighter front makes the direction legible.
    "bars_three":     ("Bars", _bars("expand", buffer=PERMODEL, bars="3")),
    "bars_two_up":    ("Bars", _bars("up", buffer=PERMODEL, bars="2",
                                     highlight="1")),
    # Travel ALONG each strand (physically up/down on a vertical strip).
    # `Bounce from Left` does the out-and-back in a single effect, which is
    # what v5 used and what reads as a strip pulsing up and down.
    "strand_bounce":  ("SingleStrand",
                       {**_chase("Bounce from Left", span=False), **PMSL}),
    "strand_up":      ("SingleStrand",
                       {**_chase("Left-Right", span=False), **PMSL}),
    "strand_down":    ("SingleStrand",
                       {**_chase("Right-Left", span=False), **PMSL}),
    # VARIETY THAT STAYS ON THE STRAND. The chase_*_pp keys added for outline
    # groups use Per Model Per Preview, which is right for a group of separate
    # props and WRONG for a single vertical strip: the strip's preview buffer
    # is a few cells wide, so a chase across it renders as two or three blocks
    # switching on and off. "Their up/down motion is bad now, it's just split
    # into three sections and they just flash on and off." A vertical strip has
    # to keep Per Model Single Line, where the chase runs the length of the
    # strand -- which is physically up and down. So these carry the new speeds,
    # fades and FX on the buffer that works.
    "strand_drift":   ("SingleStrand",
                       {**_chase("Bounce from Left", span=False,
                                 rotations="0.4"), **PMSL}),
    "strand_spark":   ("SingleStrand",
                       {**_chase("Bounce from Left", span=False,
                                 fade="None", fx="Fireworks 1D"), **PMSL}),
    "strand_mid_out": ("SingleStrand",
                       {**_chase("From Middle", span=False), **PMSL}),
    "strand_mid_in":  ("SingleStrand",
                       {**_chase("To Middle", span=False), **PMSL}),
    # --- linear-prop waves (icicles, verticals, C9s) ---
    # --- dramatic reveals ---
    # NOTE: no forced buffer style here -- see group_extra() below. "Per
    # Preview" (spanning a real group's members as one canvas) is only valid
    # when there are multiple physically distinct models to span; forcing it
    # onto a single Custom-type model with an irregular node layout (e.g.
    # Garage Roof, Tunnel Matrix Real -- no NumStrings at all) is exactly the
    # kind of unsupported combination that can make xLights render an error
    # state (solid red) instead of the effect.
    "curtain_close":  ("Curtain", {"E_CHECKBOX_Curtain_Repeat": "0",
                                    "E_CHOICE_Curtain_Edge": "middle",
                                    "E_CHOICE_Curtain_Effect": "close",
                                    "E_SLIDER_Curtain_Swag": "0",
                                    "E_TEXTCTRL_Curtain_Speed": "1.0"}),
    "curtain_open":   ("Curtain", {"E_CHECKBOX_Curtain_Repeat": "0",
                                    "E_CHOICE_Curtain_Edge": "middle",
                                    "E_CHOICE_Curtain_Effect": "open",
                                    "E_SLIDER_Curtain_Swag": "0",
                                    "E_TEXTCTRL_Curtain_Speed": "1.0"}),
    # --- hits / accents ---
    "shockwave":      ("Shockwave", {"B_CHOICE_BufferStyle": "Default",
                                      "E_CHECKBOX_Shockwave_Blend_Edges": "1",
                                      "E_CHECKBOX_Shockwave_Scale": "0",
                                      "E_NOTEBOOK_Shockwave": "Position",
                                      "E_SLIDER_Shockwave_Accel": "0",
                                      "E_SLIDER_Shockwave_CenterX": "50",
                                      "E_SLIDER_Shockwave_CenterY": "50",
                                      "E_SLIDER_Shockwave_Cycles": "1",
                                      "E_SLIDER_Shockwave_Start_Radius": "1",
                                      "E_SLIDER_Shockwave_End_Radius": "100",
                                      "E_SLIDER_Shockwave_Start_Width": "25",
                                      "E_SLIDER_Shockwave_End_Width": "5"}),
    "ripple_implode": ("Ripple", {"E_CHECKBOX_Ripple3D": "0",
                                   "E_CHOICE_Ripple_Movement": "Implode",
                                   "E_CHOICE_Ripple_Object_To_Draw": "Circle",
                                   "E_FILEPICKERCTRL_Ripple_SVG": "",
                                   "E_SLIDER_Ripple_Thickness": "3",
                                   "E_SLIDER_Ripple_XC": "0", "E_SLIDER_Ripple_YC": "0",
                                   "E_TEXTCTRL_Ripple_Cycles": "1.0"}),
    "ripple_explode": ("Ripple", {"E_CHECKBOX_Ripple3D": "0",
                                   "E_CHOICE_Ripple_Movement": "Explode",
                                   "E_CHOICE_Ripple_Object_To_Draw": "Circle",
                                   "E_FILEPICKERCTRL_Ripple_SVG": "",
                                   "E_SLIDER_Ripple_Thickness": "3",
                                   "E_SLIDER_Ripple_XC": "0", "E_SLIDER_Ripple_YC": "0",
                                   "E_TEXTCTRL_Ripple_Cycles": "1.0"}),
    "meteors_icicle": ("Meteors", {"B_CHOICE_BufferStyle": "Vertical Stack",
                                    "E_CHECKBOX_Meteors_UseMusic": "0",
                                    "E_CHOICE_Meteors_Effect": "Icicles",
                                    "E_CHOICE_Meteors_Type": "Palette",
                                    "E_SLIDER_Meteors_Count": "3",
                                    "E_SLIDER_Meteors_Length": "25",
                                    "E_SLIDER_Meteors_Speed": "10",
                                    "E_SLIDER_Meteors_Swirl_Intensity": "0"}),
    "meteors_down":   ("Meteors", {"E_CHECKBOX_Meteors_UseMusic": "0",
                                    "E_CHOICE_Meteors_Effect": "Down",
                                    "E_CHOICE_Meteors_Type": "Palette",
                                    "E_SLIDER_Meteors_Count": "8",
                                    "E_SLIDER_Meteors_Length": "40",
                                    "E_SLIDER_Meteors_Speed": "18",
                                    "E_SLIDER_Meteors_Swirl_Intensity": "0"}),
    # --- large-canvas imagery (settings copied from real sequences) ---
    "spirals":        ("Spirals", {"E_CHECKBOX_Spirals_3D": "1", "E_CHECKBOX_Spirals_Blend": "0",
                                    "E_CHECKBOX_Spirals_Grow": "0", "E_CHECKBOX_Spirals_Shrink": "0",
                                    "E_SLIDER_Spirals_Count": "1", "E_SLIDER_Spirals_Thickness": "50",
                                    "E_SLIDER_Spirals_Rotation": "20",
                                    "E_TEXTCTRL_Spirals_Movement": "1.0"}),
    "pinwheel":       ("Pinwheel", {"E_CHECKBOX_Pinwheel_Rotation": "1",
                                     "E_CHOICE_Pinwheel_3D": "None",
                                     "E_CHOICE_Pinwheel_Style": "New Render Method",
                                     "E_SLIDER_Pinwheel_ArmSize": "100",
                                     "E_SLIDER_Pinwheel_Arms": "4",
                                     "E_SLIDER_Pinwheel_Offset": "0",
                                     "E_SLIDER_Pinwheel_Speed": "10",
                                     "E_SLIDER_Pinwheel_Thickness": "22",
                                     "E_SLIDER_Pinwheel_Twist": "110"}),
    "galaxy":         ("Galaxy", {"E_CHECKBOX_Galaxy_Blend_Edges": "1",
                                   "E_CHECKBOX_Galaxy_Inward": "0",
                                   "E_CHECKBOX_Galaxy_Reverse": "0",
                                   "E_CHECKBOX_Galaxy_Scale": "0",
                                   "E_NOTEBOOK_Galaxy": "Start",
                                   "E_SLIDER_Galaxy_Accel": "0",
                                   "E_SLIDER_Galaxy_CenterX": "50",
                                   "E_SLIDER_Galaxy_CenterY": "50",
                                   "E_SLIDER_Galaxy_Duration": "20",
                                   "E_SLIDER_Galaxy_End_Radius": "10",
                                   "E_SLIDER_Galaxy_End_Width": "5",
                                   "E_SLIDER_Galaxy_Revolutions": "1440",
                                   "E_SLIDER_Galaxy_Start_Angle": "0",
                                   "E_SLIDER_Galaxy_Start_Radius": "40",
                                   "E_SLIDER_Galaxy_Start_Width": "20"}),
    "butterfly":      ("Butterfly", {"E_CHOICE_Butterfly_Colors": "Palette",
                                      "E_CHOICE_Butterfly_Direction": "Normal",
                                      "E_SLIDER_Butterfly_Chunks": "1",
                                      "E_SLIDER_Butterfly_Skip": "2",
                                      "E_SLIDER_Butterfly_Speed": "10",
                                      "E_SLIDER_Butterfly_Style": "1"}),
    # Organic flowing color -- native, file-free psychedelic/fabric-like
    # texture. Settings copied verbatim from a real sequence, no ValueCurve
    # extras, so this is the maximally safe minimal Plasma.
    "plasma_organic": ("Plasma", {"E_CHOICE_Plasma_Color": "Normal",
                                   "E_SLIDER_Plasma_Line_Density": "1",
                                   "E_SLIDER_Plasma_Speed": "10",
                                   "E_SLIDER_Plasma_Style": "2"}),
    # --- real GLSL shaders ---
    # Found on 2026-08-09: a genuine, complete shader pack already sitting on
    # this volume at "/Volumes/XLIGHTS/XLights 2025/Shaders/" (proper ISF-
    # format headers, credited authors, real working GLSL -- not something I
    # wrote). Parameter structure (E_0FILEPICKERCTRL_IFS + E_*_SHADERXYZZY_*
    # + E_SLIDER_Shader_Speed + E_TEXTCTRL_Shader_*) copied verbatim from a
    # real Shader effect instance in Silent Night, minus its ValueCurve zoom
    # animation (not safely reproducible without seeing the exact curve
    # syntax validated elsewhere -- static Zoom=0 instead).
    # Shader entries are NOT hardcoded here -- see register_shaders() below.
    # Every xLights shader parameter is named after that specific .fs file's
    # own declared ISF input, so a settings block copied from one shader is
    # meaningless (silently ignored) on another.
    # --- quiet presence ---
    "twinkle_soft":   ("Twinkle", {"E_CHECKBOX_Twinkle_ReRandom": "0",
                                    "E_CHECKBOX_Twinkle_Strobe": "0",
                                    "E_CHOICE_Twinkle_Style": "New Render Method",
                                    "E_SLIDER_Twinkle_Count": "3",
                                    "E_SLIDER_Twinkle_Steps": "30",
                                    "T_TEXTCTRL_Fadein": "0.6", "T_TEXTCTRL_Fadeout": "0.8"}),
    "shimmer":        ("Shimmer", {"E_CHECKBOX_PRE_2017_7": "0",
                                    "E_CHECKBOX_Shimmer_Use_All_Colors": "0",
                                    "E_SLIDER_Shimmer_Duty_Factor": "74"}),
    "wash":           ("Color Wash", {"B_CHOICE_BufferStyle": "Default",
                                       "E_CHECKBOX_ColorWash_HFade": "1",
                                       "T_TEXTCTRL_Fadein": "0.5",
                                       "T_TEXTCTRL_Fadeout": "0.5"}),
    # --- shapes learned from Krists's sequences ------------------------------
    # Ripple as he uses it: a POLYGON with a point count, imploding or
    # exploding, rather than the plain circle this generator had. Six points on
    # a snowflake is the snowflake's own shape collapsing into itself, which is
    # why it reads as belonging to the prop instead of being drawn on top of
    # it. 246 uses on his window matrices alone.
    "ripple_hex_in":  ("Ripple", {"E_CHOICE_Ripple_Object_To_Draw": "Polygon",
                                  "E_SLIDER_RIPPLE_POINTS": "6",
                                  "E_CHOICE_Ripple_Movement": "Implode",
                                  "E_CHOICE_Ripple_Draw_Style": "Old",
                                  "E_CHECKBOX_Ripple3D": "0",
                                  "E_SLIDER_Ripple_Thickness": "12",
                                  "B_CHOICE_BufferStyle": "Per Model Default"}),
    "ripple_hex_out": ("Ripple", {"E_CHOICE_Ripple_Object_To_Draw": "Polygon",
                                  "E_SLIDER_RIPPLE_POINTS": "6",
                                  "E_CHOICE_Ripple_Movement": "Explode",
                                  "E_CHOICE_Ripple_Draw_Style": "Old",
                                  "E_CHECKBOX_Ripple3D": "0",
                                  "E_SLIDER_Ripple_Thickness": "12",
                                  "B_CHOICE_BufferStyle": "Per Model Default"}),
    # Lines radiating outward in 3D -- his second-most-used ripple, and a
    # different look entirely: rays rather than rings.
    "ripple_rays":    ("Ripple", {"E_CHOICE_Ripple_Object_To_Draw": "Circle",
                                  "E_CHOICE_Ripple_Draw_Style": "Lines Outward",
                                  "E_CHOICE_Ripple_Movement": "Explode",
                                  "E_CHECKBOX_Ripple3D": "1",
                                  "E_SLIDER_Ripple_Thickness": "8",
                                  "B_CHOICE_BufferStyle": "Per Model Default"}),
    # Pinwheel as a SWEEP rather than a spinning fan -- the arm smears round
    # instead of strobing past, which is why his version survives on a matrix
    # where the plain fan was "too basic".
    # Spinners with the arm counts and speeds Krists actually uses (arms 3-6,
    # speed 2/5/7/10 against this generator's single fixed setting), and a
    # twisted variant. These are the "spinning and wobbling kinds of effects"
    # for the showstopper, EFL, Rosa Grande and Baby Grand Illusion props.
    "pinwheel_4":     ("Pinwheel", {"E_SLIDER_Pinwheel_Arms": "4",
                                    "E_SLIDER_Pinwheel_Speed": "7",
                                    "E_CHECKBOX_Pinwheel_Rotation": "1",
                                    "E_SLIDER_Pinwheel_Thickness": "20",
                                    "E_CHOICE_Pinwheel_Style": "New Render Method",
                                    "B_CHOICE_BufferStyle": "Per Model Default"}),
    "pinwheel_6_slow": ("Pinwheel", {"E_SLIDER_Pinwheel_Arms": "6",
                                     "E_SLIDER_Pinwheel_Speed": "2",
                                     "E_CHECKBOX_Pinwheel_Rotation": "1",
                                     "E_SLIDER_Pinwheel_Thickness": "30",
                                     "E_CHOICE_Pinwheel_Style": "New Render Method",
                                     "B_CHOICE_BufferStyle": "Per Model Default"}),
    "pinwheel_twist": ("Pinwheel", {"E_SLIDER_Pinwheel_Arms": "3",
                                    "E_SLIDER_Pinwheel_Speed": "10",
                                    "E_SLIDER_Pinwheel_Twist": "40",
                                    "E_CHECKBOX_Pinwheel_Rotation": "1",
                                    "E_SLIDER_Pinwheel_Thickness": "10",
                                    "E_CHOICE_Pinwheel_Style": "New Render Method",
                                    "B_CHOICE_BufferStyle": "Per Model Default"}),
    "pinwheel_sweep": ("Pinwheel", {"E_CHOICE_Pinwheel_3D": "Sweep",
                                    "E_CHOICE_Pinwheel_Style": "New Render Method",
                                    "E_CHECKBOX_Pinwheel_Rotation": "1",
                                    "E_SLIDER_Pinwheel_ArmSize": "100",
                                    "E_SLIDER_Pinwheel_Arms": "3",
                                    "E_SLIDER_PinwheelXC": "0",
                                    "E_SLIDER_PinwheelYC": "0",
                                    "B_CHOICE_BufferStyle": "Per Model Default"}),
    # A curtain closing from the MIDDLE across a whole group as one canvas --
    # 746 uses on his mini trees. The group becomes one object that shuts,
    # which no per-model effect can express.
    "curtain_mid":    ("Curtain", {"E_CHOICE_Curtain_Edge": "middle",
                                   "E_CHOICE_Curtain_Effect": "close",
                                   "E_CHECKBOX_Curtain_Repeat": "0",
                                   "E_SLIDER_Curtain_Swag": "0",
                                   "E_TEXTCTRL_Curtain_Speed": "1.0",
                                   "B_CHOICE_BufferStyle": "Per Preview"}),
    "curtain_mid_open": ("Curtain", {"E_CHOICE_Curtain_Edge": "middle",
                                     "E_CHOICE_Curtain_Effect": "open",
                                     "E_CHECKBOX_Curtain_Repeat": "0",
                                     "E_SLIDER_Curtain_Swag": "0",
                                     "E_TEXTCTRL_Curtain_Speed": "1.0",
                                     "B_CHOICE_BufferStyle": "Per Preview"}),
    # --- hanging spheres ------------------------------------------------
    # Seven spheres, 240 nodes each as 8 strands of 30 -- enough resolution for
    # travel ALONG the sphere, not just around it. "You can do up and down
    # movements since there are enough lights, in addition to the rotation
    # you're doing now. You can also do sparkling... you could use different
    # colors for alternating strands or something."
    #
    # Fill is the up/down: it floods the buffer from one edge to the other, so
    # on a sphere it reads as the light rising or draining. Krists uses it 143
    # times in his own sequences and this generator had no Fill at all.
    # Fill_Position needs a VALUE CURVE or the fill just lands and sits there:
    # measured at a constant 158 across its whole span, which is a lit prop,
    # not a rising one. The curve is what makes the level travel.
    "fill_up":        ("Fill", {"E_CHOICE_Fill_Direction": "Up",
                                "E_TEXTCTRL_Fill_Position": "0",
                                "E_VALUECURVE_Fill_Position":
                                "Active=TRUE|Id=ID_VALUECURVE_Fill_Position|"
                                "Type=Ramp|Min=0.00|Max=100.00|P1=0.00|"
                                "P2=100.00|RV=TRUE|",
                                "E_CHECKBOX_Fill_Wrap": "0",
                                "B_CHOICE_BufferStyle": "Per Model Default"}),
    "fill_down":      ("Fill", {"E_CHOICE_Fill_Direction": "Down",
                                "E_TEXTCTRL_Fill_Position": "0",
                                "E_VALUECURVE_Fill_Position":
                                "Active=TRUE|Id=ID_VALUECURVE_Fill_Position|"
                                "Type=Ramp|Min=0.00|Max=100.00|P1=100.00|"
                                "P2=0.00|RV=TRUE|",
                                "E_CHECKBOX_Fill_Wrap": "0",
                                "B_CHOICE_BufferStyle": "Per Model Default"}),
    # ALTERNATING STRANDS. Per Model Horizontal Per Strand gives each strand
    # its own row of the buffer, so a two-colour palette lands different
    # colours on neighbouring strands instead of blending across the sphere.
    # Another buffer style measured straight out of his sequences.
    "strand_stripes": ("Color Wash",
                       {"E_CHECKBOX_ColorWash_VFade": "1",
                        "E_CHECKBOX_ColorWash_CycleCount": "1",
                        "B_CHOICE_BufferStyle": "Per Model Horizontal Per Strand"}),
    "sphere_sparkle": ("Twinkle", {"E_SLIDER_Twinkle_Count": "40",
                                   "E_SLIDER_Twinkle_Steps": "12",
                                   "E_CHECKBOX_Twinkle_Strobe": "0",
                                   "E_CHECKBOX_Twinkle_ReRandom": "1",
                                   "B_CHOICE_BufferStyle": "Per Model Default"}),
    # --- bursts ---
    "on_hit":         ("On", {"T_TEXTCTRL_Fadein": "0.02", "T_TEXTCTRL_Fadeout": "0.12"}),
    "on_pop":         ("On", {"T_TEXTCTRL_Fadein": "0.01", "T_TEXTCTRL_Fadeout": "0.22"}),
    "on_swell":       ("On", {"T_TEXTCTRL_Fadein": "0.15", "T_TEXTCTRL_Fadeout": "0.35"}),
    "off":            ("Off", {"X_Effect_Locked": "True"}),
}

# Palettes: high-contrast, saturated -- bass music reads better with bold
# colour than with pastel Christmas washes, while staying holiday-plausible.
# Shaders the display can actually use, chosen from the show's own Shaders/
# folder for being abstract/organic generators (no input image, no text, no
# single-subject imagery that would look wrong stretched over a roofline).
# Verified assembling AND compiling to a Metal pipeline on this Mac via:
#     /Applications/xLights.app/Contents/MacOS/xLights \
#         -s "<show>/Shaders" --shadertranslate
# which reported "24 assembled, 24 ok, 0 fail" -- so a shader rendering wrong
# is a parameter/authoring problem, never a compile problem, on this machine.
SHADER_PICKS = [ "Metablob Starz", "InnerDimensionalMatrix",
    "Hex Vortex", "Mobius Spiral", "Spacels", "Hex 3D Spiral",
    # Written for this show (2026-08-11) to fill real gaps in the vocabulary
    # rather than add more of what was already there -- "I think we also need
    # to build more shaders or other effects." Every stock pick above is some
    # form of rotating/radial figure, which is part of why the large canvases
    # started to feel same-y. Each of these declares a SIGNED `rate`, so
    # register_shader_sway() gets a true direction reversal from all four.
    #   Aurora Curtain  -- slow vertical sheets; the only shader here built to
    #                      survive being stretched across the whole house,
    #                      where fast patterns read as strobing.
    #   Ripple Grid     -- expanding rings over a lattice; STRUCTURE rather
    #                      than texture, for a matrix that would otherwise wash.
    #   Drift Cells     -- breathing Voronoi membranes; organic with no scan
    #                      direction at all, unlike every chase and sweep.
    #   Falling Embers  -- sparse drifting particles with trails; the vocabulary
    #                      had no particle look whatsoever.
    "Aurora Curtain", "Ripple Grid", "Drift Cells", "Falling Embers",
    # Second batch (same session). Chosen against the gaps that were left:
    #   Kaleido Bloom  -- mirrored wedges, symmetric by construction, so round
    #                     props (flakes, spinners) don't get a pattern that
    #                     looks accidentally lopsided.
    #   Tide Lines     -- horizontal bands that swell in place with NO
    #                     left/right travel, filling the hole left by removing
    #                     the Wave effect without reintroducing a side scan.
    #   Ember Swarm    -- circling points around a drifting centre; sparse and
    #                     alive for quiet passages on big canvases.
    #   Contour Drift  -- topographic level sets. LINE-based, so it is
    #                     deliberately low-coverage and stays legible on a
    #                     matrix instead of lighting every pixel.
    #   Pulse Rings    -- discrete rings that fire and decay rather than
    #                     scrolling, so each one reads as a hit landing; pairs
    #                     well with the beat-locked brightness curve.
    "Kaleido Bloom", "Tide Lines", "Ember Swarm", "Contour Drift", "Pulse Rings",
    # Third batch. Gaps these fill:
    #   Interference -- two crossing ripple sources; moire that never repeats,
    #                   complex output from a trivial rule.
    #   Woven Bands  -- rectilinear lattice, for window matrices and roof lines
    #                   where round patterns look imposed on the shape.
    #   Bokeh Drift  -- soft out-of-focus discs at varied sizes; real depth,
    #                   gentle enough for quiet passages.
    #   Slow Dawn    -- ONE very slow gradient front, no repeating structure at
    #                   all. Built for the whole-house bed, where anything with
    #                   visible structure reads as strobing across 50 feet.
    #   Fracture     -- hard-edged crystalline shards. The vocabulary is almost
    #                   entirely soft, so this reads as a deliberate change of
    #                   texture rather than more of the same.
    "Interference", "Woven Bands", "Bokeh Drift", "Slow Dawn", "Fracture",
    # Fourth batch -- deliberately EVEN, centreless textures. "You're often
    # using the same shader across all matrices which works when it's a more
    # abstract shape or texture but when it's a circular or radial pattern it
    # looks a bit repetitive like showing the same movie on 10 screens." These
    # are field-filling by construction: every part of the canvas looks like
    # every other part, so ten props running one read as a single material
    # rather than ten copies of a picture.
    #   Herringbone   -- woven twill; textile grain
    #   Corduroy      -- ribbed, with a sheen travelling along the ribs
    #   Marble Vein   -- turbulent stone veining, seamless
    #   Iso Blocks    -- isometric 3D blocks with shaded faces; real depth
    #   Circuit Trace -- routed angular traces with charge pulses
    #   Liquid Chrome -- domain-warped molten metal with specular rolls
    "Herringbone", "Corduroy", "Marble Vein", "Iso Blocks", "Circuit Trace",
    "Liquid Chrome",
    # Fifth batch -- experiments in what a house-sized canvas can REPRESENT,
    # not just texture it. "These are large canvases covering the whole house
    # so you can make it look like anything."
    #   Rain on Glass -- droplets cling, swell, break and run, leaving trails.
    #                    A subject people name instantly rather than a pattern.
    #   Cable Knit    -- braided cables twisting over stocking stitch; the
    #                    front strand takes the highlight so it reads as
    #                    over-and-under, not two flat waves.
    #   Murmuration   -- a flock banking together, condensing and opening out.
    #                    CENTRED: the flock has a middle, so it belongs on one
    #                    canvas at a time.
    #   Stained Glass -- leaded panes each holding a steady tone, light passing
    #                    behind so they lift in turn. Architectural.
    #   Lava Lamp     -- metaballs under slow convection, with a taut skin at
    #                    the threshold. CENTRED and deliberately very slow.
    #   Smoke Drift   -- curl-warped density on a rising current, thinning with
    #                    height so it dissipates instead of filling the frame.
    "Rain on Glass", "Cable Knit", "Murmuration", "Stained Glass", "Lava Lamp",
    "Smoke Drift",
    # Sixth batch. All twelve of batches 5-6 were gated through
    # shader_smoke_test.py -- rendered on Garage Roof and measured for
    # blackness, flatness, stillness and panel-level strobe -- because
    # reading GLSL has repeatedly failed to catch shaders that compile to
    # black or sit still.
    #   Ink Bloom    -- ink hitting water, boundary torn by creeping angular
    #                   noise. CENTRED.
    #   Wind Wheat   -- stalks bending as a GUST TRAVELS across, not all
    #                   swaying in unison. The travelling envelope is the idea.
    #   Velvet Nap   -- crushed velvet: fibres in a swirled lay, catching a
    #                   sweeping light only where they face it.
    #   Oil Film     -- thin-film interference, written as a clean full-range
    #                   value RAMP so a multi-colour palette reads as bands of
    #                   colour sliding rather than one colour breathing. The
    #                   most literal answer to "colour movement".
    #   Frost Growth -- ice creeping in from the edges with crystals branching
    #                   ahead of the front, then thawing back.
    #   Paper Layers -- cut-paper depth: stacked ridges at different parallax
    #                   speeds, each shadowing the one behind.
    "Ink Bloom", "Wind Wheat", "Velvet Nap", "Oil Film", "Frost Growth",
    # NOT "Circle Tunnel" / "Zoom Loop" -- both declare a `color` TYPE ISF
    # input (POINT_COLOR_A/B, colorInput) with no `uColMode`-style palette
    # fallback, so isf_shader.py correctly leaves it unset ("color" inputs
    # are meant to be driven by the effect's assigned xLights palette where
    # supported -- these two just don't support that). Caught this round
    # once the new "share one shader across every large canvas per section"
    # cohesion feature made it visible: with Zoom Loop applied to 7 large
    # canvases at once, its own periodic colour behaviour, previously diluted
    # by different canvases running different shaders, spiked measured red
    # fraction to ~39% every ~1.8s during one section. Re-add either if a
    # verified-safe colour override is found.
    #
    # Added 2026-08-10, then partly reverted same day -- of the 23 real .fs
    # files on disk, 5 passed both existing safety checks (no required input
    # image, no uncontrolled colour input). "Fly Through" is NOT one of the
    # 5 kept: rendering it under the `xmas` palette (shared across several
    # large canvases at once) measured a SUSTAINED ~35-40% red for its whole
    # active window, confirmed by direct pixel sampling. Also removed
    # "Vortex-V2" (in the pool since round 11 with no prior complaint) after
    # the SAME measurement showed it doing the identical thing under the
    # SAME palette. Softening `xmas`'s red component first (from `#E82A2A`
    # to a muted `#B33A3A`, still in the PALETTES dict below) barely moved
    # the number -- these two shaders' own contrast/saturation math pushes
    # rendered output well past whatever base colour they're fed, so a
    # flatter/safer palette wasn't the fix. Neither shader is provably
    # unsafe under every OTHER palette, only confirmed unsafe under this
    # one, but with 8 safe shaders left and 5 more palettes with no red
    # component at all, excluding both outright is the same low-cost,
    # high-certainty call as Circle Tunnel/Zoom Loop -- re-add either if a
    # future round finds a settings combination that keeps it from
    # amplifying colour this way.
    "Black Cherry Cosmos", "Space Spinning", "Target Circle",
    # NOT "Great Ball of Fire" -- kept in this pool for one round on the
    # theory that its elevated (but under-gate) red on ONE song was just a
    # fire shader looking like fire. Regenerating more songs disproved that:
    # it measured 36-40% red on TWO more songs, once under `ice` (cold
    # blue/white) and once under `gold` -- i.e. regardless of an assigned
    # palette that has no red in it at all. This is the exact "no declared
    # colour input, but internally colour-fixed" gap flagged when it was
    # kept -- confirmed the general lesson holds: `has_uncontrolled_color()`
    # can only check ISF-declared colour inputs, it cannot detect a shader
    # that hardcodes colour in its GLSL body with no ISF hook whatsoever.
    # Excluded like Fly Through/Vortex-V2/Circle Tunnel/Zoom Loop before it.
    # --- 2026-08-29 batch. Written against what the library still could not do
    # after 44 shaders: almost everything was either a rotating/radial figure
    # or a continuous drift, and NOTHING carried rhythm in colour rather than
    # in movement. "Think about shapes and textures... think about the movement
    # and color as rhythmic elements as well."
    #
    #   Argyle March    -- lozenges that never move while their colour marches
    #                      in steps, one diagonal row at a time. Rhythm with
    #                      zero motion, so it is safe on the biggest props.
    #   Tartan Shift    -- warp and weft with an overcheck; the whole sett
    #                      steps to the next shade while the weave holds still.
    #   Sequin Flip     -- thousands of discs each turning on their own
    #                      schedule: a steady glitter made of tiny events
    #                      rather than one big one.
    #   Crackle Glaze   -- craquelure in a ceramic glaze, with a sheen
    #                      travelling through a structure that stays put.
    #   Chromatophore   -- cephalopod skin; cells open and close in a
    #                      travelling wave and the COLOUR is the pulse.
    #   Caustic Pool    -- concentrated liquid filaments with dark between,
    #                      a brightness distribution nothing else here has.
    #   Halftone Pulse  -- a print screen whose dot weight swells across the
    #                      prop. Quantised time was tried first and the smoke
    #                      test caught it strobing the whole panel; the swell
    #                      travels instead.
    #   Ribbon Braid    -- three ribbons with real over/under occlusion, so a
    #                      linear prop gets depth instead of another chase.
    #   Zoetrope Slats  -- a hidden pattern revealed in moving strips; the
    #                      pattern never travels, only the opening does.
    #   Moire Rings     -- fine fast rings whose interference figure is huge
    #                      and slow, giving a big prop large readable movement
    #                      out of small detail.
    "Argyle March", "Tartan Shift", "Sequin Flip", "Crackle Glaze",
    "Chromatophore", "Caustic Pool", "Halftone Pulse", "Ribbon Braid",
    "Zoetrope Slats", "Moire Rings",
    # NOT "Paper Layers": rendered alone on a matrix it draws a hard horizontal
    # seam across the middle -- rows 7-8 of 16 carry a completely different
    # structure from the smooth gradient above and below, which is the "bottom
    # half is a mostly solid shape and the upper half has a shader" that Krists
    # asked twice to remove (The WhatsApp Shakespeare 0:39 and 1:55). The
    # earlier split test missed it because it compared half-MEANS, and a band
    # in the middle leaves both halves equal.
    # NOT "Wisps": the smoke test reports "still, rate does nothing" -- it
    # renders a texture but never animates, so a canvas holding it for a whole
    # section is a static picture. Same reason as the black and strobing picks
    # removed above.
]


# BIG-IMPACT GESTURES, so one effect stops being the answer to everything.
# "I also think we overuse the shockwave effect a bit, maybe we can find other
# effects that give a similar impact, swipe across or something else." What
# these share is that they cross the whole prop quickly from a single origin;
# they differ in the shape of the crossing, which is what keeps repeated big
# moments from looking like the same moment.
IMPACT_KEYS = ["shockwave", "ripple_explode", "sweep_right", "sweep_left",
               "sweep_up", "sweep_down", "chase_mid_out", "pinwheel"]


def impact_key(seed: str, wide: bool = True) -> str:
    """One of the big-impact gestures, chosen deterministically from `seed`.

    Deterministic so a given moment always renders the same way, and so
    successive moments in a song walk the pool instead of re-rolling onto the
    same effect."""
    pool = IMPACT_KEYS if wide else [k for k in IMPACT_KEYS
                                     if not k.startswith("sweep")]
    return pool[hash(seed) % len(pool)]


def envelope_palette(palette: ColorPalette, rms, rt, start_ms: int,
                     end_ms: int, depth: float = 1.0) -> ColorPalette:
    """Copy of `palette` whose brightness follows the MUSIC'S OWN SHAPE over
    this span, instead of following the beat.

    In a drumless passage there is no beat to accent -- "they need to be more
    smooth and flowy" -- but there is still a contour: the synth swells, the
    pad thins out, the bass arrives. Sampling the section's measured RMS and
    turning it into the brightness curve makes the whole house breathe with
    that contour, which is a gesture a viewer reads as the music being
    followed even though nothing is marking time.

    Deliberately smoothed and normalised WITHIN the span: what matters is the
    passage's own rise and fall, not where it sits against the rest of the
    song, or a quiet intro would render as a flat dark line.
    """
    import numpy as _np
    s_s, e_s = start_ms / 1000.0, end_ms / 1000.0
    if e_s - s_s < 1.0:
        return palette
    m = (rt >= s_s) & (rt < e_s)
    if m.sum() < 4:
        return palette
    e = _np.asarray(rms)[m].astype(float)
    # Smooth over a BAR, not 0.6s.
    #
    # This is where the ambient pulsing actually came from, after three other
    # theories measured as inert (replacing brightness curves, slowing every
    # shader and chase, clearing the rhythm layers outright -- all applied,
    # all changing beat-periodicity by less than 0.01). The envelope follows
    # the music's RMS, and the music is amplitude-modulated at the beat: at
    # 117.5 BPM the beat is 0.51s, so a 0.6s window passes beat-rate
    # modulation almost untouched and the "smooth" treatment was faithfully
    # reproducing the pulse it was supposed to replace.
    #
    # A bar-length window keeps the passage's shape -- swells, builds, the
    # arrival of an instrument -- while removing the beat from it, which is
    # exactly the ask: "pulsing more like each bar or couple of bars."
    _bar_s = 4.0 * 60.0 / max(SONG_TEMPO, 1.0)
    w = max(3, int(_bar_s / max(float(rt[1] - rt[0]), 1e-6)))
    k = _np.ones(w) / w
    e = _np.convolve(e, k, mode="same")
    lo, hi = float(e.min()), float(e.max())
    if hi - lo < 1e-9:
        return palette
    e = (e - lo) / (hi - lo)
    n = min(VC_POINT_CAP - 2, 40)
    idx = _np.linspace(0, len(e) - 1, n).astype(int)
    pts = []
    for j, i in enumerate(idx):
        t = j / (n - 1)
        # Map to the same scale beat_pulse_palette uses: VC_NOMINAL is 100%.
        # Deliberately wide. "We can also exaggerate the brightness connection
        # to the music loudness/intensity a bit, so the brightness grows during
        # an ambient section that's building." At depth 1 this runs 25% to
        # 185% of nominal across the passage, so a section that builds visibly
        # grows rather than merely brightening a little.
        v = VC_NOMINAL * (1.0 - 0.75 * depth + 1.60 * depth * float(e[i]))
        pts.append((t, max(0.02, min(0.5, v))))
    clean = []
    for t, v in pts:
        if clean and t <= clean[-1][0] + 1e-4:
            continue
        clean.append((t, v))
    if len(clean) < 3:
        return palette
    vc = ("C_VALUECURVE_Brightness=Active=TRUE|Id=ID_VALUECURVE_Brightness|"
          "Type=Custom|Min=0.00|Max=400.00|RV=TRUE|Values="
          + ";".join(f"{t:.3f}:{v:.2f}" for t, v in clean))
    return palette.model_copy(update={
        "extra_parts": [x for x in (palette.extra_parts or [])
                        if not x.startswith("C_VALUECURVE_Brightness=")] + [vc]})


def fade_palette(palette: ColorPalette) -> ColorPalette:
    """Copy of `palette` with a HIT-then-SLOW-FADE brightness curve.

    "There is a single kick drum for each drop and everything between that and
    the next single kick drum is the slow fade."

    A cycle hold at constant brightness puts the right pattern in the right
    place and still reads wrong, because nothing decays: measured on Gooey,
    the kicks at 15.8/26.9/37.9/49.0 lifted the display only 1.0-1.6x, since
    the previous cycle was still sitting at full level when the next one
    landed. The shape has to be in the effect itself.

    Fast off the hit, then a long tail -- the same asymmetry a real drum decay
    has, and the reason it reads as one gesture rather than a triangle wave.
    """
    pts = [(0.00, 0.46), (0.04, 0.34), (0.12, 0.24),
           (0.30, 0.16), (0.60, 0.10), (1.00, 0.06)]
    vc = ("C_VALUECURVE_Brightness=Active=TRUE|Id=ID_VALUECURVE_Brightness|"
          "Type=Custom|Min=0.00|Max=400.00|RV=TRUE|Values="
          + ";".join(f"{t:.2f}:{v:.2f}" for t, v in pts))
    return palette.model_copy(update={
        "extra_parts": [x for x in (palette.extra_parts or [])
                        if not x.startswith("C_VALUECURVE_Brightness=")] + [vc]})


def percussive_ratio() -> float:
    """How much of this song's level is drums, 0-1. Cached per song.

    Used to tell a driving track from a mellow one -- see the pulse-mode gate.
    Measured across the library: 0.441 (So Easy) to 0.773 (Aperture).
    """
    cache = SHOW_PATH / ".generation" / SONG_NAME / "percussive.json"
    try:
        return float(json.loads(cache.read_text())["drums_over_mix"])
    except (OSError, ValueError, KeyError):
        pass
    try:
        import librosa
        y, _ = librosa.load(str(SHOW_PATH / f"{SONG_NAME}.mp3"), sr=22050, mono=True)
        d, _ = librosa.load(str(SHOW_PATH / "stems" / SONG_NAME / "drums.wav"),
                            sr=22050, mono=True)
        import numpy as _np
        r = float(_np.sqrt((d ** 2).mean()) / max(_np.sqrt((y ** 2).mean()), 1e-9))
    except Exception:
        return 1.0          # unknown -> don't let this gate block anything
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"drums_over_mix": round(r, 4)}))
    return r


def register_shaders() -> list[str]:
    """Add one EFFECTS entry per usable shader, with settings derived from
    that shader's own ISF header. Returns the list of registered effect keys
    (empty if the Shaders folder is missing, so the generator still runs)."""
    from isf_shader import ShaderUnusable, shader_settings

    # Per-shader parameter overrides where the ISF default is unsafe for our
    # "no strobing, ever" constraint. InnerDimensionalMatrix.fs's `flash`
    # input (ISF default 7.5 of range 0.5-10) is used in the shader source as
    # `pow(sin(id_offset + flash*TIME)*0.4+0.6, flash)` -- both the frequency
    # AND the sharpness of a per-tile flicker, i.e. it is not decorative
    # texture, it is a literal strobe control, confirmed by reading the .fs
    # source after "the tunnel matrix seems to be still flashing and
    # strobing" despite every other strobe source already being fixed.
    # Pinned near the ISF minimum for a slow, gentle shimmer instead.
    #
    # "Hex 3D Spiral"'s own `uColMode` defaults to 0 ("Shader Defaults" --
    # its own hardcoded green/blue `uC1`/`uC2`, not the assigned xLights
    # palette). Direct pixel measurement (rendering the same shader under two
    # different palettes and sampling the actual output) confirmed most
    # shaders here already track the assigned palette correctly even with NO
    # explicit colour-mode input at all -- xLights applies a generic palette
    # recolour to Shader effects independent of ISF colour inputs -- so this
    # override is a belt-and-suspenders consistency fix, not evidence the
    # other shaders needed the same treatment.
    SHADER_OVERRIDES: dict[str, dict[str, float]] = {
        "InnerDimensionalMatrix": {"flash": 1.0},
        "Hex 3D Spiral": {"uColMode": 1},
    }

    # Every parameter besides colour was frozen at the shader's ISF default,
    # identically, on every placement, in every song -- direct feedback: "I
    # have lots of songs and don't want them all to look the same." Verified
    # empirically that colour ALREADY varies correctly via the assigned
    # palette; nothing else ever did. Seeded per song (not per placement) so
    # a given shader has one consistent "personality" (a rotation speed, a
    # zoom level) for the whole song rather than jittering every individual
    # instance -- visually coherent within a song, genuinely different
    # across songs. See `shader_settings`'s `jitter_rng` docstring.
    jitter_rng = random.Random(f"{SONG_NAME}|shader-params")

    # Base shader rate scales with the song's TEMPO. Until now every song ran
    # its shaders at the same 400 regardless of whether it was 86 BPM or 143,
    # so on a slow song the texture raced ahead of the music and on a fast one
    # it dragged behind -- one concrete reason shaders never felt connected to
    # the beat even when their start times were perfectly on the bar. This
    # doesn't phase-lock anything (no ISF shader exposes a phase input), but it
    # puts the shader's own cycle in the same ballpark as the beat, which is
    # what makes the beat-locked brightness envelope on top read as one
    # coherent motion instead of two unrelated ones. Clamped so an outlier
    # tempo estimate (or a half/double-time detection) can't produce a strobe.
    base_speed = int(round(400 * min(1.5, max(0.6, SONG_TEMPO / 120.0))))

    keys: list[str] = []
    shader_dir = SHOW_PATH / "Shaders"
    for stem in SHADER_PICKS:
        fs = shader_dir / f"{stem}.fs"
        if not fs.exists():
            continue
        try:
            settings = shader_settings(fs, speed=base_speed,
                                       overrides=SHADER_OVERRIDES.get(stem),
                                       jitter_rng=jitter_rng)
        except (ShaderUnusable, ValueError):
            continue
        key = "shader_" + re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
        EFFECTS[key] = ("Shader", settings)
        keys.append(key)
    return keys


def _read_source_lines() -> list[str]:
    try:
        return Path(__file__).read_text().split("\n")
    except OSError:
        return []


SOURCE_LINES = _read_source_lines()


def _pass_label(lineno: int, lines: list[str]) -> str:
    """Name the pass that owns a line, by looking backwards for its heading.

    The generator's passes are delimited by ALL-CAPS comment headings and by
    the `print(f"...")` line each one ends with, so the nearest heading above a
    call site names it without anyone maintaining a registry.
    """
    if not lines or lineno <= 0:
        return "?"
    for i in range(min(lineno, len(lines)) - 1, max(0, lineno - 400), -1):
        t = lines[i].strip()
        if not t.startswith("#"):
            continue
        t = t.lstrip("# ").rstrip()
        if not t or t.startswith("---") or t.startswith("==="):
            continue
        letters = [c for c in t if c.isalpha()]
        # A heading, not prose: mostly capitals and short.
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7 \
                and len(t) < 80:
            return t.rstrip(".")
    return f"line {lineno}"


SHADER_KEYS = register_shaders()
# The subset safe to span across the whole house -- see the house-bed pass.
NEW_SHADER_STEMS = ("aurora_curtain", "ripple_grid", "drift_cells", "falling_embers",
                    "kaleido_bloom", "tide_lines", "ember_swarm", "contour_drift",
                    "pulse_rings", "interference", "woven_bands", "bokeh_drift",
                    "slow_dawn", "fracture", "herringbone", "corduroy",
                    "marble_vein", "iso_blocks", "circuit_trace", "liquid_chrome",
                    "rain_on_glass", "cable_knit", "murmuration", "stained_glass",
                    "lava_lamp", "smoke_drift",
                    "ink_bloom", "wind_wheat", "velvet_nap", "oil_film",
                    "frost_growth")
# NOTHING custom goes in the house bed. Slow Dawn was designed for it -- one
# slow gradient front, no repeating structure -- and it STILL took 12 to 12
# from a 15% red peak to 39.8% in one build, the identical failure mode the
# first batch caused. The bed spans every prop simultaneously, so a shader's
# coverage multiplies whatever palette bias exists across the entire facade,
# and "designed to be safe there" is not evidence of being safe there. This
# has now been measured twice; treat the bed as stock-shader-only unless a
# render proves otherwise.
HOUSE_BED_SAFE_CUSTOM = ()
# EFFECTS BUILT AROUND A CENTRE, which must never be SPANNED across separated
# props. "There's some sort of effect or shader spread across the garage roof,
# garage matrix, and fence matrix, it's like a rainbow circle, but its center
# is kind of between the matrices and not on any of them so it looks bad
# because they each just have like a curved rainbow line through them."
#
# Exactly right, and it is a property of the SHAPE, not the effect type. A
# spanning buffer stretches one canvas over props with gaps between them, so
# anything with a focal point puts that point in mid-air and leaves each prop
# holding a fragment of arc. Field-filling looks -- sheets, bands, cells,
# particles, washes -- have no centre to lose and span fine.
CENTRED_STEMS = ("target_circle", "hex_vortex", "hex_3d_spiral", "mobius_spiral",
                 "spirograph", "circle_tunnel", "kaleido_bloom", "ripple_grid",
                 "pulse_rings", "interference", "ember_swarm",
                 "murmuration", "lava_lamp", "ink_bloom")
CENTRED_BUILTINS = {"spirals", "pinwheel", "shockwave", "ripple_explode",
                    "ripple_implode", "radial", "galaxy"}


def is_centred(key: str) -> bool:
    """True for effects whose composition has a focal point."""
    return (key in CENTRED_BUILTINS
            or any(n in key for n in CENTRED_STEMS))


def _span_profile() -> dict:
    """Per-shader measurement of what happens when it is stretched across the
    whole house -- written by shader_span_test.py. Missing file = empty, and
    the hand-maintained rules below carry on alone."""
    f = SHOW_PATH / ".generation" / "shader_span.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


SPAN_PROFILE = _span_profile()


def small_prop_flicker(key: str) -> float:
    """How hard this shader swings a SMALL prop's whole brightness, 0-1.

    A shader that is calm on a matrix can strobe an orb: the matrix averages
    thousands of nodes while a small prop samples a tiny patch, so a fine
    moving texture swings that prop's entire output. shader_span_test.py has
    measured this all along and only the house bed acted on it -- so Hex 3D
    Spiral, which scores 1.000, the worst in the library, was still free to
    land on the Orbs directly. "In World Is Mine at 4:41 the orbs are strobing
    really hard."
    """
    stem = key[len("shader_"):] if key.startswith("shader_") else key
    for name, prof in SPAN_PROFILE.items():
        if re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") == stem:
            return float(prof.get("small_flicker", 0.0))
    return 0.0


CALM_ON_SMALL = 0.25        # same ceiling the house bed uses


def spans_evenly(key: str) -> bool:
    """True if this shader measurably fills the house rather than putting its
    picture in one place.

    is_centred() is a list of stems someone had to remember to add, so it only
    ever knew about shaders we had already been burned by. It did not know
    about Wisps -- a stock shader, not circular, but with a focal point all the
    same -- which is what produced "a shader on all the matrices that seems to
    have a center that's off the matrices... you just see the tendril looking
    outer edges." Measured spanned across the real layout, 63% of props fall
    below a fifth of the brightest prop, against a ~39% floor for an even
    texture (many props are small or at the edge of the frame regardless).

    The measurement is authoritative where it exists; the hand list stays as a
    backstop for shaders that have not been measured yet.
    """
    stem = key[len("shader_"):] if key.startswith("shader_") else key
    for name, prof in SPAN_PROFILE.items():
        if re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") == stem:
            return bool(prof.get("span_ok"))
    return not is_centred(key)


# The house bed is now gated on the MEASUREMENT rather than on provenance.
#
# The old rule was "nothing custom goes in the house bed", written after Slow
# Dawn regressed 12 to 12's red fraction. That reasoning no longer holds: at
# the time every custom shader emitted greyscale and xLights was assumed to
# recolour it, whereas the red actually came from STOCK shaders painting their
# own colours across the whole facade. Custom shaders now take their colour
# only from c1/c2/c3 -- i.e. only from the palette, which never leads with
# saturated red -- so they are the SAFER half of the library for this, not the
# riskier one. Keeping them out left a pool of three, which is its own problem:
# the whole-house layer is the most visible thing in the show to repeat.
#
# Red is re-checked by verify_render.py on every build; if this reopens that
# regression the number will say so immediately.
# How bright a shader may be and still sit under the whole show. Calibrated
# against the pool the bed used BEFORE this round -- InnerDimensionalMatrix
# 0.053, Black Cherry Cosmos 0.101, Space Spinning 0.045 -- which is what
# produced the lit fractions Krists has been happy with. Opening the bed to
# every evenly-spanning shader pulled in some very bright ones (Paper Layers
# 0.543, Velvet Nap 0.503) and cost 5 points of lit fraction across all three
# test songs, and hurt the run-up dip as well: the dip is a RATIO against the
# surrounding light, so a bright constant base leaves less to take away.
# 0.20 keeps the bed roughly as dim as it has always been while going from a
# pool of 3 to 11.
MAX_BED_LEVEL = 0.20


def bed_level(key: str) -> float:
    stem = key[len("shader_"):] if key.startswith("shader_") else key
    for name, prof in SPAN_PROFILE.items():
        if re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") == stem:
            return float(prof.get("level", 0.0))
    return 0.0


HOUSE_BED_SHADERS = [k for k in SHADER_KEYS
                     if not is_centred(k) and spans_evenly(k)
                     and bed_level(k) <= MAX_BED_LEVEL]

# SHADERS THAT OBEY THE ASSIGNED PALETTE.
#
# "Sometimes we have a red shader on the mega tree and a rainbow one on the
# tunnel matrix and a black and white one on the garage roof and it's a little
# disjointed... maybe shared color palettes could be configured for multiple
# shaders? Or maybe the same shader could be used but with slightly different
# colors, so they look more cohesive?"
#
# That is exactly the split here. Every shader written for this show emits
# GREYSCALE plus alpha, so xLights recolours it entirely from the ColorPalette
# assigned to the effect -- meaning several different shaders sharing one
# palette look like one family. The stock shaders generate their own colours
# internally and ignore the palette, which is where the red/rainbow/mono
# collisions come from. Preferring the palette-driven set on the hero canvases
# is what makes "different shaders that look related" possible at all.
PALETTE_DRIVEN_SHADERS = [k for k in SHADER_KEYS
                          if any(n in k for n in NEW_SHADER_STEMS)]



def is_palette_driven(key: str) -> bool:
    """True for our greyscale shaders, INCLUDING their `__swayB` variants.

    A plain `key in PALETTE_DRIVEN_SHADERS` test missed every sway variant,
    because register_shader_sway() registers those under a different key --
    so half the shader placements silently skipped the colour variant.
    """
    return any(n in key for n in NEW_SHADER_STEMS)


def palette_variant(pal: ColorPalette, n: int) -> ColorPalette:
    """The same colour family with a different colour leading.

    Gives simultaneous canvases "slightly different colors" rather than either
    identical output or unrelated palettes -- the middle option Krists asked
    for. Rotating which colour leads changes the dominant hue while keeping
    every canvas inside one family.
    """
    cols = list(pal.colors)
    if len(cols) < 2 or n % len(cols) == 0:
        return pal
    k = n % len(cols)
    return pal.model_copy(update={"colors": cols[k:] + cols[:k]})


# xLights VALUE CURVE FORMAT -- READ BEFORE EDITING ANY BRIGHTNESS CURVE.
#
# `Type=Flat` takes its level from `P1`, NOT from `Values=`. A curve written as
# `Type=Flat|...|Values=0.00:0.16;1.00:0.16` has no P1, so xLights reads the
# level as zero and THE EFFECT RENDERS COMPLETELY BLACK. It does not warn, and
# the .xsq looks perfectly correct.
#
# Every brightness curve in this generator was written that way. Measured in
# isolation on Corner Verticals: the same Color Wash renders mean 58.8 with no
# curve and 0.0 with a Flat+Values curve at ANY level. So every effect carrying
# an envelope, a dim, a hush or an accent -- thousands per song -- was rendering
# as nothing, which is why round after round of changes measured inert in the
# .fseq while looking right in the .xsq, and why Krists kept reporting that new
# versions "didn't improve much".
#
# Working forms, both verified against the renderer:
#     Type=Custom|Min=0.00|Max=400.00|RV=TRUE|Values=<t>:<v>;...   (v: 0.25 = 100%)
#     Type=Flat|Min=0.00|Max=400.00|P1=<percent>|RV=TRUE           (P1: 100 = 100%)
# Use Custom. It is what the time-varying curves already use, so there is one
# format in the file rather than two.
def accent_bright(pal: ColorPalette, level: float = 0.38) -> ColorPalette:
    """Lift a palette above the ambient background level.

    "Even though we're keeping the energy lower, the piano chords need a
    bright contrasting color." Ambient backgrounds sit near 0.09-0.15 on the
    brightness curve (VC_NOMINAL 0.25 == 100%), so a chord placed at nominal
    barely separated from them. This puts the chord itself around 150%, which
    is what makes it read as an event rather than as more background.
    """
    return pal.model_copy(update={"extra_parts": list(pal.extra_parts or []) + [
        "C_VALUECURVE_Brightness=Active=TRUE|Id=ID_VALUECURVE_Brightness|"
        f"Type=Custom|Min=0.00|Max=400.00|RV=TRUE|Values=0.00:{level:.2f};1.00:{level:.2f}"]})


def contrast_palette(pal: ColorPalette, n: int) -> ColorPalette:
    """A bright accent palette deliberately far in hue from `pal`.

    "Since we're using a faint gray and blue for most of the background, we
    should use something like an orange for all the outlines... or using them
    all but changing color on the beat."

    palette_variant rotates WITHIN a family, so a chord hit in a grey/blue
    section came out grey/blue -- present in the data, invisible on the house.
    A chord is the foreground event of an ambient passage and has to read
    against the background, so this picks a hue at least ~100 degrees away
    from the section's dominant colour and advances through the candidates as
    `n` increases, giving "changing color on the beat" for free. Greyscale
    sections have no hue to avoid, so they get the warm end by default, which
    is the orange Krists asked for.
    """
    import colorsys

    def _hue(hexs: str) -> tuple[float, float]:
        hexs = hexs.lstrip("#")
        r, g, b = (int(hexs[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        h, _l, sat = colorsys.rgb_to_hls(r, g, b)
        return h, sat

    # Warm-first, so the default accent against this show's cool backgrounds
    # is the orange that was asked for.
    ring = [
        ("#FF6A00", "#FFA033", "#FFD9A0"),   # orange
        ("#FFC400", "#FFE066", "#FFF4C2"),   # amber
        ("#FF2D6F", "#FF7AA8", "#FFD0E0"),   # magenta
        ("#7CFF3D", "#B6FF8A", "#E4FFD0"),   # lime
        # The cool half exists so a WARM section still has somewhere to go.
        # Without these, an ember/gold section filtered out every candidate
        # and fell back to the whole ring, handing an orange section an
        # orange "contrast" -- i.e. no contrast at all.
        ("#00E5FF", "#7DF3FF", "#D6FBFF"),   # cyan
        ("#2E6BFF", "#7FA5FF", "#D5E2FF"),   # azure
    ]
    # Judge against EVERY colour in the palette, not just the first. A palette
    # that leads with white but carries a blue was being treated as having no
    # hue at all, so the accent picked cyan -- blue on blue, which is not a
    # contrast. Greyscale sections fall back to the warm half deliberately:
    # this show's ambient background is "a faint gray and blue", so warm is
    # the direction that reads against it.
    hues = [_hue(c) for c in pal.colors]
    lit = [h for h, sat in hues if sat >= 0.15]
    if not lit:
        cands = ring[:4]                     # warm half
    else:
        cands = [c for c in ring
                 if all(min(abs(_hue(c[0])[0] - bh),
                            1.0 - abs(_hue(c[0])[0] - bh)) >= 0.28 for bh in lit)] \
            or ring[:4]
    return ColorPalette(colors=list(cands[n % len(cands)]), active_colors=[1, 2, 3])


def register_shader_sway() -> dict[str, tuple[str, str]]:
    """For each registered shader, build an A/B pair that can be alternated on
    bar boundaries so the shader's OWN motion swings with the music.

    "See if the shaders' own motion sway can change on bar boundaries or
    similar, sometimes it can even be on quarter notes when there's a strong
    driving dance beat... it would make a big difference." Until now a shader
    was placed once and animated at a constant rate for its whole span, so it
    was texture that happened to be present rather than something moving WITH
    the song.

    Two mechanisms, chosen per shader from its own ISF header:
      * a SIGNED rate/rotation input (Spirograph Spectrum, Mobius Spiral,
        Spacels, Hex 3D Spiral all have one) -> negate it, giving a true
        direction reversal;
      * otherwise an unsigned rate/speed input -> alternate between full and
        roughly half rate, which reads as a breath rather than a reversal.
    Shaders with neither (Target Circle, Space Spinning, Black Cherry Cosmos,
    SearchingForSomething) simply don't sway; they keep a single variant.
    """
    pairs: dict[str, tuple[str, str]] = {}
    signed_hints = ("rotat", "rate", "urotate")
    speed_hints = ("rate", "speed", "rotation", "scroll")
    for key in SHADER_KEYS:
        ename, settings = EFFECTS[key]
        target_param, signed = None, False
        for k in settings:
            if "SHADERXYZZY" not in k:
                continue
            short = k.split("SHADERXYZZY_")[-1].lower()
            try:
                val = float(settings[k])
            except (TypeError, ValueError):
                continue
            if any(h in short for h in signed_hints) and val < 0:
                target_param, signed = k, True
                break
            if any(h in short for h in speed_hints) and target_param is None:
                target_param = k
        if target_param is None:
            pairs[key] = (key, key)
            continue
        b = dict(settings)
        v = float(b[target_param])
        b[target_param] = str(int(round(-v))) if signed else str(int(round(v * 0.45)))
        bkey = f"{key}__swayB"
        EFFECTS[bkey] = (ename, b)
        pairs[key] = (key, bkey)
    return pairs


SHADER_SWAY = register_shader_sway()


# PALETTE COLOUR ORDER MATTERS ENORMOUSLY -- this was the real "red flash" bug.
#
# Measured 2026-08-09 by rendering to .fseq and reading back actual pixel
# values (see custom_tools/fseq_reader.py). Krists's own finished sequences
# run 0.3-11% of lit pixels as saturated red, peaking at 27%. Generated v4 sat
# at 35-68% saturated red for ~50 CONTINUOUS SECONDS, precisely across the two
# sections the palette cycle had assigned `xmas` (#FF0000 first) -- 1:09-1:59,
# which is exactly where he reported "the mega tree is red around 1:13" and
# "the tree and garage roof are red for a while around 1:40". Four rounds were
# spent hunting an xLights render/missing-asset bug that never existed: the
# props were red because we told them to be red.
#
# Many effects (On, Shockwave, single-colour chases) render predominantly the
# FIRST palette colour, so a palette that leads with pure #FF0000 paints an
# entire section's large props solid saturated red. Rule going forward: no
# palette leads with a pure-red primary. Red stays available as a secondary/
# tertiary for multi-colour effects, which is how it reads as festive rather
# than as an error state.
PALETTES: dict[str, ColorPalette] = {
    "ice":    ColorPalette(colors=["#00CFFF", "#0033FF", "#FFFFFF"], active_colors=[1, 2, 3]),
    "violet": ColorPalette(colors=["#8A00FF", "#FF00C8", "#FFFFFF"], active_colors=[1, 2, 3]),
    # Was 3 shades of the same green-yellow hue -- no actual contrast color
    # in it at all. Direct feedback ("we need more contrast... palette
    # changes feel disjointed") named this exact failure mode: a palette can
    # avoid being red-led and still render as visually monochrome. Kept the
    # neon-green identity but swapped the third slot for genuine magenta
    # contrast.
    "toxic":  ColorPalette(colors=["#00FF6A", "#FF00E5", "#CCFF00"], active_colors=[1, 2, 3]),
    # was #FF2A00-led (pure red); amber/orange lead keeps the "warm" feel
    # without the solid-red wall on big props.
    "ember":  ColorPalette(colors=["#FF7A00", "#FFB300", "#FFE066"], active_colors=[1, 2, 3]),
    # was #FF0000-led (round 9), then #E82A2A as colour 3 (round 15). Direct
    # measurement: TWO different shaders (Vortex-V2, in use since round 11
    # with zero prior red complaints, and the newly-added Fly Through) both
    # spiked to 45-50% measured red under this exact palette once shared
    # across several large canvases at once -- confirmed via rendering both
    # and sampling actual pixel output, not guessed. `xmas` is the ONLY
    # palette in this dict that still has any red-family colour in it at
    # all; every other palette (ember, gold, candy, toxic, ice, violet,
    # climax, white) has none. A shader's own contrast/gradient math can
    # push values well past a base colour's own saturation, so "not
    # literally #FF0000" (E82A2A has G=42, technically fails the strict
    # r>100,g<32,b<32 pure-red test) was not enough headroom -- softened to
    # a muted berry/burgundy tone that reads as red without being a
    # near-primary a shader's rendering can amplify into looking like an
    # error state. Still unmistakably a Christmas red, just not one two
    # unrelated shaders can turn into "half the display is red."
    # Real Christmas red again (was softened to #B33A3A when red was being
    # suppressed wholesale -- see RED_PALETTES).
    "xmas":   ColorPalette(colors=["#00E83C", "#FFFFFF", "#E02A2A"], active_colors=[1, 2, 3]),
    "gold":   ColorPalette(colors=["#FFD700", "#FFFFFF", "#FF8C00"], active_colors=[1, 2, 3]),
    "candy":  ColorPalette(colors=["#FF3FA4", "#FFFFFF", "#00D9FF"], active_colors=[1, 2, 3]),
    "white":  ColorPalette(colors=["#FFFFFF", "#C0E8FF"], active_colors=[1, 2]),
    # Reserved for the song's real climax moments only (see `climax_sections`
    # below) -- deliberately the highest-contrast combination available
    # (neutral white, warm gold, saturated blue) so the biggest moment in the
    # song never has to compete for visual attention with whatever an
    # ordinary section's palette happens to be. Never placed in PAL_CYCLE --
    # if it rotated in normally it would stop reading as special.
    # Gold-led, NOT white-led. Same lesson as the original red bug: colour 1
    # is what single-colour effects render, so a white-led palette makes every
    # Shimmer/Color Wash/On under it a white wash. Measured: `Shimmer` ran
    # 95.0-110.5s (15.5 CONTINUOUS seconds) on all three front window
    # matrices under this palette -- "around 1:35-1:40 the three upper window
    # matrices are white still." It was never a shader; the earlier
    # jitter-clamping fix was a real but *different* white bug. White stays as
    # colour 2 so multi-colour effects keep the bright celebratory feel.
    "climax": ColorPalette(colors=["#FFD700", "#FFFFFF", "#0047FF"], active_colors=[1, 2, 3]),
}
# --- per-song colour identity -------------------------------------------
#
# "It seems like each song might have the same color palette... Should we
# thoughtfully create a custom color palette for each song based on the style
# or mood, making sure they're all somewhat distinct from each other? I think
# this default palette is also making the colors seem somewhat random."
#
# He was right: every song drew four palettes at random from the SAME seven,
# so songs differed only in shuffle order and none had an identity. These
# families are picked per song for mood, and deliberately spread across the
# hue circle so two songs back to back don't read as the same show.
#
# Rule kept from the red-flash rounds: never lead a family with saturated red.
# A lead colour paints the biggest canvases, and pure red at scale was the
# single worst-looking failure this project has had.
# GREYSCALE AS A COLOUR. "The grayscale was actually also very cool, could we
# keep both? Just use grayscale as another color variation? Maybe use it
# especially for lower energy parts or something but not always so it's not
# repetitive."
#
# Nothing in the shaders needs to change for this. They read c1/c2/c3 straight
# from the palette, so a palette of neutrals reproduces exactly the greyscale
# look the whole library had before the colour conversion -- it becomes one
# palette choice among the others rather than a property of the shader.
# Three of them, so a mono section still has a character of its own: cold,
# warm, and a dim near-black.
# RED IS ALLOWED AGAIN. The red measurement exists because early versions had
# props rendering as flat saturated red when an effect failed -- 35-68% of lit
# pixels for ~50 continuous seconds -- and the palettes were pulled back hard
# in response, to the point where the library measured 0.0-0.2% red. That
# overshot: "I know we measured the amount of red in each song because we
# originally had issues with some things not rendering so they looked just
# red, but that doesn't mean we should aim for zero red, it's fine to have
# that color."
#
# So `xmas` gets a real Christmas red back (#E02A2A, not the muted berry it
# was softened to), and there is a red-led family. The verifier still reports
# red as a share of lit pixels, which is what catches the actual failure mode;
# the human reference band is 0.3-11% with peaks to 27%, and being far BELOW
# it is its own kind of wrong.
RED_PALETTES: dict[str, ColorPalette] = {
    "holly":  ColorPalette(colors=["#E02A2A", "#0FA958", "#FFF3D6"],
                           active_colors=[1, 2, 3]),
    "cherry": ColorPalette(colors=["#FF2D3C", "#7A0B2E", "#FFC46B"],
                           active_colors=[1, 2, 3]),
}

MONO_PALETTES: dict[str, ColorPalette] = {
    "ash":    ColorPalette(colors=["#FFFFFF", "#9AA6B2", "#3C4650"],
                           active_colors=[1, 2, 3]),
    "bone":   ColorPalette(colors=["#FFF6E8", "#C9BBA6", "#4A423A"],
                           active_colors=[1, 2, 3]),
    "pewter": ColorPalette(colors=["#D8E2EA", "#6E7A88", "#20262C"],
                           active_colors=[1, 2, 3]),
}

MOOD_PALETTES: dict[str, ColorPalette] = {
    # deep psychedelic -- violet/teal, no warm anchor
    "abyss":    ColorPalette(colors=["#6A00FF", "#00E5C0", "#FF00A8"], active_colors=[1, 2, 3]),
    # warm psychedelic -- amber against magenta
    "resin":    ColorPalette(colors=["#FFA200", "#FF2ECC", "#B6FF00"], active_colors=[1, 2, 3]),
    # smoky R&B -- rose/plum/gold, low contrast, adult
    "velvet":   ColorPalette(colors=["#D14A7A", "#7A2E8E", "#FFC96B"], active_colors=[1, 2, 3]),
    # club/house -- cool and clean
    "chrome":   ColorPalette(colors=["#00D5FF", "#2B5BFF", "#EAF6FF"], active_colors=[1, 2, 3]),
    # indie melancholy -- indigo and silver
    "dusk":     ColorPalette(colors=["#5C6BC0", "#9FB3D9", "#E8ECF5"], active_colors=[1, 2, 3]),
    # bold pop -- hot pink and tangerine
    "neon":     ColorPalette(colors=["#FF2D95", "#FF8A00", "#FFFFFF"], active_colors=[1, 2, 3]),
    # cinematic teal/orange
    "cinema":   ColorPalette(colors=["#00B3A4", "#FF7A33", "#FFE9D6"], active_colors=[1, 2, 3]),
    # icy/ethereal
    "glacier":  ColorPalette(colors=["#8FE9FF", "#3FA9C9", "#FFFFFF"], active_colors=[1, 2, 3]),
    # sultry -- crimson present but never leading
    "smoulder": ColorPalette(colors=["#FF6B4A", "#8E1B4B", "#FFD27A"], active_colors=[1, 2, 3]),
    # sweet/warm
    "peach":    ColorPalette(colors=["#FFB07C", "#FF5E8A", "#FFF1D6"], active_colors=[1, 2, 3]),
    # regal
    "regal":    ColorPalette(colors=["#7B2FF7", "#FFD24A", "#F3E9FF"], active_colors=[1, 2, 3]),
    # forest/aurora
    "grove":    ColorPalette(colors=["#33D17A", "#0F8B8D", "#DFFFE0"], active_colors=[1, 2, 3]),
}
PALETTES.update(MOOD_PALETTES)
PALETTES.update(MONO_PALETTES)
PALETTES.update(RED_PALETTES)

# One family per song, chosen for that track's mood, plus two neighbours to
# rotate through so a long song still evolves without leaving its identity.
# Songs whose pulse mode is a judgement call rather than a measurement.
PULSE_MODE_OVERRIDE: dict[str, str] = {
    "Meridian": "groove",
}

SONG_PALETTE_FAMILY: dict[str, list[str]] = {
    # The 2026-08-16 batch: seven cinematic/atmospheric tracks (four ODESZA,
    # CloZee, Catching Flies, ROYA). Chosen by ear against each track's
    # character rather than left to the random fallback, which draws from the
    # older PAL_CYCLE set and would have given these the Christmas-primary
    # look rather than the moodier one they want. Two of them carry a red
    # family, now that red is wanted again.
    "Corners of the Earth": ["regal", "smoulder", "holly"],
    "Meridian": ["abyss", "chrome", "neon"],
    "This Version of You": ["dusk", "velvet", "glacier"],
    "The Last Goodbye": ["peach", "cherry", "regal"],
    "Harmony (Axel Thesleff Remix)": ["grove", "abyss", "velvet"],
    "Ojai": ["resin", "peach", "grove"],
    "Cruise": ["cinema", "chrome", "dusk"],
    "ENIGMA":                    ["abyss", "glacier", "regal"],
    "Gooey":                     ["resin", "abyss", "grove"],
    "The WhatsApp Shakespeare":  ["velvet", "smoulder", "regal"],
    "Too Slow (All Night)":      ["chrome", "glacier", "neon"],
    "12 to 12":                  ["dusk", "chrome", "velvet"],
    "World Is Mine":             ["neon", "peach", "resin"],
    "Aperture":                  ["cinema", "grove", "chrome"],
    "Fara":                      ["glacier", "dusk", "abyss"],
    "Lose My Cool":              ["smoulder", "velvet", "peach"],
    "So Easy (To Fall in Love)": ["peach", "neon", "cinema"],
    "The Ambition (Girls Trip)": ["regal", "neon", "smoulder"],
}

PAL_CYCLE = ["ice", "violet", "toxic", "ember", "gold", "xmas", "candy",
             "holly", "cherry"]
# How many of PAL_CYCLE's palettes a single song actually uses. Direct
# feedback: "we keep changing color palettes each section, it's a bit
# disjointed... limit the palette a little, don't go too far." Cycling
# through all 7 every single section (the round-9 fix for the opposite
# problem -- one palette holding for 50 seconds) over-corrected into the
# show never settling into a look. A smaller, per-song-fixed subset gives
# real variety across sections while staying visually cohesive within one
# song; still random per song (via SONG_NAME) so different songs don't all
# converge on the same 4.
PALETTE_FAMILY_SIZE = 4


# ---------------------------------------------------------------------------
# Submodel families -- built-in geometry to move light THROUGH
# ---------------------------------------------------------------------------
#
# "The props on the house like the showstopper props and efl flakes and all
# that stuff has tons of submodels. We could pulse some of those, or we could
# add movement to those submodels with the main beat, like a left right or up
# down or circular movement that changes direction on the beat."
#
# These props are already built as ordered families -- Showstopper Spinner has
# 292 submodels, GE Rosa Grande 1 has 432, the EFL flakes have rings and
# spokes. Stepping through a SPOKE family in order IS circular movement, and
# stepping through a RING family is radial in/out. Nothing has to be
# synthesised: the geometry is in the layout, it just was never used as a
# sequence of positions.
#
# A family is a set of submodels sharing a name stem and differing by a
# trailing number ("Spoke 01".."Spoke 12", "Ring 1".."Ring 3"), which is how
# every one of these props is named.

def discover_depth_oriented() -> set[str]:
    """Props that run FRONT-TO-BACK rather than across the facade.

    "Fence Matrix 1 is turned a full 90 degrees so it goes front to back...
    looking head on as if the layout was 2d the matrix is only one pixel wide
    in a sense... The same thing actually happens with one of the c9 segments,
    Single Line 14, and I think maybe Single Line 13 as well."

    Exactly right, and it explains the symptom. A whole-house spanning buffer
    (`Per Preview`) maps every prop onto one shared 2D canvas built from the
    preview's X/Y. A prop pointing away from the viewer occupies a single
    column of that canvas, so the entire prop takes one colour and lights as a
    slab -- which is why only SOME effects do it: the spanning ones.

    Two signals, both read straight from the show file:
      * a matrix rotated ~90 degrees about Y (Fence Matrix 1: RotateY=-90)
      * a two-point line whose run is mostly along Z (Single Line-14:
        X2=0, Z2=319; a normal one reads X2=-396, Z2=0)
    """
    import xml.etree.ElementTree as _ET
    out: set[str] = set()
    path = SHOW_PATH / "xlights_rgbeffects.xml"
    if not path.exists():
        return out
    try:
        root = _ET.parse(path).getroot()
    except Exception:                                   # noqa: BLE001
        return out
    for m in root.iter("model"):
        if m.get("Active") == "0":
            continue
        name = m.get("name") or ""
        try:
            ry = abs(float(m.get("RotateY") or 0.0)) % 180.0
        except ValueError:
            ry = 0.0
        if 60.0 <= ry <= 120.0:
            out.add(name)
            continue
        try:
            x2 = abs(float(m.get("X2") or 0.0))
            z2 = abs(float(m.get("Z2") or 0.0))
        except ValueError:
            continue
        if z2 > 40.0 and z2 > x2 * 2.5:
            out.add(name)
    return out


DEPTH_ORIENTED = discover_depth_oriented()


SUBMODEL_FAMILY_MIN = 3      # fewer than 3 positions isn't movement
SUBMODEL_FAMILY_MAX = 24     # beyond this the step is too fine to read


def discover_submodel_families() -> dict[str, dict[str, list[str]]]:
    """{model: {family stem: [submodel names in numeric order]}} for enabled
    models. Ordered by the trailing number, which for these props follows the
    physical arrangement (spoke 1 next to spoke 2, ring 1 inside ring 2)."""
    import xml.etree.ElementTree as _ET
    out: dict[str, dict[str, list[str]]] = {}
    path = SHOW_PATH / "xlights_rgbeffects.xml"
    if not path.exists():
        return out
    try:
        root = _ET.parse(path).getroot()
    except Exception:                                   # noqa: BLE001
        return out
    for m in root.iter("model"):
        if m.get("Active") == "0":
            continue
        fams: dict[str, list[tuple[int, str]]] = {}
        for sm in m.findall("subModel"):
            nm = sm.get("name") or ""
            mt = re.match(r"^(.*?)[\s_-]*(\d+)$", nm)
            if not mt:
                continue
            stem, num = mt.group(1).strip(), int(mt.group(2))
            if not stem:
                continue
            fams.setdefault(stem, []).append((num, nm))
        keep = {}
        for stem, items in fams.items():
            if SUBMODEL_FAMILY_MIN <= len(items) <= SUBMODEL_FAMILY_MAX:
                keep[stem] = [n for _i, n in sorted(items)]
        # Props whose submodels are NAMED PARTS rather than a numbered series
        # ("BigSpokes", "Circles", "Spinner", "Sm Spokes", "Pedal Spinner")
        # were being skipped entirely -- which is why the two big EFL
        # snowflakes never got submodel treatment: "we don't seem to be using
        # the EFL Snowflake props, there are the two big EFL snowflakes and I
        # don't see submodels being used on those." Stepping through named
        # parts isn't rotation, but lighting a different part of the flake on
        # each beat is exactly the "pulse some of those" ask, and it uses
        # structure the prop already has.
        if not keep:
            # Outline parts are LINEAR PROPS, not steppable parts. A matrix's
            # `Horizontals` / `Verticals` submodel is one long strand that
            # wants smooth travel along its length; stepping the parts family
            # flashes it instead. Measured on Aperture: every `*/Horizontals`
            # had 52-71 `On` flashes at 375ms -- exactly this pass's step
            # length -- and no movement at all, which is "the horizontals are
            # all erratic and just flashing all over the place."
            _OUTLINE_PARTS = ("horizontal", "vertical", "outline", "border")
            named = sorted((sm.get("name") or "") for sm in m.findall("subModel")
                           if (sm.get("name") or "").strip()
                           and not any(w in (sm.get("name") or "").lower()
                                       for w in _OUTLINE_PARTS))
            if SUBMODEL_FAMILY_MIN <= len(named) <= SUBMODEL_FAMILY_MAX:
                keep["parts"] = named
        if keep:
            out[m.get("name")] = keep
    return out


SUBMODEL_FAMILIES = discover_submodel_families()


# ---------------------------------------------------------------------------
# Beat-locked brightness (the thing that makes a shader move WITH the song)
# ---------------------------------------------------------------------------
#
# Standing feedback, given three separate times: "the shaders have always been
# not tied to the beat enough, let's focus there." Everything tried before
# operated on WHEN a shader starts and stops -- change_marks() puts every
# start on a bar boundary, register_shader_sway() flips its internal direction
# every couple of bars. Both are true and neither is visible, because between
# those boundaries a shader is a continuous wash: nothing happens ON the beat,
# so there is nothing for a viewer to lock onto. Four bars is 8 seconds; a
# viewer cannot hear a relationship at that timescale.
#
# The fix is a per-beat BRIGHTNESS envelope riding on top, so the shader keeps
# its own texture but breathes on every beat. Krists's own sequences do exactly
# this -- one of his palettes carries
#     C_VALUECURVE_Brightness=...Type=Custom|Values=0.00:0.67;0.09:0.39;
#         0.20:0.67;0.30:0.21;0.40:0.67;...
# which is a hand-drawn 5-cycle pulse across one effect. Found by grepping his
# .xsq files for VALUECURVE; this is his technique, not an invention.
#
# Custom curve semantics (from those same files): points are `t:v` pairs where
# t is normalised 0-1 across the EFFECT's own span and v is normalised 0-1 of
# the Min..Max range, and Max=400 means v=0.25 is ordinary 100% brightness.
# Points are placed at the REAL detected beat times, not at a fixed rate, so
# the pulse follows tempo drift instead of approximating it with a sine.

VC_MAX = 400.0          # xLights brightness value-curve range is 0-400%
VC_NOMINAL = 0.25       # ...so 0.25 of it is plain 100%
VC_POINT_CAP = 40       # keep the settings string sane on long spans


def beat_pulse_palette(palette: ColorPalette, beat_times: list[float],
                       start_ms: int, end_ms: int,
                       depth: float = 1.0) -> ColorPalette:
    """Copy of `palette` with a brightness value curve that peaks on every beat
    inside [start_ms, end_ms].

    `depth` scales how deep the trough goes: 1.0 is a pronounced pulse for a
    hero canvas, ~0.4 a gentle breath for the whole-house ambient bed (which
    must stay a bed -- pumping it hard would just be a house-wide strobe, and
    "no strobing" is the one hard rule).

    Returns the palette UNCHANGED when the span is too short or holds fewer
    than two beats: a one-point curve is not a pulse, it's a dimmer.
    """
    s, e = start_ms / 1000.0, end_ms / 1000.0
    span = e - s
    if span < 0.6:
        return palette
    inside = [b for b in beat_times if s - 0.02 <= b <= e]
    if len(inside) < 2:
        return palette
    # NEVER PULSE FASTER THAN THE FLASH LIMIT.
    #
    # A value curve's times are normalised across the effect's own span, so a
    # curve built for one span and then applied to a shorter one pulses faster
    # than the beat it was built from. Measured on Harmony: a 7.8s span
    # carried curve points 78ms apart -- 12.9 Hz -- and the strobe verifier
    # duly found flicker on the flake and cane props. "No songs should do
    # that", and floods especially "can give people migraines", so the pulse
    # rate is capped at the same 240ms the effect-spacing pass uses.
    _MIN_PULSE_S = 0.24
    _thin, _last = [], -9.0
    for _b in inside:
        if _b - _last >= _MIN_PULSE_S:
            _thin.append(_b)
            _last = _b
    inside = _thin
    if len(inside) < 2:
        return palette
    # Thin to the sparsest musical rate that still fits the point cap, so a
    # long span pulses on half notes rather than dropping beats arbitrarily.
    every = 1
    while (len(inside) + every - 1) // every > VC_POINT_CAP // 2:
        every *= 2
    inside = inside[::every]

    hi = VC_NOMINAL * (1.0 + 0.55 * depth)      # accent on the beat
    lo = VC_NOMINAL * (1.0 - 0.50 * depth)      # trough between beats
    pts: list[tuple[float, float]] = [(0.0, lo if inside[0] - s > 0.05 else hi)]
    for i, b in enumerate(inside):
        t = (b - s) / span
        pts.append((t, hi))
        nxt = inside[i + 1] if i + 1 < len(inside) else e
        # Trough sits 70% of the way to the next beat: a fast decay after the
        # accent and a short lift back into it, which reads as a hit rather
        # than as a symmetric wobble.
        trough_t = (b + 0.70 * (nxt - b) - s) / span
        if trough_t < 1.0:
            pts.append((trough_t, lo))
    pts.append((1.0, lo))

    # Monotonic, deduplicated, clamped -- xLights ignores a curve whose points
    # are out of order and silently renders full brightness instead.
    clean: list[tuple[float, float]] = []
    for t, v in pts:
        t = min(1.0, max(0.0, t))
        if clean and t <= clean[-1][0] + 1e-4:
            continue
        clean.append((t, v))
    if len(clean) < 3:
        return palette

    values = ";".join(f"{t:.2f}:{v:.2f}" for t, v in clean)
    vc = ("C_VALUECURVE_Brightness=Active=TRUE|Id=ID_VALUECURVE_Brightness|"
          f"Type=Custom|Min=0.00|Max={VC_MAX:.2f}|RV=TRUE|Values={values}|")
    return palette.model_copy(update={
        "extra_parts": list(palette.extra_parts) + [vc]})


# ---------------------------------------------------------------------------
# Cast construction
# ---------------------------------------------------------------------------

# Large / centerpiece props that Krists's own finished sequences consistently
# target INDIVIDUALLY rather than as part of a group (measured across four
# sequences). They're big enough to carry their own choreography, and letting
# a small incidental group absorb them wastes them -- e.g. "Mega Tree GRP"
# lumps the Mega Tree with the physically distant Tunnel Matrix.
HERO_MODELS = {
    "Mega Tree", "Garage Roof", "Sun Room Window Matrix", "Megatree Star",
    "Showstopper Snowflake", "Showstopper Spinner", "Sphere-2", "Sphere-3 Inner",
    # "Tunnel Matrix Real" IS the real, wired walk-through tunnel and must be
    # sequenced as a centrepiece. Corrected 2026-08-09 after a misreading of
    # earlier feedback: it is the ~40-80 "Tunnel Arches N" models that are
    # disabled/unwired -- Krists modelled the tunnel as one matrix because
    # wiring it as individual arches was impractical, and keeps the arch
    # version in the layout only to preview how an arch build would look.
    "Tunnel Matrix Real",
}
# The "Tunnel" GROUP (the ~40-80 individual "Tunnel Arches N" models) stays
# out of MUST_HAVE_GROUPS: those models are the disabled, unwired arch-build
# preview. They're harmless to leave in the cast as an ordinary group -- if
# Krists ever enables them they'll get baseline coverage automatically -- but
# they must not claim a guaranteed slot ahead of props that physically exist.

# Physically distinct models arranged as one visual ensemble (left/center/
# right, left/right) that should look uniform, not independently animated --
# direct feedback: the 3 front window matrices looked "odd and asymmetrical"
# when each got its own independently-rotated ambient role. Built as
# symmetric synthetic groups in build_cast() (same broadcast mechanism as
# leftover-prefix families): one role/effect/timing decision, applied
# identically to every member. NOT the same thing as a physically-overlapping
# container/panel pair (those get fully excluded via shadowing, not
# synchronized) -- these are separate, non-overlapping models that should
# just all do the same thing.
SYMMETRY_FAMILIES: dict[str, list[str]] = {
    "Front Window Matrices": [
        "Front Center Window Matrix", "Front Left Window Matrix", "Front Right Window Matrix",
    ],
}

# Big 2D canvases that can carry genuinely complex imagery, not just a wash.
LARGE_CANVAS = {
    "Mega Tree", "Garage Roof", "Sun Room Window Matrix", "Tunnel Matrix Real",
    "Curtain Front", "Curtain North", "Curtain South",
}

# Large props that are round/cone/tube shaped rather than flat panels. They
# read completely differently from a roofline or a window matrix -- an effect
# that wraps or travels along an axis suits them, a flat left-to-right sweep
# does not. Krists treats the walk-through tunnel the same way he treats the
# mega tree, so they share this treatment. See feature_effect_for().
CONE_SHAPED = {"Mega Tree", "Tunnel Matrix Real"}

# Groups whose members are DISCRETE STANDALONE OBJECTS scattered around the
# display, rather than sections of one continuous surface. A group-spanning
# effect on these can only express itself by lighting members one after
# another, which at rhythm speed reads as strobing rather than as motion --
# the stars complaint. These render Per Model instead, so each object performs
# the effect inside its own geometry and the group reads as many small props
# animating together.
PER_MODEL_GROUPS = {
    "Stars", "Window Stars", "Mini Trees", "Mini Tree Star",
    "Floods", "Orbs",
    # Seven separate hanging spheres. Spanning one canvas across scattered
    # objects can only light them one after another, which at rhythm speed
    # reads as strobing rather than motion -- the same reason the Stars and
    # Orbs are here.
    "Hanging Spheres",
}

# Props that must never move quickly, whatever rhythm they're carrying.
# "The floodlights are too stroby... I like how the orbs flash and stuff
# because they're all together and it looks great when they sort of sparkle,
# but with the floods doing that it's just strobing, the floods must be slower
# and not random like that, they can add drama when we cut out a lot of lights
# because they light up the whole house." A flood is a wash light aimed at the
# whole house, so any hit from it changes the brightness of EVERYTHING -- the
# same sparkle that's charming on a cluster of small orbs is the entire scene
# blinking when floods do it.
SLOW_PROPS = {"Floods"}
SLOW_PROP_MIN_GAP_BEATS = 2.0

# Groups whose effects are written onto each MEMBER MODEL individually rather
# than onto the group element, with a small per-member time stagger.
#
# Setting `Per Model Default` as the buffer style on the group element should
# in principle already render per model, and the .xsq confirmed it was being
# written (256/256 effects on Stars) -- but the reported result was still
# "I still don't see any per-star effects, only on the group." Rather than
# keep arguing with the buffer style, this writes one effect per star model,
# which is unambiguous: the .xsq then literally contains elements named
# "Star 3 24in" with their own effects, and the stagger means the ten stars
# visibly animate as ten separate props instead of in lockstep.
#
# (The stars Krists means are `Star 1..4` in 12in/24in -- all active. The
# similarly-named bare `Star 1`/`Star 2`/`Megatree Star` are Active=0 and are
# excluded by the layout filter in build_cast.)
BROADCAST_GROUPS = {"Stars"}
BROADCAST_STAGGER_S = 0.06

# "Curtain Garage" LOOKS like the right container for the 5 garage curtain
# panels (empty Controller, NumStrings=3) but its StartChannel is
# "!Front Curtain Center:1" -- it's wired to the FRONT curtain's controller,
# not the garage's. Confirmed via xlights_rgbeffects.xml (2026-08-09): its
# ScaleX (19.6) is also wildly smaller than the real 5-panel span (160).
# Effects placed on it land on the wrong physical curtain. Exclude it
# entirely; the 5 real panels (Curtain Garage 1, Curtain Garage-2..5) get
# grouped by the prefix fallback below instead. This is a data issue in the
# show file itself, worth fixing in xLights directly -- flag it to Krists.
MISWIRED_MODELS = {"Curtain Garage"}

# Group/submodel names that represent thin outline/linear shapes (window
# outlines built from Verticals/Horizontals submodels, snowflake spokes/
# outlines) even though their parent model's category is matrix/custom.
# These read best with the same chase/wave treatment as the C9s and vertical
# strips, per direct feedback that that treatment works well.
OUTLINE_NAME_HINTS = ("vertical", "horizontal", "outline", "spoke", "arm", "petal")

# The major named whole-prop-family groups -- always kept regardless of cast
# trimming, since these are the ones a viewer actually notices and names
# (arches, C9s, orbs, floods...). Includes the synthetic leftover-prefix
# names used when no real xLights group covers a family (e.g. "Curtain
# Garage", once the miswired container of the same name is excluded).
MUST_HAVE_GROUPS = {
    "Arches", "C9s", "Orbs", "Floods", "Candy Canes",
    "Curtain Garage", "Curtains All", "Icicles", "Window Matrices",
    "Front Window Matrices",  # keep every SYMMETRY_FAMILIES entry name here too
    # NOT "Tunnel" -- that group is the disabled arch-preview models, see the
    # HERO_MODELS comment above. The real tunnel is "Tunnel Matrix Real",
    # which is a hero MODEL and so is guaranteed a slot by that route.
    #
    # "Mini Trees" (Mini Tree 1 + Mini Tree 2) was direct feedback -- "I
    # don't think we're using the mini trees at all... basically dark the
    # whole time." Confirmed by dumping build_cast()'s actual output: the
    # group was never even entering the cast pool (only "Mini Tree Star",
    # a DIFFERENT group for the star toppers, made it in). At only 2
    # members it was too small to win the general weighted-sample trim
    # against much larger groups, every single run.
    "Mini Trees",
    # Added to the layout Sep 2026 -- seven hanging spheres Krists is building:
    # "I added them to a new group called Hanging Spheres, so please make sure
    # you use them." A brand-new group loses the weighted-sample trim on every
    # run, exactly as Mini Trees did: it reached add() with all seven members
    # and was then sampled out of a ~70-slot cast drawn from ~200 groups, so
    # nothing in the library targeted them at all.
    "Hanging Spheres",
    # Same failure, found the same way, after "we're not really utilizing the
    # stars very much, they just have the whole house effects but don't seem
    # to do much of their own." The layout has a real 12-member "Stars" group
    # and a 15-member "House Stars", but NEITHER was reaching the cast --
    # only four tiny star fragments (Megatree Star n=1, Sun Room Star Left
    # n=1, Mini Tree Star n=2, Window Star Outlines n=2) had made it in, so
    # the stars as a family had no collective presence at all and only lit up
    # because the whole-house shader bed underlays everything.
    "Stars", "Window Stars",
}

MEGA_GROUPS = {
    "All", "ALL Display", "ALL House", "ALL Yard", "ALL Roof", "All Pixels GRP",
    "All Pixels GRP - No Mega Tree Matrix", "All Pixels GRP - No Yard",
    "All Pixels No Yard GRP", "All Pixels Yard Only", "All House No Props",
    "All House Props", "ALL House Decorations", "All Pixels GRP - Top Center",
    "ALL Stars", "All Snowflakes", "Windows All", "Window Matrices",
}


def _base_model(member: str) -> str:
    """'Front Center Window Matrix/Verticals' -> 'Front Center Window Matrix'."""
    return member.split("/", 1)[0]


def select_submodels(sub_names: list[str], max_n: int = 6) -> list[str]:
    """Pick a representative sample from a prop's submodel list, preferring
    curated top-level names over section-header markers or a numbered
    per-arm/per-strand breakdown of an already-selected pattern. Some props
    (Showstopper Spinner) define BOTH: "01 Cascading Arches" (one curated
    named pattern, what we want) AND "Cascading Arches-01".."12" (the 12
    individual arms making up that one pattern -- redundant/too granular if
    the pattern-level name is already available)."""
    section_header = lambda n: n.startswith("*")
    numbered_subpart = lambda n: bool(re.match(r"^.+-\d+$", n))
    curated_pattern = lambda n: bool(re.match(r"^\d+\s", n))  # "01 Name", "15 Name"

    candidates = [n for n in sub_names if not section_header(n) and not numbered_subpart(n)]
    if not candidates:
        candidates = [n for n in sub_names if not section_header(n)]
    numbered = [n for n in candidates if curated_pattern(n)]
    pool = numbered if len(numbered) >= max_n else candidates
    return pool[:max_n]


def build_cast(show_config, positions: dict[str, dict]):
    """Choose a non-overlapping set of targets (groups + container models +
    submodel groups), preferring real xLights groups, and never lighting two
    models that physically overlap (container vs its per-controller panels)."""
    models = {m.name: m for m in show_config.models}

    # Models Krists has DISABLED in the layout (`Active="0"`). Effects placed
    # on these render nothing at all -- they are wired-up-but-switched-off or
    # preview-only props.
    #
    # This layout has 123 of them, and the generator had never checked the
    # attribute in any round, which quietly wasted effort and produced several
    # "that prop doesn't seem to do anything" reports:
    #   * `Megatree Star` is inactive, yet was in HERO_MODELS and had been
    #     assigned an entire instrument voice -- a whole instrument was being
    #     performed by a prop that cannot light up.
    #   * `Window Stars` is 3-for-3 inactive, so adding it to MUST_HAVE_GROUPS
    #     last round guaranteed a slot to something completely invisible.
    #   * `Stars` and `House Stars` each carry a couple of dead members, so
    #     any group-spanning effect on them has holes in it.
    # The `Tunnel Arches 1..80` preview models are in here too, which is the
    # same fact earlier rounds encoded by hand as a special case.
    # IMPORTANT nuance, established by measurement: `Active="0"` means two
    # different things depending on whether the model drives a controller.
    #   * WITH a Controller (120 models here) -- a real output prop that is
    #     switched off. Sequencing it is wasted, and Krists's own finished
    #     sequences essentially never target these (0-6 per sequence, and
    #     those are leftovers from when the props were still enabled).
    #   * WITHOUT a Controller (only 3 here: Curtain Front/North/South) --
    #     a VIRTUAL CONTAINER whose child panels do the actual output. These
    #     must still be sequenced: Silent Night puts 196 effects on each of
    #     those three, and container-vs-panel targeting is the thing several
    #     earlier rounds were spent getting right.
    # Filtering on Active alone would therefore have destroyed the curtain
    # targeting. Verified before enabling: no group in the cast is emptied by
    # this rule (Candy Canes keeps 4 of 8 live members, Stars 10 of 12).
    inactive: set[str] = set()
    try:
        _rgb = (SHOW_PATH / "xlights_rgbeffects.xml").read_text(errors="replace")
        for _m in re.finditer(r'<model name="([^"]*)"([^>]*)', _rgb):
            _attrs = _m.group(2)
            if 'Active="0"' not in _attrs:
                continue
            # Active=0 means OFF, controller or not. The old rule only
            # excluded disabled models that HAD a controller, so that
            # controller-less virtual containers (the curtains) survived --
            # correct when those containers were the only way to target the
            # curtain panels, wrong the moment Krists disables them on purpose.
            # Measured after he disabled them: Curtain Front / North / South
            # were still in the cast AND were carrying beat anchors.
            inactive.add(_m.group(1))
    except OSError:
        pass

    def is_active(n: str) -> bool:
        return _base_model(n) not in inactive

    def valid_member(n: str) -> bool:
        return (n in models or _base_model(n) in models) and is_active(n)

    # A GROUP whose name is also a disabled MODEL is disabled too. The
    # curtains exist as both -- a container model and a same-named group -- so
    # filtering members alone left `Curtain Front/North/South` in the cast
    # (and anchoring beats) after Krists switched them off.
    # The `Tunnel` group is the PREVIEW representation of the walk-through
    # tunnel -- 80 `Tunnel Arches N` models, every one `No Controller`. It must
    # not be sequenced independently: measured on Gooey, it carried 245 effects
    # of its own against the real matrix's 88, which is exactly why "the tunnel
    # arches don't always match the tunnel matrix." It receives content solely
    # by mirroring from `Tunnel Matrix Real` at write time.
    PREVIEW_ONLY_GROUPS = {"Tunnel"}
    groups = {g.name: [n for n in g.members if valid_member(n)]
              for g in show_config.model_groups
              if _base_model(g.name) not in inactive
              and g.name not in PREVIEW_ONLY_GROUPS}
    # Drop groups left with nothing real in them (e.g. Window Stars).
    groups = {k: v for k, v in groups.items() if v}

    # SPLIT OUTLINE-STRIP GROUPS BY PARENT SIZE.
    #
    # These groups render per-model (see add_target), so every strip traverses
    # its own length in the SAME time -- which means a long strip moves faster
    # than a short one. "The three smaller window matrices have the right speed
    # for the movement in the horizontals, but the bigger matrices like the sun
    # room, garage, and fence, the movement is too fast." Splitting lets the
    # big strips take a slower chase while the small ones keep the speed
    # Krists already likes. NumStrings is the size proxy: the small windows are
    # 1-string, Sun Room 3, Fence 4, Garage Door 5.
    # NumStrings read here directly -- model_strings is populated further down
    # with the container scan, after this point.
    _nstr: dict[str, int] = {}
    try:
        _rgb2 = (SHOW_PATH / "xlights_rgbeffects.xml").read_text(errors="replace")
        for _mm in re.finditer(r'<model name="([^"]*)"([^>]*)', _rgb2):
            _ns = re.search(r'NumStrings="(\d+)"', _mm.group(2))
            _nstr[_mm.group(1)] = int(_ns.group(1)) if _ns else 1
    except OSError:
        pass

    synthetic_groups: set[str] = set()
    _split: dict[str, list[str]] = {}
    for gname, mem in list(groups.items()):
        if not any("/" in m for m in mem):
            continue
        big = [m for m in mem if _nstr.get(m.split("/")[0], 1) >= 3]
        small = [m for m in mem if m not in big]
        if len(big) >= 2 and len(small) >= 2:
            _split[f"{gname} (Large)"] = big
            _split[gname] = small
    # These split names are NOT real xLights groups -- Krists has no
    # "Right Verticals (Large)" in his layout, and xLights flagged the sequence
    # for referencing models that don't exist. They are recorded here so the
    # cast can treat them as SYNTHETIC: effects placed on them broadcast to
    # their real submodel members instead of to the invented name. No layout
    # changes required, which is the point -- the split is a sequencing
    # concept, not a prop.
    for k, v in _split.items():
        groups[k] = v
        synthetic_groups.add(k)
    if inactive:
        print(f"Layout: skipping {len(inactive)} models marked Active=0")

    # Per-model geometry, used for the measured `large` test below.
    model_strings: dict[str, int] = {}
    model_nodes: dict[str, int] = {}

    # Container models: no controller + multiple strings => they span their panels.
    raw = ET.parse(SHOW_PATH / "xlights_rgbeffects.xml").getroot().find("models")
    containers: set[str] = set()
    for m in raw:
        if m.tag != "model":
            continue
        name = m.get("name", "")
        if name in MISWIRED_MODELS:
            continue
        # Container detection ran before any Active check, so a disabled
        # container walked straight back into the cast -- this is the path that
        # kept Curtain Front/North/South alive through two earlier fixes.
        if not is_active(name):
            continue
        try:
            nstr = int(m.get("NumStrings") or "1")
        except ValueError:
            nstr = 1
        model_strings[name] = nstr
        try:
            model_nodes[name] = (int(m.get("parm1") or "1")
                                 * int(m.get("parm2") or "1"))
        except ValueError:
            model_nodes[name] = 0
        if not (m.get("Controller") or "").strip() and nstr > 1:
            containers.add(name)
    # Panels shadowed by a container get excluded entirely.
    shadowed: set[str] = set()
    for c in containers:
        for name in models:
            if name != c and (name.startswith(c + " ") or name.startswith(c + "-")):
                shadowed.add(name)

    covered: set[str] = set()
    cast: list[dict] = []

    def cat_of(member: str) -> str:
        base = _base_model(member)
        return models[base].model_category if base in models else "custom"

    # 2a-pre. OUTLINE submodels are not standalone targets.
    #
    # A `<Matrix>/Horizontals` or `/Verticals` submodel is one strand with a
    # single member, so every movement path (which needs >=2 members to chase
    # across, and a real spread to chase along) refuses it and it falls through
    # to flashing. Measured on Aperture: Sun Room's four outline submodels
    # carried 94-103 `On` effects each and no movement whatsoever -- the last
    # remaining source of "the matrix horizontals still have messy constant
    # flashing." The parent matrix already lights that area as part of its own
    # canvas, so dropping these loses no coverage, only the mess.
    OUTLINE_SUBMODEL_WORDS = ("horizontal", "vertical", "outline", "border")

    def _is_outline_submodel(nm: str) -> bool:
        return ("/" in nm
                and any(w in nm.split("/")[-1].lower()
                        for w in OUTLINE_SUBMODEL_WORDS))

    def add(name, kind, members, synthetic=False, symmetric=False):
        # See OUTLINE_SUBMODEL_WORDS: a single-strand outline submodel can
        # never satisfy any movement path, so it only ever flashes.
        if _is_outline_submodel(name):
            return
        members = [m for m in members if m not in covered and m not in shadowed
                   and _base_model(m) not in MISWIRED_MODELS]
        if not members:
            return
        cats = Counter(cat_of(m) for m in members)
        cat = cats.most_common(1)[0][0]
        # Thin linear/outline shapes (window verticals/horizontals, snowflake
        # spokes/outlines) read best with the C9-style chase/wave treatment
        # regardless of their parent model's own category.
        if any(h in name.lower() for h in OUTLINE_NAME_HINTS):
            cat = "single_line"
        cast.append({
            "name": name, "kind": kind, "members": members,
            "cat": cat, "n": len(members), "synthetic": synthetic,
            "symmetric": symmetric,
        })
        covered.update(members)

    # 1. Container models first -- they own their panels.
    for c in sorted(containers):
        if c in models:
            add(c, "model", [c])

    # 2. Hero props claim their own slot before any group can absorb them.
    #    HERO_MODELS is the curated list; anything MEASURABLY large joins it
    #    automatically, so a big canvas Krists adds later is a hero from the
    #    first render instead of being swallowed by whatever group happens to
    #    contain it. Measured this round: both Fence Matrices (4 strings each)
    #    and the new Garage Door Matrix were reaching the cast only as group
    #    members, which denies them the large-canvas effect pool entirely.
    auto_heroes = {n for n, ns in model_strings.items()
                   if ns >= 3 and n in models and is_active(n)
                   and (models[n].display_as or "").lower() in ("matrix", "custom")}
    for h in sorted(set(HERO_MODELS) | auto_heroes):
        if h in models and h not in MISWIRED_MODELS and is_active(h):
            add(h, "model", [h])

    # 2a. Auto-discover submodels for rich hero props, as ADDITIONAL targets
    # alongside the whole-model hero entry (not instead of -- a submodel and
    # its parent legitimately both carrying effects is the normal, supported
    # xLights pattern real sequences use, not a conflict). Direct, strong
    # feedback: "I really rely on submodels a lot to make things look more
    # complex" -- a prop like Showstopper Spinner has 80+ named parts
    # (cascading arches, petals, rings, swirls) that whole-model treatment
    # alone never touches.
    SUBMODEL_SAMPLE_PER_HERO = 6
    for h in sorted(HERO_MODELS):
        if h not in models or not is_active(h):
            continue
        sub_names = [s.name for s in models[h].submodels]
        if len(sub_names) < 3:
            continue
        for sub in select_submodels(sub_names, SUBMODEL_SAMPLE_PER_HERO):
            full_name = f"{h}/{sub}"
            add(full_name, "model", [full_name])

    # 2b. Symmetry families: physically distinct models meant to look like
    # one uniform ensemble (see SYMMETRY_FAMILIES above). synthetic=True so
    # every role broadcasts identically to each member (add_target); marked
    # symmetric=True so PASS 2's individual-member-cycling passes (melody/
    # bass arpeggio chase, synth-stab rotation) skip them -- those passes
    # deliberately hit ONE member at a time to represent a traveling note,
    # which is exactly what breaks symmetry for an ensemble that should
    # always move as a unit.
    for fam_name, fam_members in SYMMETRY_FAMILIES.items():
        add(fam_name, "group", fam_members, synthetic=True, symmetric=True)

    # 3. Real xLights groups. Must-have named families claim their members
    # FIRST (regardless of size), so a smaller overlapping group can't starve
    # e.g. "C9s" out of existence before it's even constructed -- guaranteeing
    # it survives the later cast-size trim is useless if dedup already ate
    # its members via a different group. Everything else follows smallest
    # first, skipping mega catch-alls.
    ordered_groups = sorted(
        groups.items(),
        key=lambda kv: (kv[0] not in MUST_HAVE_GROUPS, len(kv[1])),
    )
    for name, members in ordered_groups:
        if name in MEGA_GROUPS or len(members) < 2:
            continue
        fresh = [m for m in members if m not in covered and m not in shadowed]
        # Only take the group if it still contributes most of its own
        # membership -- except must-haves, which take whatever's left even
        # if a sibling must-have (e.g. "Orbs" nested inside "Floods") already
        # claimed a big chunk of it; still worth keeping with fewer members
        # than losing the family entirely on a threshold technicality.
        min_needed = 2 if name in MUST_HAVE_GROUPS else max(2, len(members) * 0.6)
        if len(fresh) >= min_needed:
            add(name, "group", members,
                synthetic=(name in synthetic_groups))

    # 3b. Leftover numbered families no xLights group covers (e.g. the 5 real
    # "Curtain Garage 1..5" panels, orphaned once "Curtain Garage" itself is
    # excluded as miswired) -- same prefix-stripping fallback used when no
    # xLights groups exist at all, applied only to what's left uncovered.
    # (`re` is imported at module scope -- a redundant local `import re` here
    # made it a function-local name, which broke the earlier Active=0 lookup
    # above that also uses it.)
    leftover = [n for n in models
                if n not in covered and n not in shadowed and is_active(n)]
    buckets: dict[str, list[str]] = defaultdict(list)
    for n in leftover:
        m = re.match(r"^(.*?)[\s\-]*(\d+)$", n)
        if m:
            buckets[m.group(1).strip()].append(n)
    for prefix, names in buckets.items():
        if len(names) >= 2:
            # "prefix" is NOT a real xLights model or group name -- it's just
            # the common text stripped off a numbered family (e.g. "Curtain
            # Garage" stripped from "Curtain Garage 1"). Placing an effect on
            # it directly would either orphan (if the string matches nothing)
            # or silently mis-target (if it happens to collide with a real
            # but unrelated model, as "Curtain Garage" itself does here).
            # synthetic=True makes every role broadcast per-member instead.
            # symmetric=True: these leftover families are almost always
            # panels of one physical surface (the garage-curtain case this
            # mechanism exists for), not a sequence meant to move
            # independently -- keep PASS 2's individual-member-cycling
            # passes off them for the same reason as SYMMETRY_FAMILIES above.
            add(prefix, "group", names, synthetic=True, symmetric=True)

    # 4. Anything still uncovered, individually. `is_active` matters most
    # here: this is the catch-all, so every disabled model in the layout
    # would otherwise land in the cast as its own target -- which is how
    # `Megatree Star`, `Star 1/2`, `Sun Room Star Left` and the disabled
    # `Tunnel Arches N` were still receiving effects after the other three
    # entry points had been filtered.
    for name in sorted(models):
        if (name not in covered and name not in shadowed
                and name not in MISWIRED_MODELS and is_active(name)):
            add(name, "model", [name])

    # Trim the pool toward the human target-vocabulary size (45-57 elements
    # measured across four finished sequences). Submodel groups (window
    # outlines, snowflake parts) roughly doubled the pool's raw size to
    # ~115+; left uncapped, a whole song's worth of section-by-section
    # rotation ends up touching nearly all of them at least once, which is
    # what pushed total on-screen target count and placement count well past
    # the human range even with per-section activity capped. Keep every
    # container/hero/individual-model entry (they're load-bearing), and
    # randomly sample the GROUP entries down, biased toward larger (more
    # visually significant) groups.
    # Bumped from 62/12 (2026-08-09): explicit feedback "I really rely on
    # submodels a lot... I want more submodel and model group usage in
    # general." The new hero-submodel auto-discovery step (2a, above) adds
    # ~16 always-kept individual entries (Sun Room Window Matrix verticals/
    # horizontals, Showstopper Snowflake rings, Showstopper Spinner curated
    # patterns) that bypass this budget entirely (they're kind="model", not
    # "group") -- without raising TARGET_CAST_SIZE to make room, that growth
    # would silently zero out the group-entries budget and undo the outline/
    # symmetry-family sampling below. The per-section n_active_target bands
    # are absolute (not relative to pool size), so a bigger pool means more
    # VARIETY across the song, not more simultaneous on-screen density.
    TARGET_CAST_SIZE = 70
    OUTLINE_SAMPLE_SIZE = 16  # keep the new capability, but not all ~24-37 of them

    def weighted_sample(pool, k, rng):
        pool = sorted(pool, key=lambda t: (-t["n"], t["name"]))
        chosen = []
        while pool and len(chosen) < k:
            weights = [1.0 / (i + 1) for i in range(len(pool))]
            r = rng.random() * sum(weights)
            acc = 0.0
            for i, w in enumerate(weights):
                acc += w
                if acc >= r:
                    chosen.append(pool.pop(i))
                    break
        return chosen

    trim_rng = random.Random(f"{SONG_NAME}|cast-trim")
    keep = [t for t in cast if t["kind"] != "group"]
    all_groups = [t for t in cast if t["kind"] == "group"]

    # The major named whole-prop-family groups are always kept -- these are
    # what people actually notice (arches, C9s, orbs, floods, tunnel...).
    # Explicit MUST_HAVE_GROUPS names, plus anything from SYMMETRY_FAMILIES
    # automatically (via the "symmetric" flag) -- a name has to be listed in
    # BOTH SYMMETRY_FAMILIES and MUST_HAVE_GROUPS to survive trimming
    # otherwise, which is exactly how the window-matrix trio silently ended
    # up with zero effects the first time (defined as a symmetry family,
    # forgot to also add it to MUST_HAVE_GROUPS, random trim dropped it).
    must_have = [t for t in all_groups if t["name"] in MUST_HAVE_GROUPS or t.get("symmetric")]
    remaining = [t for t in all_groups if t not in must_have]

    # Outline/submodel groups (window verticals/horizontals, snowflake spokes/
    # rings/outlines) are new this round and specifically requested -- don't
    # let random sampling drop the capability entirely, but don't guarantee
    # literally all ~24-37 of them either, or THEY squeeze out the families
    # above. Sample a bounded subset -- but a flat weighted sample over the
    # raw list is dominated by whichever prop happens to have the most
    # redundant sub-variant groups (GE Grand Illusion and GE Rosa Grande
    # alone contribute a dozen+ overlapping spoke/arm groups for essentially
    # 2 props), which crowded out completely different props like the plain
    # snowflakes/stars entirely. Cluster by the prop each group actually
    # belongs to (its members' majority base model) and round-robin across
    # clusters instead, so sampling spreads across DIFFERENT decorated props.
    def cluster_key(t):
        bases = Counter(_base_model(m) for m in t["members"])
        return bases.most_common(1)[0][0]

    clusters: dict[str, list[dict]] = defaultdict(list)
    for t in remaining:
        if any(h in t["name"].lower() for h in OUTLINE_NAME_HINTS):
            clusters[cluster_key(t)].append(t)

    outline_sample: list[dict] = []
    cluster_order = list(clusters.keys())
    trim_rng.shuffle(cluster_order)
    for c in clusters.values():
        trim_rng.shuffle(c)
    round_idx = 0
    while len(outline_sample) < OUTLINE_SAMPLE_SIZE and any(clusters[c] for c in cluster_order):
        for c in cluster_order:
            if clusters[c] and len(outline_sample) < OUTLINE_SAMPLE_SIZE:
                outline_sample.append(clusters[c].pop())
        round_idx += 1
    other_groups = [t for t in remaining if t not in outline_sample]

    keep = keep + must_have + outline_sample
    budget = max(0, TARGET_CAST_SIZE - len(keep))
    group_entries = weighted_sample(other_groups, budget, trim_rng)
    cast = keep + group_entries

    for t in cast:
        # LARGE is now MEASURED, not listed. A hardcoded name set means every
        # prop Krists adds is treated as a small incidental until someone edits
        # the list -- measured after this round's changes, the new Garage Door
        # Matrix (5 strings, replacing the curtains) and both Fence Matrices
        # came through as ordinary props and would have been denied the hero
        # treatment entirely. He has said he will keep adding props, so this
        # has to work without code edits: a flat panel with several strings, or
        # a big node count, IS a large canvas.
        # Node/string counts are keyed by MODEL name, so a GROUP never found
        # itself here and every group fell through as not-large -- which is
        # why "Front Window Matrices" (three 640-node panels, 1920 nodes
        # between them) was classed small, took the no-motion branch, and drew
        # flat washes while every individual matrix around it ran patterns:
        # "the upper three window matrices often show solid colors or they're
        # mostly blank with just the outlines doing things, when the other
        # matrices all have effects on them."
        # A GROUP of matrices is a big canvas -- measuring it by node count
        # does not work here (model_nodes reports 1 for these models, so the
        # arithmetic version of this test silently classed a 3-panel window
        # group as small). Membership is the reliable signal: three matrix
        # panels side by side are collectively a drawing surface, and treating
        # them as small is what left them on flat washes while every
        # individual matrix around them ran patterns.
        _members = t.get("members") or [t["name"]]
        _grp_of_matrices = (t["kind"] == "group" and t["cat"] == "matrix"
                            and len(_members) >= 2)
        t["large"] = (
            t["name"] in LARGE_CANVAS
            or _grp_of_matrices
            or (t["cat"] == "matrix"
                and (model_strings.get(t["name"], 1) >= 3
                     or model_nodes.get(t["name"], 0) >= 400))
        )
        # Discrete scattered objects render inside each member, not across
        # the group -- see PER_MODEL_GROUPS. Also applied to any tree-category
        # group: a 2-member "Mini Trees" group spanned as one canvas can only
        # chase BETWEEN the two trees, which is why the mini-tree movement
        # read as "always side to side" -- per-model lets each tree animate up
        # and down within its own cone instead.
        t["per_model"] = (t["name"] in PER_MODEL_GROUPS
                          or (t["cat"] == "tree" and not t.get("large")))
        t["slow"] = t["name"] in SLOW_PROPS
        t["broadcast"] = t["name"] in BROADCAST_GROUPS

    # Attach spatial info for direction decisions. Submodel-qualified members
    # ("X/Verticals") aren't in the position table themselves -- fall back to
    # their parent model's position.
    for t in cast:
        pts = [positions[m] if m in positions else positions.get(_base_model(m))
               for m in t["members"]]
        pts = [p for p in pts if p]
        if pts:
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            t["cx"], t["cy"] = sum(xs) / len(xs), sum(ys) / len(ys)
            t["spread_x"], t["spread_y"] = max(xs) - min(xs), max(ys) - min(ys)
        else:
            t["cx"] = t["cy"] = t["spread_x"] = t["spread_y"] = 0.0
    return cast


# ---------------------------------------------------------------------------
# Role assignment (orchestration)
# ---------------------------------------------------------------------------

ROLE_OFF, ROLE_OUTLINE, ROLE_TEXTURE, ROLE_PULSE, ROLE_FEATURE = (
    "off", "outline", "texture", "pulse", "feature")

# Direction flip used to alternate a feature effect bar to bar (bounce feel).
REVERSE = {
    "chase_lr": "chase_rl", "chase_rl": "chase_lr",
    "chase_dots_lr": "chase_dots_rl", "chase_dots_rl": "chase_dots_lr",
    "chase_pair_lr": "chase_pair_rl", "chase_pair_rl": "chase_pair_lr",
    "chase_mid_out": "chase_mid_in", "chase_mid_in": "chase_mid_out",
    "sweep_right": "sweep_left", "sweep_left": "sweep_right",
    "sweep_up": "sweep_down", "sweep_down": "sweep_up",
    "bars_up_in": "bars_down_in", "bars_down_in": "bars_up_in",
    "strand_up": "strand_down", "strand_down": "strand_up",
    "strand_bounce": "strand_bounce",
    "curtain_close": "curtain_open", "curtain_open": "curtain_close",
}

# Rotated so rhythmic hits aren't all the same flash.
PULSE_KEYS = ["on_hit", "shockwave", "chase_mid_out", "sweep_right",
              "on_hit", "ripple_explode", "chase_mid_in", "sweep_left"]


# Which kind of prop should carry which instrument, and how it should move.
#
# The point of this table is LEGIBILITY, which was the core ask: "I want a
# person watching to see some prop doing a rhythm and have a second to hear
# that oh it's doing that to the hi hats or to the piano chords." That only
# works if (a) one prop represents exactly one instrument, and (b) the
# assignment HOLDS -- a prop that changes which instrument it's following
# every section is unreadable no matter how accurately each individual
# section is timed. So assignment is made ONCE for the whole song.
#
# `cats` is a preference order, not a hard requirement -- assignment falls
# back to any unclaimed target if a voice's preferred shapes are all taken.
# `key` is the effect a hit on that voice plays.
VOICE_PROP_PREFS: dict[str, dict] = {
    # Kick already has a dedicated whole-facade pulse pass (the kick-locked
    # body pulse). This entry is for a SINGLE prop that additionally reads as
    # "the kick drum", separate from that ambient bed.
    # NOT matrix, and NOT large. Excluding big canvases from the beat ANCHOR
    # was only half the fix -- the kick voice was still handing the Mega Tree
    # 167 short flashes a song, and the snare/vocal voices were doing the same
    # to the window matrices. "I don't like pulsing the whole mega tree though,
    # just like pulsing matrices doesn't make sense, nor the tunnel matrix."
    # Same principle wherever a flash-shaped voice picks its prop.
    "kick":  {"cats": ["tree", "custom", "arch"], "key": "on_hit",
              "large_ok": False},
    # Backbeat -- wants something substantial and central, and it's slow
    # enough (~1 hit per beat or slower) that a big prop won't feel strobey.
    "snare": {"cats": ["custom", "tree", "arch"], "key": "shockwave",
              "large_ok": False},
    # Hats run at eighths/sixteenths. Only ever a small, scattered,
    # twinkle-appropriate prop -- this is the one voice where fast is correct,
    # and putting it on anything large would be exactly the "hard on the eyes"
    # problem from earlier rounds.
    "hat":   {"cats": ["custom", "icicle", "single_line"], "key": "on_pop", "large_ok": False},
    "bass":  {"cats": ["arch", "single_line", "matrix"], "key": "chase_mid_out", "large_ok": False},
    # Chords/piano are sustained -- a wide linear prop swaying reads as
    # harmony far better than a flash does.
    # A single fixed key means one gesture repeated for the whole song, on the
    # same prop, in every song. "We seem to be overusing the jagged wave effect
    # thing that moves left and right across the matrices, it's too repetitive
    # to use that." `keys` is a per-song pool -- one is chosen per song, so a
    # chord line still reads consistently within a song but differs across the
    # show. Wave stays in the pool; it just stops being the only answer.
    "chord": {"cats": ["single_line", "matrix"],
              "keys": ["chase_mid_out", "chase_bounce", "sweep_right",
                       "chase_lr"],
              "key": "chase_mid_out", "large_ok": False},
    "vocal": {"cats": ["custom", "icicle"], "key": "on_swell", "large_ok": False},
}

# A voice has to actually be playing this densely (onsets per second) within a
# section before its assigned prop performs it there -- stops a prop stuttering
# out a handful of stray detections during a section where that instrument
# isn't really present.
# Voices switched OFF. `chord` comes from demucs' `other` stem, which is a
# catch-all: every synth, pad, guitar, key, FX sweep and reverb tail that isn't
# drums/bass/vocals. Its "onsets" are any spectral change, not necessarily
# rhythmic events, and it measures as the DENSEST voice on most songs (550 on
# ENIGMA, 417 on World Is Mine). Dense plus irregular is what reads as chaotic.
# Krists: "I feel like there is some stem we're using that often can just be
# chaotic seeming... let's try disabling chord other."
# chord/other is back ON -- disabling it did not help ("I don't think the
# chord experiment has helped actually"). Trying `hat` instead: it is the
# highest-frequency voice, the second densest, and fast stems have been
# flagged as looking bad independently several times.
DISABLED_VOICES = {"hat"}

VOICE_MIN_DENSITY = 0.35
# A voice whose rhythm repeats this consistently bar-to-bar is worth showing
# even when it is sparse -- see Voice.pattern_score_in and the gate that uses
# it. 0.25 sits below every real figure measured on ENIGMA (0.26-0.67) and
# above the noise floor of an unpatterned part.
VOICE_MIN_PATTERN = 0.25

# A rhythm can be carried by MOVEMENT rather than by a flash, and on the right
# prop shape that reads far better. Direct feedback: "the rhythmic voice
# visuals seem to mostly focus on flashing the prop, but we could have more
# complexity because movement can also be rhythmic. Like when the mini trees
# flash to the kick drum, they could also have an up down movement to the
# rhythm."
#
# Keyed by prop CATEGORY, because what "moving in rhythm" means is a property
# of the prop's shape, not of the instrument: a tree wants vertical travel, a
# line of C9s wants to run along its length, a matrix wants to sweep across.
# Each entry is (effect_key, reverse_key) and successive hits alternate
# direction, so the prop visibly rocks back and forth on the beat instead of
# restarting the same one-way gesture forever.
VOICE_MOTION_BY_CAT: dict[str, tuple[str, str]] = {
    # Genuinely VERTICAL travel. `sweep_up`/`sweep_down` are Bars effects, so
    # combined with per-model rendering (see PER_MODEL_GROUPS) each tree runs
    # a bar up and down its own cone. The previous pairing used a horizontal
    # chase as its second half, which on a 2-member "Mini Trees" group could
    # only chase BETWEEN the two trees -- hence "I notice it's always side to
    # side, we should do up and down too."
    "tree":        ("sweep_up", "sweep_down"),
    # Multi-dot rather than one dot crossing the whole run -- these are long
    # lines (the C9 gutter especially). See _chase()'s `chases` note.
    "single_line": ("chase_dots_lr", "chase_dots_rl"),
    "icicle":      ("chase_lr", "chase_rl"),
    "matrix":      ("chase_mid_out", "chase_mid_in"),
    "arch":        ("chase_lr", "chase_rl"),
    "custom":      ("chase_mid_out", "chase_mid_in"),
}

# Movement needs room to be seen. Below this gap between hits a moving effect
# can't complete enough of its travel to read as movement -- it just becomes a
# smear, which is the "too random and stroby on anything faster" complaint.
# Faster voices keep a short flash instead, which at those rates reads as
# shimmer/texture rather than as motion. An eighth note at 120-130 BPM is
# ~0.23s, so this sits just under that: quarters and eighths move, anything
# faster doesn't.
VOICE_MOTION_MIN_GAP_S = 0.22


def line_motion_for(target: dict, seed: str = "") -> tuple[str, str]:
    """Pick the right chase for a linear prop, by ORIENTATION and LENGTH.

    Three separate corrections are encoded here, all from the same round:

    * **Orientation.** "I don't see any vertical movement chases on the
      verticals or window outline verticals. I think I might see left/right
      movement across verticals which just makes them strobe randomly." A
      group of vertical strips spanned as one canvas can only chase
      HORIZONTALLY -- across the strips -- and since each strip lights whole,
      that reads as sequential flashing, not movement. Vertical groups instead
      get per-model Bars travelling up/down INSIDE each strip.
    * **Length.** "The chases on the window outlines and verticals etc don't
      need that many because they're smaller." Dot count scales with the run's
      physical width: long gutter runs get 4, short outlines get 2.
    * **Speed.** "The c9 chases are too fast too." Handled by the caller,
      which stretches a long run's movement across two hits instead of one.
    """
    name = target["name"].lower()
    if "vertical" in name or "icicle" in name:
        # Chase ALONG each strand -- on a vertically-mounted strip that is
        # physically up-and-down travel. See PMSL for why Bars(direction=up)
        # does NOT work here: these props are `DisplayAs="Single Line"`, so
        # their buffer has no second dimension for "up" to mean anything.
        # VARIETY WITHIN THE ORIENTATION RULE. The rule above is about
        # geometry and stays; what was missing is that every linear prop in a
        # song got the SAME pair, so 2136 of Gooey's outline chases were six
        # keys with one settings profile between them. Krists's own sequences
        # run Bounce, From/To Middle, Dual Chase and the Fireworks FX across
        # the same props, with head fades and per-model-per-preview buffers.
        #
        # `seed` keeps a prop's choice stable within a section and lets it
        # change between sections, so it is variety rather than jitter.
        _r = random.Random(f"{target['name']}|{seed}")
        return _r.choice([
            ("strand_up", "strand_down"),
            ("strand_up", "strand_down"),
            ("strand_bounce", "strand_bounce"),
            ("strand_drift", "strand_drift"),
            ("strand_spark", "strand_spark"),
            ("strand_mid_out", "strand_mid_in"),
        ])
    wide = target.get("spread_x", 0) >= 40 or "c9" in name or "gutter" in name
    _r = random.Random(f"{target['name']}|{seed}")
    if wide:
        # A long run still needs several dots so nothing races the length of
        # it -- that correction is preserved in every option here.
        return _r.choice([
            ("chase_dots_lr", "chase_dots_rl"),
            ("chase_dots_lr", "chase_dots_rl"),
            ("chase_spark_lr", "chase_spark_lr"),
            ("chase_bounce_drift", "chase_bounce_drift"),
        ])
    return _r.choice([
        ("chase_pair_lr", "chase_pair_rl"),
        ("chase_bounce_pp", "chase_bounce_pp"),
        ("chase_mid_out_pp", "chase_mid_in_pp"),
        ("chase_dual", "chase_dual"),
        ("chase_spark", "chase_spark"),
        ("chase_lr_pp", "chase_rl_pp"),
    ])


# The fan is out of the LARGE rotation -- "the spinning fan thing is just too
# basic... works on smaller props or props with weird shapes but not big
# matrices." Pinwheel stays available to small/odd props via their own pools.
SAFE_LARGE = ["spirals", "plasma_organic",
              *SHADER_KEYS,
              "chase_mid_out", "chase_lr", "chase_rl"]
# Weighted toward shaders for ONGOING section choices (shared_large_key and
# feature_effect_for's own fallback) -- direct feedback: "maybe less
# built-in effects like the spiral and wave... would love it if they were
# more complex so maybe use the shaders." SAFE_LARGE alone already leans
# shader-heavy by raw count, but the round-11 "one shared choice per
# section" cohesion feature means whichever ONE key gets picked is what
# EVERY large canvas shows at once -- so a spiral/wave choice, even at its
# original ~40% share, reads as a much more noticeable "simple" stretch now
# than it did back when each canvas rolled independently. Doubling the
# shader weight cuts how often a whole section lands on the plain built-ins
# without removing them -- "don't lose the rhythmic movement that's there"
# was explicit, and spiral/wave/chase's clear one-bar directional sweep is
# part of what reads as rhythmic that a shader's own ambient motion doesn't
# replace on its own.
# Chases are OUT of the flat-canvas pool. On a 2D matrix a SingleStrand chase
# renders as broad vertical bars sliding side to side, and measured across
# five songs it was the single most-used thing on the big matrices by a wide
# margin -- 445 placements totalling 1300 seconds, more than double the next
# entry. "We're overusing the effect where the large vertical lines slowly
# move across the matrix side to side, it looks nice for using occasionally
# ... but we're just using it too much."
#
# They stay available for cones and curtains, where the same effect wraps or
# draws rather than striping, and they still reach flat canvases through the
# other passes -- this only stops the FEATURE roll from picking one.
SAFE_LARGE_WEIGHTED = [*SHADER_KEYS, *SHADER_KEYS,
                       "spirals", "plasma_organic"]


def _is_flood(target: dict) -> bool:
    """Floods, including via the group that broadcasts to them."""
    nm = (target.get("name") or "").lower()
    if "flood" in nm:
        return True
    return any("flood" in (m or "").lower() for m in (target.get("members") or []))


def prominence_of(t: dict) -> int:
    """How recognisable is this prop to someone standing in the street?

    3 = hero/large/must-have family, 2 = an ordinary named group,
    1 = an individual model or a sub-detail family (submodel, `*GRP*`,
    spoke/outline group). Used both for assigning instrument voices and for
    picking the beat-accent rotation -- in both cases the association only
    lands if the prop is one a person could point at and name.
    """
    name = t["name"]
    if name in HERO_MODELS or t.get("large") or name in MUST_HAVE_GROUPS:
        return 3
    if "/" in name or any(h in name.lower() for h in OUTLINE_NAME_HINTS) \
            or name.endswith("GRP"):
        return 1
    if t["kind"] == "group":
        return 2
    return 1


def wants_motion(target: dict) -> bool:
    """Targets that should never be left with a flat flash/wash/twinkle --
    they're visually significant enough that "just flash and fade" reads as
    under-used. Large canvases and the arch cascade already had this; direct
    feedback added Candy Canes specifically ("they're big enough to do
    movements... not just flash and color fade"), which generalizes to any
    scattered-but-substantial decorative group (n>=4 matches the threshold
    `feature_effect_for`'s own fallback branch already uses to decide a
    group is big enough to sweep prop-to-prop rather than sit static).

    Tree-shaped props are included at ANY size, on the strength of how
    Krists's own finished sequences treat them rather than a size argument:
    Mini Trees (only 2 members, physically small) gets SingleStrand x326 in
    Cold Heart, Morph/Wave/Bars/Fan in Pink Pony Club, and Meteors in
    Blinding Lights -- movement in every case, essentially never a flash. A
    tree has a vertical axis to travel along, so "small" doesn't imply
    "flash is the right idiom" the way it does for a point-like prop.

    `matrix` joined `custom` in the n>=4 clause after direct feedback named
    Orbs (matrix, n=10) and Floods (matrix, n=13) specifically: "they keep
    having these really brief flashes that are just one or two at a time and
    it just seems random... if they're not on a main beat it doesn't make
    sense." Both fell through every existing check because they're
    categorized `matrix`, not `custom` -- category name was gating a
    size-based rule it had no business gating. This still leaves Orbs/Floods
    real motion via their `matrix`-category branch in `feature_effect_for`
    (chase_lr/rl, wash, plasma, shaders) once redirected off PULSE, which
    keeps the "big rhythmic cross-model sweep" character already validated
    as their right treatment -- just via the FEATURE role instead of being
    one incidental stop in an unrelated whole-display sweep pass."""
    return bool(target.get("large") or target["cat"] in ("arch", "tree")
                or (target["cat"] in ("custom", "matrix") and target["n"] >= 4))


def feature_effect_for(target: dict, rng: random.Random, escalation: int,
                       shared_large_key: str | None = None) -> str:
    """Pick a motion effect suited to the target's shape and category.

    `shared_large_key`, when given, is used verbatim for any `large` target
    instead of rolling an independent random choice -- direct feedback:
    "we're often using different shaders on each matrix... I feel like we
    could share shaders across matrices... more cohesive overall." Callers
    that want several large canvases to visibly move in lockstep within one
    section pick ONE key via `rng.choice(SAFE_LARGE)` and pass it to every
    call for that section; callers that don't care pass None and each
    target still gets its own independent choice.
    """
    cat = target["cat"]
    wide = target["spread_x"] >= target["spread_y"]
    n = target["n"]

    # Big 2D canvases can carry real imagery -- give them shaped, layered
    # motion rather than flat washes. Red-flash reports have now clustered on
    # every large-canvas member at some point (Mega Tree, Garage Roof, Sun
    # Room Window Matrix), so the "safe core" (Spirals/Pinwheel/Plasma/real
    # GLSL shaders/Chase/Wave -- all proven across many rounds with zero
    # complaints) is shared by everyone in LARGE_CANVAS. Curtain/Galaxy/
    # Ripple/Butterfly are held back to the ONE shape they're each unambiguous
    # for: Curtain only for genuine curtain-shaped containers (that's
    # literally what the effect does), never for a roof/tree/matrix.
    if target.get("large") and "Curtain" in target["name"]:
        return shared_large_key or rng.choice(SAFE_LARGE_WEIGHTED + ["curtain_close", "curtain_open"])
    # Cone/tunnel-shaped large props get shape-specific motion rather than the
    # generic flat-canvas pool. Direct feedback: "the mega tree needs some
    # more of its own unique effects, we're using it like another matrix a lot
    # which works but a tree is specifically round and cone shaped and there
    # are different things that work well for it... we can apply the same
    # concepts to the tunnel, I usually treat the tunnel similarly to a mega
    # tree." Spirals/pinwheel wrap a cone the way a flat chase can't, and
    # meteors run down its axis. Deliberately NOT taking `shared_large_key`
    # here -- the cross-canvas cohesion rule exists so flat canvases match
    # each other, but a cone matching a flat roofline is what made the tree
    # look like "just another matrix" in the first place.
    if target.get("large") and target["name"] in CONE_SHAPED:
        # Shaders stay in the mix (they were explicitly liked, and they're
        # what the cone-motion overlay layers ON TOP of) alongside the
        # cone-native motion. Roughly half shader / half wrapping motion.
        # Cone-native MOTION stays cone-specific, but when the roll lands on a
        # shader it uses the SAME shader every other large canvas is running
        # this section. "Sometimes we have a red shader on the mega tree and a
        # rainbow one on the tunnel matrix and a black and white one on the
        # garage roof and it's a little disjointed." That was this branch
        # deliberately opting out of `shared_large_key` -- right for the
        # wrapping motion, wrong for the shaders, because a shader carries a
        # whole colour world with it and three different ones visible at once
        # never resolve into a look.
        # Weighted toward shaders. "I see them on the matrices but I don't
        # think I see them on the tunnel matrix or mega tree as much, let's
        # make sure they're included. I know we use the cone effects for those
        # but the shaders can be used too." Cone-native motion stays in the
        # pool, it just no longer crowds the shaders out.
        pick = rng.choice(["spirals", "meteors_down", "chase_mid_out",
                           *PALETTE_DRIVEN_SHADERS, *PALETTE_DRIVEN_SHADERS])
        if pick.startswith("shader_") and shared_large_key \
                and shared_large_key.startswith("shader_"):
            return shared_large_key
        return pick
    if target.get("large"):
        return shared_large_key or rng.choice(SAFE_LARGE_WEIGHTED)


    if cat == "icicle":
        # Was 2-in-3 Wave. Icicle rows read as horizontal lines across the
        # windows, so an over-weighted Wave here is a big part of what looked
        # like the same left-right jagged gesture everywhere.
        return rng.choice(["meteors_icicle", "chase_mid_out", "chase_bounce",
                           "chase_lr"])
    if cat == "single_line":
        # Widened to the vocabulary measured in Krists's own sequences, where
        # outline groups carry Bounce far more than anything else, use head
        # fades rather than head-and-tail, and about half the chases carry the
        # Fireworks FX inside them. The old four-way choice was three settings
        # deep; this is the same gestures with his parameters plus the drift
        # and flurry speeds he uses to separate a reveal from a fill.
        return rng.choice([
            "chase_bounce_pp", "chase_bounce_pp", "chase_bounce_drift",
            "chase_lr_pp", "chase_rl_pp", "chase_mid_out_pp", "chase_mid_in_pp",
            "chase_dual", "chase_spark", "chase_spark_lr",
            "chase_bounce", "chase_mid_out", "chase_lr", "chase_rl"])
    if cat == "arch":
        # Deliberate, readable sweeps -- NOT fast multi-rotation blur.
        return rng.choice(["chase_lr", "chase_rl", "chase_mid_out", "chase_mid_in"])
    if cat == "matrix":
        # No Curtain, no Ripple here -- both have been occasional red-flash
        # sources on matrix-category targets across multiple rounds even
        # with the buffer-style fix. Curtain stays reserved for genuine
        # curtain-shaped containers (LARGE_CANVAS); real shaders substitute
        # for the texture Ripple/Curtain would have added.
        # Ripple IS used on matrices in Krists's sequences -- 450 times on the
        # front window matrices alone -- but as a POLYGON that implodes, not
        # the plain circle that caused the old red flashes. Pinwheel likewise,
        # as a Sweep rather than a spinning fan. Both were excluded here on
        # evidence from a version of the effects that he does not use.
        if escalation >= 2:
            return rng.choice(["chase_mid_out", "ripple_hex_in", "pinwheel_sweep",
                               *SHADER_KEYS])
        return rng.choice(["chase_lr", "chase_rl", "wash",
                           "ripple_hex_in", "ripple_hex_out", "pinwheel_sweep",
                           "bars_expand", "bars_hexpand", "bars_alt_up"])
    if "hanging sphere" in target["name"].lower():
        return rng.choice(["fill_up", "fill_down", "sphere_sparkle",
                           "strand_stripes", "ripple_hex_out", "meteors_down",
                           "pinwheel_4", "chase_bounce_drift"])
    if cat == "tree":
        return rng.choice(["chase_mid_out", "meteors_down", "spirals",
                           "chase_bounce_drift"])
    # scattered decorative props -> step prop to prop
    # SUBMODELS TRACE THE PROP'S SHAPE. "For the submodels on the flakes and
    # spinners and rosa grande and grand illusion and those kinds of props, I
    # like what we're doing and want to add some more effects for variety, for
    # example doing single strand chases and fans and radials and stuff like
    # that on the submodels instead of the whole model can look cool because
    # it traces a certain shape."
    #
    # A submodel is usually a named part with real geometry -- a ring, a
    # spoke set, an arm, a petal row. Running a chase or a radial burst along
    # THAT rather than across the whole prop is what draws the shape out. The
    # pool is deliberately shape-flavoured: linear chases (including the
    # centre-out pair, which reads as a fan opening on a spoke or arm set),
    # and radial bursts that expand from the middle of a ring.
    if "/" in target["name"]:
        return rng.choice([
            "chase_mid_out", "chase_mid_in",      # fans open / close
            "chase_lr", "chase_rl",               # traces along the part
            "chase_pair_lr", "chase_pair_rl",     # two dots meeting
            "shockwave", "ripple_explode",        # radial from the centre
            "pinwheel",                           # rotates about the middle
            # From Krists's sequences: a six-point polygon collapsing into the
            # middle is a snowflake closing on its own geometry, and it is what
            # he actually uses on flake and snowflake parts. Rays are the same
            # idea drawn as spokes rather than rings.
            "ripple_hex_in", "ripple_hex_out", "ripple_rays",
            "chase_bounce_pp", "chase_spark",
            "bars_alt_up", "bars_alt_down", "bars_expand", "bars_hexpand",
            "pinwheel_4", "pinwheel_twist", "spirals",
        ])
    if n >= 4:
        return rng.choice(["sweep_right", "sweep_left"] if wide else ["sweep_up", "sweep_down"])
    return rng.choice(["chase_lr", "chase_rl", "shockwave",
                       "chase_bounce_pp", "ripple_hex_in"])


MIN_FLASH_GAP_S = 0.28  # ~3.5Hz cap -- feedback: faster than 2-4/sec reads as random strobing


def split_flashes_and_bursts(times_sec, min_gap_s=MIN_FLASH_GAP_S, burst_run=3):
    """Onsets closer together than min_gap_s can't read as individual on-beat
    flashes -- feedback was explicit that fast flashing feels random rather
    than musical, and should be a movement pattern instead. Split a time
    series into isolated times (kept as flashes) and runs of `burst_run`+
    closely-spaced onsets (returned as (start, end) spans meant for a
    chase/sweep instead of discrete hits)."""
    times = sorted(times_sec)
    if not times:
        return [], []
    flashes: list[float] = []
    clusters: list[tuple[float, float]] = []
    run = [times[0]]
    for t in times[1:]:
        if t - run[-1] < min_gap_s:
            run.append(t)
        else:
            (clusters.append((run[0], run[-1])) if len(run) >= burst_run
             else flashes.extend(run))
            run = [t]
    (clusters.append((run[0], run[-1])) if len(run) >= burst_run else flashes.extend(run))
    return flashes, clusters


def detect_big_hits(analysis, kick_times=None, downbeats=None,
                     energy_ceiling=0.5, min_height=0.35, min_prominence=0.15,
                     min_gap_s=6.0):
    """Prominent RMS peaks in low/mid-energy sections -- the 'big drop, slow
    fade, big drop, slow fade' pattern common in an intro. Distinct from the
    predrop-cutout detector: that one requires near-silence immediately
    before the jump, but this pattern decays to a moderate baseline (not
    silence) between hits, so cutout detection only catches the first one."""
    rms = np.array(analysis.spectrum.rms_energy)
    times = np.array(analysis.spectrum.rms_times)
    peaks, _ = find_peaks(rms, height=min_height, distance=40, prominence=min_prominence)
    hits = []
    for p in peaks:
        t = float(times[p])
        sec = next((s for s in analysis.sections if s.start_time <= t < s.end_time), None)
        if sec is None or sec.energy_level >= energy_ceiling:
            continue
        # A BROADBAND RMS PEAK IS NOT A MUSICAL EVENT. Krists diagnosed this
        # exactly: "sometimes they happen kind of randomly. I see the waveform
        # for the song has a spike there and that's what seems to trigger it,
        # but usually it's not where drums kick in or a chorus starts, it's
        # just where some voice or instrument spiked the waveform briefly,
        # probably in the higher frequencies." RMS sums all frequencies, so a
        # belted vocal note or a bright synth stab is indistinguishable from a
        # drum hit -- and this gesture is a full white flash across half the
        # display, so a false positive is extremely conspicuous.
        #
        # Requiring a real KICK underneath is the frequency discrimination the
        # RMS curve cannot provide, and requiring a DOWNBEAT makes it a
        # structural moment rather than an accident of level.
        if kick_times is not None:
            if not any(abs(t - k) <= 0.12 for k in kick_times):
                continue
        if downbeats:
            if not any(abs(t - d) <= 0.12 for d in downbeats):
                continue
        if hits and t - hits[-1] < min_gap_s:
            continue
        hits.append(t)
    return hits


def detect_kick_pattern(mp3_stem_dir: Path, beat_times, tol_s=0.09):
    """Find the REAL kick-drum onset times, and whether the kick is present
    persistently enough to justify a dedicated rhythm layer.

    Was `detect_four_on_floor`, returning only a yes/no genre guess and using
    the generic beat grid for placement. Direct correction on an R&B track
    that measured a persistent-kick coverage of 84% (well above the 80%
    house-track-validated threshold): "it's not really four on the floor,
    it's more of an r&b beat... I liked the rhythm on the arches but it
    wasn't aligned with the kick drum." The coverage METRIC was measuring the
    right thing (a kick present near most beats) but the OLD caller then
    placed hits on an idealized uniform beat grid regardless -- which reads
    as a mechanical dance-music pulse no matter what the underlying pattern
    actually does, and drifts audibly out of sync whenever the real kick is
    syncopated rather than landing exactly on every beat.

    Fix: return the actual onset times detected in the low band (a kick
    lives below ~150Hz), not just a coverage score. Callers place hits AT
    THOSE TIMES instead of assuming a uniform grid -- house tracks (kick on
    every beat) and R&B/hip-hop grooves (kick on a real, possibly syncopated
    pattern) both come out sounding like themselves instead of both getting
    flattened onto "four on the floor."

    The coverage score is still useful as a gate for "is a dedicated kick
    layer worth having at all" (below ~80%, the kick isn't a strong enough
    through-line to build a persistent visual layer around) -- kept for
    that. Measured: house track 0.93, Gooey (bass music) 0.58, this R&B
    track 0.84 -- all three "is a kick clearly present" calls were correct;
    what was wrong was assuming that answer alone also meant "hits land on
    every beat like a metronome."
    """
    drums_path = mp3_stem_dir / "drums.wav"
    if not drums_path.exists() or len(beat_times) < 8:
        return [], False, 0.0
    y, sr = librosa.load(str(drums_path), sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = librosa.util.normalize(S[freqs <= 150].sum(axis=0))
    times = librosa.times_like(low, sr=sr, hop_length=256)

    kick_frames = librosa.onset.onset_detect(onset_envelope=low, sr=sr,
                                              hop_length=256, backtrack=False)
    kick_times = librosa.frames_to_time(kick_frames, sr=sr, hop_length=256).tolist()

    beats = np.asarray(beat_times, dtype=float)
    beats = beats[beats < times[-1]]
    if len(beats) < 8:
        return kick_times, False, 0.0
    peaks = []
    for bt in beats:
        m = (times >= bt - tol_s) & (times <= bt + tol_s)
        peaks.append(float(low[m].max()) if m.any() else 0.0)
    peaks = np.asarray(peaks)

    active = peaks[peaks > 0.02]          # beats where drums are playing at all
    if len(active) < 8:
        return kick_times, False, 0.0
    median_kick = float(np.median(active))
    coverage = float((active >= 0.35 * median_kick).mean())
    return kick_times, coverage >= 0.80, coverage


def detect_legato_phrase(mp3_stem_dir: Path, sections, min_note_s=0.25, max_note_s=3.0,
                          max_notes=60):
    """Note-duration segmentation for a sustained/legato melodic line (e.g. a
    string pad), distinct from detect_cues.py's arpeggio detector (tuned for
    short *repeating cyclical* note patterns) and synth-stab detector (tuned
    for percussive attacks) -- a legato phrase has irregular note lengths and
    doesn't necessarily repeat on a short cycle, so neither existing detector
    catches it. Restricted to chorus-labeled sections (or the highest-energy
    label if none are marked chorus) since that's where a repeating melodic
    hook typically lives. Returns notes classified short/long by a duration
    split, in time order, ready to drive a directional sway effect."""
    other_path = mp3_stem_dir / "other.wav"
    if not other_path.exists():
        return []
    y, sr = librosa.load(str(other_path), sr=22050, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    env_times = librosa.times_like(onset_env, sr=sr)

    chorus_secs = [s for s in sections if s.label == "chorus"]
    if not chorus_secs:
        if not sections:
            return []
        best_label = max({s.label for s in sections},
                          key=lambda lbl: max(s.energy_level for s in sections if s.label == lbl))
        chorus_secs = [s for s in sections if s.label == best_label]

    notes = []
    for sec in chorus_secs:
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, backtrack=True, units="frames", onset_envelope=onset_env)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_times = sorted(t for t in onset_times if sec.start_time <= t < sec.end_time)
        for i, t in enumerate(onset_times):
            end = onset_times[i + 1] if i + 1 < len(onset_times) else sec.end_time
            dur = end - t
            if min_note_s <= dur <= max_note_s:
                strength = float(np.interp(t, env_times, onset_env))
                notes.append({"start": float(t), "end": float(end), "duration": float(dur),
                               "strength": strength})

    if not notes:
        return []
    # A lead melodic line reads as more prominent onsets than incidental
    # background texture -- keep only the strongest ones if there are more
    # candidates than a single repeating phrase would plausibly need. 426
    # raw onsets across a chorus was clearly "most of the mix", not "the
    # melody", on the first attempt at this detector.
    if len(notes) > max_notes:
        notes = sorted(notes, key=lambda n: -n["strength"])[:max_notes]
        notes.sort(key=lambda n: n["start"])
    durations = sorted(n["duration"] for n in notes)
    split = durations[int(len(durations) * 0.6)]
    for n in notes:
        n["long"] = n["duration"] >= split
    return notes


def assign_voices_to_props(voices: dict, cast: list[dict],
                           rng: random.Random) -> dict[str, dict]:
    """Give each detected instrument voice one dedicated prop for the WHOLE
    song. Returns {voice_name: target}.

    Held song-wide rather than re-rolled per section deliberately -- see
    VOICE_PROP_PREFS. A viewer needs several seconds of the same prop doing
    the same instrument before the association lands, and repeats of that
    association across the song are what turn it from coincidence into a
    noticed detail.

    Props already carrying a dedicated, differently-timed behaviour are
    excluded so two mechanisms never fight over the same prop: `symmetric`
    ensembles (they move as one unit and would break), and arch-category
    groups (they have the beat-paced cascade).
    """
    assignments: dict[str, dict] = {}
    claimed: set[str] = set()

    prominence = prominence_of      # module-level; see its docstring

    # Strongest/most-present voices choose first, so if the display runs out
    # of suitable props it's a marginal voice that misses out, not the kick.
    order = sorted(voices.items(), key=lambda kv: -len(kv[1]))
    for vname, voice in order:
        prefs = VOICE_PROP_PREFS.get(vname)
        if prefs is None or len(voice) < 8:
            continue
        candidates = [
            t for t in cast
            if t["name"] not in claimed
            and _base_model(t["name"]) not in claimed
            and not t.get("symmetric")
            and t["cat"] != "arch"
            and (prefs["large_ok"] or not t.get("large"))
            and prominence(t) >= 1
            # Floods are NOT an instrument. Handing them a voice gave every
            # song a flood swelling every couple of beats for its whole
            # runtime: "the floods do a slow pulse too often, seems like in
            # every song." A flood is a stage-lighting tool, not a performer --
            # see flood_pass() for the two things they are actually good at.
            and not t.get("slow")
        ]
        if not candidates:
            continue
        # Rank across ALL preferred categories at once, with recognisability
        # outranking category preference. An earlier version broke out of the
        # loop at the first category with ANY candidate, which handed the
        # chords to "GE Grand Illusion Spokes GRP" (a 50-member sub-detail
        # family that happens to be categorised single_line) instead of
        # falling through to Orbs -- a nameable prop one category later.
        # Shape preference only matters once the prop is recognisable at all.
        def rank(t: dict) -> tuple:
            try:
                cat_rank = prefs["cats"].index(t["cat"])
            except ValueError:
                cat_rank = len(prefs["cats"])
            return (-prominence(t), cat_rank, -t["n"])

        # ROTATE ACROSS SONGS. `min(candidates, key=rank)` is fully
        # deterministic, so every song handed the same voice to the same prop:
        # hats->Stars, chords->C9s, bass->Orbs, kick->Mini Trees, identically
        # on all ten songs. That in turn left the SAME leftovers free to anchor
        # the beat, which is why nine of ten songs ended up marking the kick on
        # Arches and Front Window Matrices. "I definitely want to rotate so
        # they're not all the same."
        #
        # Quality discipline is unchanged: take the best (prominence, category)
        # tier exactly as before, and only rotate WITHIN that tier, keyed by
        # song name. So a song never gets a worse prop for a voice -- it gets a
        # different equally-good one.
        best_tier = rank(min(candidates, key=rank))[:2]
        tier = [t for t in candidates if rank(t)[:2] == best_tier]
        tier.sort(key=lambda t: t["name"])
        pick_rng = random.Random(f"{SONG_NAME}|voice|{vname}")
        picked = pick_rng.choice(tier) if tier else min(candidates, key=rank)
        assignments[vname] = picked
        claimed.add(picked["name"])
        # Claim the PARENT too. "As long as we don't use multiple submodels on
        # the same prop for different rhythms" -- claiming the base model
        # stops a second voice landing on a sibling submodel of the same
        # physical prop, which would put two competing rhythms on one object
        # and destroy the attribution the whole system exists to create.
        claimed.add(_base_model(picked["name"]))
        for m in picked["members"]:
            claimed.add(_base_model(m))

    # --- reinforce the song's rhythmic drivers with EXTRA props ------------
    # "Let's use more props for the main drum and bass and obvious stab and
    # rhythmic driver voices, it's okay if multiple props all follow the same
    # beat, if anything that makes it more clear and rhythmic and dramatic.
    # Use smaller props for following melodies or vocals."
    #
    # Which voice dominates is song-specific and must be measured, not
    # hardcoded -- "in Gooey in particular I want more focus on the bass...
    # Not sure how to generalize that... somehow we need to identify if it's
    # bass heavy then let's emphasize that or if another voice is very
    # prominent." Prominence = how much of the song that voice actually
    # occupies (onset count x mean onset strength), so a bass-led track
    # naturally elects bass and a drum-led one elects kick/snare.
    RHYTHM_DRIVERS = ("kick", "snare", "bass", "chord")
    prominence_score = {
        v: len(voices[v]) * (sum(voices[v].strengths) / max(1, len(voices[v])))
        for v in assignments if v in voices
    }
    drivers = sorted((v for v in assignments if v in RHYTHM_DRIVERS),
                     key=lambda v: -prominence_score.get(v, 0))[:2]
    extra_for: dict[str, list[dict]] = {v: [] for v in assignments}
    for vname in drivers:
        prefs = VOICE_PROP_PREFS[vname]
        pool = [t for t in cast
                if t["name"] not in claimed
                and _base_model(t["name"]) not in claimed
                and not t.get("symmetric")
                and t["cat"] != "arch"
                and (prefs["large_ok"] or not t.get("large"))
                and prominence(t) >= 2
                and t["n"] >= 2
                and not t.get("slow")]        # floods: see flood pass
        pool.sort(key=lambda t: (-prominence(t), -t["n"]))
        for extra in pool[:2]:
            extra_for[vname].append(extra)
            claimed.add(extra["name"])
            claimed.add(_base_model(extra["name"]))
            for m in extra["members"]:
                claimed.add(_base_model(m))
    return assignments, extra_for


def main():

    def _probe(_lbl):
        """Which pass ate my effect? PROBE=30.0,36.0 python generate_song.py ...

        This class of bug -- an effect placed correctly and then quietly
        removed or moved by a later pass -- has come up often enough that
        guessing is a waste of time. Counting the same moment after every
        major pass localises it in one run.
        """
        import os
        _env = os.environ.get("PROBE")
        if not _env:
            return
        if _env.startswith("mt:"):
            _rows = [(round(e.start_time_ms / 1000.0, 2),
                      round(e.end_time_ms / 1000.0, 2))
                     for e in effects
                     if e.model_name == "Mega Tree" and e.effect_name == "Shader"
                     and float(_env.split(":")[1] or 10)
                     < e.start_time_ms / 1000.0
                     < float(_env.split(":")[2] or 20)]
            print(f"    MT {_lbl:32s} {sorted(_rows)}")
            return
        for _t in [float(x) for x in _env.split(",") if x.strip()
                   and x.strip()[0].isdigit()]:
            _n = sum(1 for _e in effects
                     if abs(_e.start_time_ms / 1000.0 - _t) < 0.35
                     and any(h in _e.model_name.lower()
                             for h in ("vertical", "horizontal", "c9", "outline")))
            print(f"    PROBE {_lbl:28s} t={_t}: {_n}")
    show_config = load_show_config(SHOW_PATH)
    layout = json.loads((SHOW_PATH / ".generation" / "layout.json").read_text())
    positions = {m["name"]: m for m in layout["models"]}
    analysis = SongAnalysis.model_validate_json((GEN_DIR / "analysis.json").read_text())
    cues = json.loads((GEN_DIR / "cues.json").read_text())
    # Every moment a drop could land on, used by change_marks() to keep a
    # pattern change from happening just BEFORE one.
    _drop_marks = sorted({round(float(ev["drop_time"]), 2)
                          for k in ("predrop_cutouts", "energy_drops",
                                    "instrument_dropouts")
                          for ev in cues.get(k, []) if ev.get("drop_time")})

    cast = build_cast(show_config, positions)
    covered = sum(t["n"] for t in cast)
    print(f"Cast: {len(cast)} targets covering {covered} models "
          f"({sum(1 for t in cast if t['kind']=='group')} groups, "
          f"{sum(1 for t in cast if t['kind']=='model')} models)")

    effects: list[EffectPlacement] = []
    cue_log: list[dict] = []

    def _caller_line() -> int:
        """Line in this file of the pass that asked for a placement.

        sys._getframe rather than traceback.extract_stack: this runs once per
        effect, ~10k times a song, and extract_stack builds a whole formatted
        stack each call.
        """
        f = sys._getframe(1)
        while f is not None and f.f_code.co_name in ("add", "add_target",
                                                     "_caller_line"):
            f = f.f_back
        return f.f_lineno if f is not None else 0

    REMOVALS: list[dict] = []
    GUARD_LOG: list[dict] = []

    def drop_effects(all_effects, doomed_ids, reason):
        """Remove effects and RECORD what went, and why.

        Provenance answers "which pass placed this". The other half of every
        inert-fix mystery in this project is "which pass took it away again" --
        the run-up dip shredding ambient holds, the cohort lock dragging chord
        hits off the beat, the fills refilling what the final dropout cleared.
        Routing removals through one place makes that answerable too: each
        entry records the reason, the count, and which passes had placed the
        effects being removed.
        """
        if not doomed_ids:
            return all_effects
        gone = [e for e in all_effects if id(e) in doomed_ids]
        by_src: Counter = Counter(e.settings.get("X_Src", "0") for e in gone)
        REMOVALS.append({
            "reason": reason,
            "count": len(gone),
            "from_lines": {k: v for k, v in by_src.most_common(6)},
            "span": [min((e.start_time_ms for e in gone), default=0) / 1000.0,
                     max((e.end_time_ms for e in gone), default=0) / 1000.0],
        })
        return [e for e in all_effects if id(e) not in doomed_ids]

    def add(target_name, layer, key, start_ms, end_ms, palette, extra=None):
        # A HANGING SPHERE NEVER JUST FLASHES -- enforced here for the same
        # reason as the rule below: this is the one function every placement
        # goes through, and the spheres were reached mostly by GROUP-level
        # flashes expanded to member names, which never consult the per-target
        # pool in beat_hit_key. Measured: 92% of the `On` on the spheres
        # arrived that way, so adding fills and sparkle to that pool changed
        # almost nothing until this existed.
        if key in FLAT_KEYS and "hanging sphere" in (target_name or "").lower():
            key = random.Random(f"sphere|{target_name}|{start_ms}").choice(
                ["fill_up", "fill_down", "sphere_sparkle", "strand_stripes",
                 "shockwave", "chase_mid_out", key])
        # NOTHING BIG FLAT-FLASHES -- enforced HERE, at the one function every
        # placement goes through.
        #
        # "There are some parts where I notice the whole mega tree and arch
        # tunnel / tunnel matrix flashing but I thought we made a rule not to
        # do that because they're too big to flash, it's just too much for the
        # eyes in person. And I'm not talking about pulsing, it's full flashing
        # from on to off and on again."
        #
        # There WAS such a rule, in add_target, and it covered only cones.
        # Measured on Gooey v126: the Mega Tree still made 75 full
        # dark-to-bright-to-dark transitions and the Tunnel Matrix 33, because
        # the drop staging, big-hit and signature passes reach `add()`
        # directly -- either by model name or through a group broadcast -- and
        # never saw the guard. Putting it at the bottom of the stack is the
        # only way it actually holds.
        #
        # Substituted, not dropped: the moment still lands, as a burst that
        # travels across the prop rather than a slab of colour switching on.
        # A SWELL IS NOT A FLASH. What makes a big prop unwatchable is coming
        # on and going off; a sharp attack followed by a long decay is the
        # "gently pulse brightness" gesture Krists likes, just bigger. Without
        # this escape the no-flash guard also strips the drops of their punch
        # -- measured on Gooey, the 1:00 drop peaked at 175 before the guard
        # and 89.7 after, because the big canvases' hit was being converted to
        # a travelling pattern. Anything with a real fade-out is allowed
        # through at full brightness.
        _fade_out = float((extra or {}).get("T_TEXTCTRL_Fadeout", 0) or 0)
        if key in FLAT_KEYS and _fade_out >= 0.8:
            pass
        elif key in FLAT_KEYS and target_name in FLASH_SPLIT:
            # A GROUP holding a big prop: flash its other members instead of
            # the group. Substituting a shaped effect is not enough here --
            # a Shockwave across a yard-wide group still sweeps the Mega Tree
            # from dark to full and back, which measured slightly WORSE than
            # the flat flash it replaced (42 -> 45 transitions). The gesture
            # belongs to the small props in the group; the big one simply sits
            # it out and keeps running whatever it was showing.
            for _m in FLASH_SPLIT[target_name]:
                add(_m, layer, key, start_ms, end_ms, palette, extra)
            return
        elif key in FLAT_KEYS and target_name in NO_FLAT_PROPS:
            key = ("chase_mid_out" if target_name in CONE_SHAPED
                   else "shockwave")
        name, settings = EFFECTS[key]
        if extra:
            settings = {**settings, **extra}
        # SNAP TO THE RENDER FRAME GRID. xLights renders at FRAME_MS (25ms =
        # 40fps here) and rounds each effect's start to the nearest frame.
        # Measured on Fara: of 262 non-frame-aligned anchor flashes, 129
        # rendered a frame EARLY and 129 a frame late, purely on where the
        # sub-frame remainder happened to fall. That is +-12.5ms of jitter on
        # what is supposed to be a metronomic pulse, and irregular timing is
        # what reads as sloppy -- a consistent small offset would not. "I want
        # to make sure the pulsing (and anything rhythmic really) is always
        # aligned perfectly with the beat, otherwise it looks sloppy."
        #
        # Rounding here instead of letting xLights do it makes every placement
        # land on exactly the frame we intend, deterministically. It cannot
        # remove the inherent 25ms resolution, but it removes the jitter, which
        # is the part the eye actually notices.
        start_ms = int(round(start_ms / FRAME_MS) * FRAME_MS)
        end_ms = int(round(end_ms / FRAME_MS) * FRAME_MS)
        # PROVENANCE. Every placement records the line of this file that asked
        # for it. `add` is the one chokepoint every effect goes through, so one
        # tag here covers the whole generator, and the marker survives
        # model_copy (which is why an id()-based scheme did not).
        #
        # This exists because the passes interact in ways that are not
        # predictable by reading them: three separate ordering bugs in one day
        # (the replacement cut running before the fills that add effects, the
        # pulse ceiling running before the pass that shortens spans, cohesion
        # counting coverage from before its own fills), and then six straight
        # hypotheses about a strobe regression that each measured completely
        # inert. Guessing does not scale; asking the file does.
        #
        # Read it back with custom_tools/explain_moment.py.
        settings = {**settings, "X_Src": str(_caller_line())}
        effects.append(EffectPlacement(
            model_name=target_name, layer=layer, effect_name=name,
            start_time_ms=start_ms,
            end_time_ms=max(end_ms, start_ms + FRAME_MS),
            settings=settings, palette=palette,
        ))

    def log(t, kind, desc):
        cue_log.append({"time": t, "kind": kind, "desc": desc})

    SPAN_SENSITIVE_KEYS = {"curtain_close", "curtain_open", "ripple_implode", "ripple_explode"}

    FLAT_KEYS = {"on_hit", "on_pop", "on_swell", "wash"}
    # Every prop too big to flash: the cones and every large canvas, plus the
    # member models a group placement broadcasts to (the Tunnel Matrix is
    # reached both ways). Derived from the cast, so a prop Krists adds later
    # is covered without a code edit.
    NO_FLAT_PROPS = set(CONE_SHAPED)
    for _t in cast:
        if _t.get("large") or _t["name"] in CONE_SHAPED:
            NO_FLAT_PROPS.add(_t["name"])
            NO_FLAT_PROPS.update(_t.get("members") or [])
    # ...and any GROUP that contains one of them. This is where the Mega Tree
    # flashing actually came from: `All Yard Props` is a group, so it is not
    # itself "large" and passed the guard, but the Mega Tree is a member and
    # xLights broadcasts the flash to it. Measured on Gooey, 45 of the tree's
    # full dark-to-bright transitions after the intro trace to 300ms flashes
    # placed on that one group.
    #
    # A flat flash on a group is only ever as safe as its biggest member.
    # Membership has to come from the LAYOUT, not from the cast.
    #
    # A cast target's `members` list is our own bookkeeping and is not what
    # xLights broadcasts to: measured, the cast thinks "All Yard Props" has 80
    # members and that the Mega Tree is not among them, while the xLights
    # group definition says it is -- so a flat flash placed on that group lit
    # the tree anyway, and every guard keyed on the cast's view missed it.
    # What matters is what the renderer will do.
    _xgroups = layout.get("xlights_model_groups") or {}
    # ...but never expand to a model Krists has switched off. This list is
    # placed on MEMBER NAMES directly, bypassing the cast, so it was the one
    # route by which disabled models still got sequenced: after he deleted the
    # unused props and left others inactive, Gooey still carried 548 effects on
    # Candy Cane 1-4, all of them Active="0". "They should not be sequenced if
    # they're not visible or not active."
    _inactive_models: set[str] = set()
    try:
        _rgbtxt = (SHOW_PATH / "xlights_rgbeffects.xml").read_text(errors="replace")
        _inactive_models = {_m.group(1) for _m in
                            re.finditer(r'<model name="([^"]*)"([^>]*)', _rgbtxt)
                            if 'Active="0"' in _m.group(2)}
    except OSError:
        pass
    FLASH_SPLIT: dict[str, list[str]] = {}
    for _gname, _mem in _xgroups.items():
        if _gname in NO_FLAT_PROPS or not _mem:
            continue
        if any(_m in NO_FLAT_PROPS for _m in _mem):
            _safe = [_m for _m in _mem
                     if _m not in NO_FLAT_PROPS
                     and _m.split("/")[0] not in _inactive_models]
            if _safe:
                FLASH_SPLIT[_gname] = _safe
            else:
                NO_FLAT_PROPS.add(_gname)

    def add_target(target, layer, key, start_ms, end_ms, palette, extra=None):
        # NOTHING FLAT-FLASHES A CONE. "There are parts in Gooey where the
        # whole tunnel flashes but it's too big a prop to flash. It can pulse
        # and do all the movements and shaders and everything you've been doing
        # with matrices and the mega tree, but the whole thing can't just
        # flash." The beat passes already respect this via beat_hit_key, but
        # the drop staging, big-hit and signature passes all place `on_hit`
        # directly on every cast target -- 6 flat flashes on the tunnel in
        # Gooey. Substituting a wrapping burst keeps the moment without the
        # slab of colour.
        if target.get("name") in CONE_SHAPED and key in FLAT_KEYS:
            key = "chase_mid_out"
        # NO PLASMA ON MATRICES. "I don't like the plasma effect so let's
        # remove that completely from the matrices." Guarded here rather than
        # by editing the pools, because plasma_organic reaches a matrix from
        # several directions -- SAFE_LARGE, SAFE_LARGE_WEIGHTED, the matrix
        # branch of the escalation roll, and the shader fallbacks -- and a
        # pool edit would leave whichever path was missed still placing it.
        # It stays available to cones, trees and curtains, where it wraps
        # rather than mottles.
        # PLASMA IS RETIRED. It was first removed from matrices; it then turned
        # up on the Garage Roof at Gooey 3:30 and was flagged again ("I want to
        # get rid of that effect"). Two complaints, never once praised, and the
        # library now carries 51 shaders that cover the same mottled-texture
        # ground better -- so it goes everywhere rather than being chased from
        # one prop family to the next.
        # A HIGH-FLICKER SHADER NEEDS A CANVAS WITH REAL EXTENT.
        # Anything made of small discrete objects -- orbs, stars, mini trees,
        # flakes, canes -- samples a tiny patch of the shader, so its whole
        # brightness swings with the texture. Only props that can actually
        # hold a picture get the busy shaders.
        if key.startswith("shader_") and small_prop_flicker(key) > CALM_ON_SMALL:
            _nm2 = target.get("name", "")
            _big_enough = (target.get("cat") == "matrix"
                           or _nm2 in CONE_SHAPED
                           or "matri" in _nm2.lower()
                           or _nm2 == "Garage Roof")
            if not _big_enough or target.get("per_model") \
                    or _nm2 in PER_MODEL_GROUPS:
                _calm = [k for k in SHADER_KEYS
                         if small_prop_flicker(k) <= CALM_ON_SMALL]
                if _calm:
                    key = random.Random(f"calm|{_nm2}|{key}").choice(_calm)
        # BARS ARE RETIRED. "Let's just remove all the diagonal bars effects
        # completely, it's just not working and it's everywhere. Maybe it's not
        # bars, bars are just what I visually see."
        #
        # Every Bars-based key is substituted with the nearest thing that gives
        # the same directional movement without drawing a bar: a chase travels
        # the same way, a shockwave opens from the middle the way expand did.
        # The effect and its whole parameter vocabulary -- alternate
        # directions, tilt, cycles curves -- goes with it.
        if key in _BARS_SUBSTITUTE:
            key = _BARS_SUBSTITUTE[key]
        if key == "plasma_organic":
            key = SHADER_KEYS[0] if SHADER_KEYS else "chase_mid_out"
        # A FLOOD IS ONE PIXEL. Measured: every flood in this show is 3
        # channels -- a single RGB node. It has no spatial extent, so a shader,
        # a Shockwave, Spirals or a chase placed on it cannot render as a
        # pattern; it renders as that pattern's brightness over time, sampled
        # at one point. That is flicker, and at shader speed it is fast
        # flicker: Gooey measured 2521 flood turn-ons with 52% of the gaps
        # shorter than a quarter note, while the flood-discipline pass -- which
        # counts EFFECTS -- saw nothing wrong and dropped 5.
        #
        # "The floods are doing a lot of strobing and rapid flashing... I
        # thought we had already ensured the floods would never go faster than
        # quarter or half notes", and floods "can give people migraines".
        # A flood can express exactly two things, colour and brightness, so it
        # gets a flat wash and nothing else.
        if _is_flood(target) and key not in FLAT_KEYS:
            key = "wash"
        """Place on the target's own name, UNLESS it's synthetic (not a real
        xLights model/group -- see build_cast's leftover-prefix step), in
        which case broadcast to every member individually so nothing ends up
        orphaned or, worse, silently mis-targeted onto an unrelated model
        that happens to share the synthetic name.

        Also decides the Per-Preview spanning buffer for a handful of
        effects: only valid when actually addressing a real xLights GROUP
        (multiple physical models to span as one canvas). Forcing it onto an
        individual model -- including via a synthetic broadcast, which
        places on ONE real model at a time -- risks an unsupported buffer/
        model combination, which is a plausible cause of effects rendering
        as a solid error color instead of the intended motion."""
        if key in SPAN_SENSITIVE_KEYS and target["kind"] == "group" and not target.get("synthetic"):
            extra = {**(extra or {}), **SPAN}
        # The buffer style a placement will ACTUALLY get: `extra` wins over the
        # recipe, but the recipe is where most styles are declared. Every guard
        # below has to test this rather than `extra` alone -- checking `extra`
        # meant a style set in the recipe was invisible to them, which is how
        # Per Model Per Preview kept being rewritten to Per Model Default on
        # 96% of chases despite three separate guards meant to preserve it.
        def _eff_buffer() -> str | None:
            return (extra or {}).get(
                "B_CHOICE_BufferStyle",
                EFFECTS.get(key, ("", {}))[1].get("B_CHOICE_BufferStyle"))

        _PER_MODEL_OK = ("Per Model Per Preview", "Per Model Single Line")
        # A centred effect never spans -- see is_centred(). Per-model instead,
        # so each prop gets a whole shape of its own rather than a slice of one
        # whose middle falls in the gap between props.
        if is_centred(key) and (extra or {}).get("B_CHOICE_BufferStyle") == "Per Preview":
            extra = {**extra, **PERMODEL}
        # A GROUP OF SUBMODEL STRIPS NEVER SPANS.
        #
        # `Other Horizontals` is a group whose members are the matrix outline
        # submodels (`Front Left Window Matrix/Horizontals`, ...). Spanning
        # builds ONE canvas across seven physically separate strips on
        # different matrices, so a chase does not travel along anything -- the
        # light jumps from strip to strip, which is the erratic flashing that
        # survived four earlier fixes. Per-model gives each strip its own
        # buffer so they all chase along their own length together, which is
        # what the verticals (real adjacent models) already do naturally.
        if (target["kind"] == "group" and target["members"]
                and sum(1 for m in target["members"] if "/" in m)
                    >= max(1, len(target["members"]) // 2)):
            # ...unless a PER-MODEL style was asked for explicitly. These
            # rules exist to stop a group SPANNING one canvas across separate
            # props; Per Model Per Preview and Per Model Single Line already
            # render each model in its own buffer, so replacing them with Per
            # Model Default only discards the preview orientation Krists's own
            # sequences rely on -- it is his most-used buffer by a wide margin
            # (2624 of his outline chases) and this was silently rewriting it.
            if _eff_buffer() not in _PER_MODEL_OK:
                extra = {k: v for k, v in (extra or {}).items()
                         if k != "B_CHOICE_BufferStyle"}
                extra = {**extra, **PERMODEL}
        # A depth-oriented prop can never take a spanning buffer -- it is one
        # column wide on the shared canvas, so spanning lights the whole thing
        # as a slab. Per-model gives it its own buffer and a real render.
        # The whole GROUP goes per-model when ANY member points front-to-back.
        # Measured: the C9s group was spanning (`Per Preview`) while Single
        # Line-13/14 carried no buffer style at all, so those two rendered as a
        # single column -- "the single lines are back to acting like single
        # pixels, worse than before." Per-model is also what keeps the group
        # CONSISTENT: every member renders in its own buffer, so they all show
        # the same gesture instead of two of them behaving differently.
        if (target["name"] in DEPTH_ORIENTED
                or any(m in DEPTH_ORIENTED for m in target["members"])):
            # ...unless a PER-MODEL style was asked for explicitly. These
            # rules exist to stop a group SPANNING one canvas across separate
            # props; Per Model Per Preview and Per Model Single Line already
            # render each model in its own buffer, so replacing them with Per
            # Model Default only discards the preview orientation Krists's own
            # sequences rely on -- it is his most-used buffer by a wide margin
            # (2624 of his outline chases) and this was silently rewriting it.
            if _eff_buffer() not in _PER_MODEL_OK:
                extra = {k: v for k, v in (extra or {}).items()
                         if k != "B_CHOICE_BufferStyle"}
                extra = {**extra, **PERMODEL}
        # Per-model rendering wins over any spanning buffer style: for a group
        # of discrete scattered objects (stars, mini trees, floods) spanning
        # is exactly the thing that turns motion into strobing. Applied last
        # so it overrides the SPAN injection above.
        if (target.get("per_model") and target["kind"] == "group"
                and not target.get("synthetic")
                and _eff_buffer() not in _PER_MODEL_OK):
            # Per Model Per Preview satisfies this rule already: it renders
            # each member in its own buffer. Overwriting it with Per Model
            # Default threw away the preview orientation for no gain, and
            # between them these five override sites were rewriting 96% of
            # the chases back to one buffer style.
            extra = {**(extra or {}), **PERMODEL}
        # "Default" ON A GROUP IS A TRAP.
        #
        # For a group, xLights' Default buffer is one canvas in which each
        # MEMBER MODEL is a single cell. A Color Wash across nine vertical
        # strips therefore paints nine pixels, not nine strips. Measured on
        # This Version of You: at 27.5s, with a chord wash on Corner
        # Verticals, eight of the nine members were pure black and the ninth
        # had a single lit pixel -- while an `On` effect (which carries no
        # buffer style at all) lit all nine to 241 at 47.5s.
        #
        # This is why so many changes have measured inert in the render while
        # looking right in the .xsq: the effects were placed, and rendered
        # into almost no pixels. Every recipe defaulting to "Default" -- wash,
        # shockwave and the rest -- was affected on every group target.
        # The style has to be read from the RECIPE as well as from `extra` --
        # most effects carry "Default" in their own settings dict and never
        # mention it in `extra`, so a check against `extra` alone silently
        # matched nothing.
        _eff_style = (extra or {}).get(
            "B_CHOICE_BufferStyle",
            EFFECTS.get(key, ("", {}))[1].get("B_CHOICE_BufferStyle"))
        if (target["kind"] == "group" and not target.get("synthetic")
                and _eff_style == "Default"):
            extra = {**(extra or {}), **PERMODEL}
        # Cone/tunnel props need their strings stacked for a built-in 2D
        # effect to render as the real pattern rather than as lines. Shaders
        # are exempt -- they render their own coordinate space and already
        # look right. See VSTACK.
        if target["name"] in CONE_SHAPED and not key.startswith("shader_"):
            # Vertical Stack for BUILT-IN 2D effects only. Those genuinely need
            # the cone's strings stacked or they render as lines, which is what
            # Krists saw. Shaders are exempt again: they render in their own
            # coordinate space, and stacking them fills the cone uniformly --
            # "it's only now that they started fully just flashing" pinned it
            # to the round where that exemption was removed. Big props pulsing
            # brightness under another effect is wanted; the whole prop lighting
            # as one slab is not.
            extra = {**(extra or {}), **VSTACK}
        if target.get("broadcast"):
            # One effect per member MODEL, slightly staggered, so the .xsq
            # genuinely contains per-prop effects rather than relying on a
            # group buffer style to imply them. See BROADCAST_GROUPS.
            for i, m in enumerate(target["members"]):
                off = int(i * BROADCAST_STAGGER_S * 1000)
                add(m, layer, key, start_ms + off, end_ms + off, palette, extra)
        elif target.get("synthetic"):
            for m in target["members"]:
                add(m, layer, key, start_ms, end_ms, palette, extra)
        else:
            add(target["name"], layer, key, start_ms, end_ms, palette, extra)

    beat_ms = 60000.0 / max(analysis.beats.tempo, 60.0)
    beats = analysis.beats.beat_times
    # Snap downbeats onto the beat grid.
    #
    # `downbeat_times` and `beat_times` come from different estimators and do
    # not perfectly agree: measured 26-62ms of disagreement on both test
    # songs. Everything bar-aligned (shader/pattern changes, feature motion)
    # is placed on downbeats while everything pulse-aligned (the kick pulse,
    # voice hits) is placed on beats/real onsets -- so that 40-ish ms shows up
    # as the two families of effects not quite landing together. Direct
    # feedback: "the shaders often are not on beat... the shaders suddenly
    # come on the wrong beat." Snapping to the nearest beat costs nothing
    # (a downbeat IS a beat musically) and removes the discrepancy for every
    # downbeat-placed effect at once.
    if beats:
        _b = np.asarray(beats, dtype=float)
        # Snap ONLY when a real beat is close by. The two grids don't cover
        # identical spans -- some downbeats sit in stretches with no detected
        # beat at all, and a naive nearest-match moved one by 21 SECONDS,
        # which would have stacked many bar boundaries onto a single instant.
        # Anything further than a quarter-beat away keeps its original time.
        _tol = (beat_ms / 1000.0) * 0.25
        downbeats, _snapped, _worst = [], 0, 0.0
        for d in analysis.beats.downbeat_times:
            nearest = float(_b[int(np.argmin(np.abs(_b - d)))])
            if abs(nearest - d) <= _tol:
                if abs(nearest - d) > 0.001:
                    _snapped += 1
                    _worst = max(_worst, abs(nearest - d))
                downbeats.append(nearest)
            else:
                downbeats.append(float(d))
        if _snapped:
            print(f"Beat grid: snapped {_snapped}/{len(downbeats)} downbeats "
                  f"onto the beat grid (max correction {_worst * 1000:.0f}ms)")
    else:
        downbeats = analysis.beats.downbeat_times
    # Beat + half-beat grid, for filtering out isolated one-off transients
    # that don't land anywhere near the felt pulse -- direct feedback: a
    # flash with "no rhythm to it... tied to some random one-off transient"
    # looks glitchy and should be dropped rather than shown.
    half_beat_s = beat_ms / 2000.0
    rhythm_grid = np.array(sorted(beats + [b + half_beat_s for b in beats]))

    def near_beat_grid(t: float, tolerance_s: float = 0.09) -> bool:
        if len(rhythm_grid) == 0:
            return True
        idx = np.searchsorted(rhythm_grid, t)
        candidates = rhythm_grid[max(0, idx - 1):idx + 1]
        return bool(len(candidates)) and float(np.min(np.abs(candidates - t))) <= tolerance_s

    # (The old whole-drums-stem `drum_onsets` list that used to drive the
    # PULSE role is gone -- superseded by the per-instrument voices below.
    # It was accurate but unreadable: one onset list conflating kick, snare,
    # hats and fills means a prop driven from it represents "drums in
    # general", which is precisely the chaos this round set out to fix.)

    # Real kick-drum onset times, used as the timing source for anything that
    # wants to feel "locked to the kick" -- the arch cascade below, and the
    # persistent pulse in PASS 2. Computed once, up front, because the arch
    # cascade (inside PASS 1) needs it and used to fall back to the generic
    # beat grid instead. See detect_kick_pattern()'s docstring for why using
    # the generic grid instead of real onset times was the actual bug behind
    # "the rhythm wasn't aligned with the kick drum."
    kick_times, kick_persistent, kick_coverage = detect_kick_pattern(
        SHOW_PATH / "stems" / SONG_NAME, beats)
    kick_times_arr = np.asarray(kick_times, dtype=float)

    # --- Per-instrument voices, drum fills, and the melody line ------------
    # See custom_tools/instrument_voices.py. This replaces driving the
    # rhythmic layer from whole-stem onset lists, which was the root of "it
    # can just seem chaotic in a lot of places": a whole drums stem is a
    # kick AND a snare AND hats collapsed together, so a prop driven from it
    # represents "drums in general", which nothing in the audience can lock
    # onto.
    from instrument_voices import (detect_drum_fills, detect_instrument_voices,
                                    extract_melody_notes)

    stem_dir = SHOW_PATH / "stems" / SONG_NAME
    drum_fills = detect_drum_fills(stem_dir, beats, beat_ms)
    voices = detect_instrument_voices(stem_dir, beats, beat_ms, drum_fills)

    # SECTION BOUNDARIES MOVE TO THE HIT THE BAND LANDS ON -- once, here, so
    # every pass downstream inherits the corrected time rather than each one
    # snapping its own way.
    #
    # The section detector reports where its analysis window crosses, which can
    # sit BETWEEN two beats; rounding to the nearest is then a coin flip.
    # Measured on Gooey: the section at 15.46s sits exactly between beats at
    # 15.16 and 15.86, rounds DOWN, and the kick and bass land at 15.85. "The
    # second big drum hit in the beginning around 0:16 is off -- the shaders
    # change before the kick/bass hits." A whole beat early, and early reads
    # far worse than late, because the change pre-empts the thing it marks.
    #
    # Correcting the section itself also answers the other half: "the section
    # changes and shaders changing all together at the same time as each
    # other." They can only agree if they are all reading the same number.
    if beats:
        _bt_arr = np.asarray(beats, dtype=float)
        _beat_s = beat_ms / 1000.0
        _moved = 0
        for _sec in analysis.sections:
            _lo, _hi = _sec.start_time - 0.12, _sec.start_time + 1.15 * _beat_s
            _cands = [float(t) for _vn in ("kick", "snare", "bass")
                      if (_v := voices.get(_vn)) is not None
                      for t in _v.onsets if _lo <= t <= _hi]
            # ONLY LOOK AHEAD IF NOTHING IS ALREADY HERE. Taking the first
            # onset in the window unconditionally pushes a boundary that was
            # already right onto the NEXT hit: Gooey's 3:01 drop moved from
            # 181.37 to 182.07 and the canvases split across two times. If a
            # hit sits at the boundary, the boundary is already correct.
            _here = [t for t in _cands if abs(t - _sec.start_time) <= 0.15]
            # ...and never move further than three-quarters of a beat. The
            # correction exists to fix a boundary that rounded to the WRONG
            # SIDE of a beat, which is at most half a beat of error. A larger
            # move means the nearest hit is a different musical event, and
            # taking it pushes the boundary onto the next one: Gooey's 3:01
            # moved a full beat to 182.07 and split the canvases across two
            # times, where 181.37 was right.
            _cap = 0.75 * _beat_s
            _near = [t for t in _cands if abs(t - _sec.start_time) <= _cap]
            _target = (_sec.start_time if _here or not _near else min(_near))
            _snap = float(_bt_arr[int(np.argmin(np.abs(_bt_arr - _target)))])
            if abs(_snap - _sec.start_time) > 0.02:
                _sec.start_time = _snap
                _moved += 1
        # Keep the section list consistent: each end is the next one's start.
        for _i in range(len(analysis.sections) - 1):
            analysis.sections[_i].end_time = analysis.sections[_i + 1].start_time
        if _moved:
            print(f"Section boundaries: {_moved} moved onto the drum/bass hit "
                  f"they mark (of {len(analysis.sections)})")

    # INSTRUMENT ENTRANCES ARE STRUCTURAL MOMENTS.
    #
    # "In WhatsApp Shakespeare in the intro, the shaders come in right before
    # the main drum beat starts, which takes away from the impact of that drum
    # entrance -- the shader should start right when the section changes and
    # the drums start."
    #
    # Measured there, the drums arrive at 8.68s while the nearest section
    # boundary is 6.48s: the detector simply does not mark the entrance, so
    # nothing had anything to align to and the shader landed 1.18s early. A
    # voice arriving after several seconds of absence is exactly the kind of
    # moment a viewer reads as a change, whether or not the boundary detector
    # agrees, so those times join the grid that change_marks snaps to.
    _entrances: list[float] = []
    for _vn in ("kick", "snare", "bass", "chord"):
        _v = voices.get(_vn)
        if _v is None or len(_v) < 4:
            continue
        _prev = None
        for _t in sorted(_v.onsets):
            if _prev is None or _t - _prev >= 4.0:
                _entrances.append(float(_t))
            _prev = _t
    # ...and the mirror image: a sustained energy STEP DOWN is a section
    # change too. The generator detects steps up (drops) and nothing for the
    # other direction, so a chorus ending had nothing to mark it:
    #
    #   "In Harmony at 2:12 the chorus ends and an ambient part starts but the
    #    lights stay the same... you don't see the section change, it should
    #    be more visible."
    #
    # Measured there, the music halves from 0.659 to 0.345 across 132s while
    # the nearest section boundary sits at 121.8s -- the detector simply misses
    # it. A level that falls by a third and STAYS down is a structural moment
    # whatever the boundary detector thinks.
    _rms_e = np.asarray(analysis.spectrum.rms_energy)
    _rt_e = np.asarray(analysis.spectrum.rms_times)
    _win = 4.0
    _step = float(_rt_e[1] - _rt_e[0]) if len(_rt_e) > 1 else 0.023
    _w = max(2, int(_win / _step))
    # Deduped against OTHER step-downs only, not against entrances: Harmony
    # has 26 detected entrances spaced about 5.5s apart, so anything suppressed
    # by proximity to one is suppressed everywhere, and the 2:12 drop-down --
    # the exact moment Krists reported -- was being discarded.
    _stepdowns: list[float] = []
    for _i in range(_w, len(_rms_e) - _w):
        _before = _rms_e[_i - _w:_i].mean()
        _after = _rms_e[_i:_i + _w].mean()
        if _before > 1e-6 and _after / _before <= 0.65:
            # Advance to where the level actually SETTLES, not where it
            # starts to fall: the first sample past the midpoint between the
            # old and new plateaus. Snapping from the fall's start put
            # Harmony's 2:12 change at 131.08 -- a full bar early.
            _mid = (_before + _after) / 2.0
            _j = _i
            while _j < len(_rms_e) - 1 and _rms_e[_j] > _mid:
                _j += 1
            _t = float(_rt_e[_j])
            # Snap forward to the next downbeat. The detector fires where the
            # level STARTS falling -- 130.6 on Harmony, while the new section
            # begins at 132.0 -- the same off-by-a-moment the drop snapping
            # had to fix in the other direction. A section change lands on a
            # downbeat, so that is where the look should turn over.
            _nd = next((d for d in downbeats if d >= _t - 0.05), None)
            if _nd is not None and _nd - _t <= 3.0:
                _t = float(_nd)
            if all(abs(_t - x) > 6.0 for x in _stepdowns):
                _stepdowns.append(_t)
    if _stepdowns:
        print(f"Energy step-downs: {len(_stepdowns)} section endings marked "
              f"({', '.join(f'{x:.1f}' for x in _stepdowns[:6])}"
              f"{'...' if len(_stepdowns) > 6 else ''})")
    _entrances = sorted(set(round(x, 2) for x in _entrances + _stepdowns))

    # SPLIT THE SECTION THERE. Marking the step-down as a moment made the
    # shader change at 2:12, and the lights still got BRIGHTER across it
    # (19.9 -> 31.7) because staging works per section: target counts, palette
    # and energy all come from a section that runs straight through the
    # change. The boundary detector missed it, so the section has to be split
    # by hand and each half given its own measured energy -- then everything
    # downstream reacts to the quieter half on its own terms.
    _split = 0
    for _sd in _stepdowns:
        for _si, _sec in enumerate(analysis.sections):
            if _sec.start_time + 3.0 < _sd < _sec.end_time - 3.0:
                _tail = type(_sec)(**{**_sec.__dict__})
                _tail.start_time = _sd
                _sec.end_time = _sd
                for _s2 in (_sec, _tail):
                    _m = (_rt_e >= _s2.start_time) & (_rt_e < _s2.end_time)
                    if _m.any():
                        _s2.energy_level = float(
                            _rms_e[_m].mean() / max(_rms_e.max(), 1e-9))
                analysis.sections.insert(_si + 1, _tail)
                _split += 1
                break
    if _split:
        print(f"Energy step-downs: split {_split} sections so the quieter half "
              f"is staged on its own terms")

    # Ambient passages are DETECTED here, before anything is staged, because
    # the ordinary section path has to know to stand down inside them. Placing
    # the ambient treatment later while the section path kept running produced
    # exactly what Krists saw: "the shaders keep changing rapidly... there are
    # just too many changes, it's supposed to feel smooth." Measured, the
    # matrices changed every 2.6s on This Version of You and every 2.7s on
    # World Is Mine, with the section path contributing most of it on top of
    # the ambient holds.
    ambient_passages: list[dict] = []
    try:
        from ambient_sections import analyse as _amb_analyse
        ambient_passages = [p for p in _amb_analyse(SONG_NAME)["passages"]
                            if p["ambient"] and p["dur"] >= 8.0]
    except Exception as _e:
        print(f"  (ambient-passage analysis unavailable: {_e})")

    def in_ambient(t: float) -> bool:
        return any(a["t0"] <= t < a["t1"] for a in ambient_passages)
    if _entrances:
        print(f"Instrument entrances: {len(_entrances)} treated as structural "
              f"({', '.join(f'{x:.1f}' for x in _entrances[:6])}"
              f"{'...' if len(_entrances) > 6 else ''})")
    if DISABLED_VOICES:
        _dropped = [v for v in voices if v in DISABLED_VOICES]
        voices = {k: v for k, v in voices.items() if k not in DISABLED_VOICES}
        if _dropped:
            print(f"Voices disabled: {', '.join(_dropped)} (see DISABLED_VOICES)")
    voice_props, voice_extra_props = assign_voices_to_props(
        voices, cast, random.Random(f"{SONG_NAME}|voice-assign"))
    # Reverse lookup: which voice (if any) does this prop represent all song?
    # Extra reinforcement props map to the same voice, so they automatically
    # perform it AND get reserved from the generic passes by the same checks.
    prop_voice = {t["name"]: vname for vname, t in voice_props.items()}
    for vname, extras in voice_extra_props.items():
        for t in extras:
            prop_voice[t["name"]] = vname
    if any(voice_extra_props.values()):
        print("Reinforced drivers: " + ", ".join(
            f"{v}+{[t['name'] for t in ts]}"
            for v, ts in voice_extra_props.items() if ts))
    print("Instrument voices: " + (", ".join(
        f"{v}({len(voices[v])})->{t['name']}" for v, t in voice_props.items())
        or "none detected"))
    print(f"Drum fills: {len(drum_fills)}")

    # Melody extraction runs pyin over the whole vocals stem, which is slow
    # (~1-2 min) and completely deterministic for a given song -- cache it
    # next to the other per-song generated data so re-running the generator
    # while iterating on effect choices doesn't pay that cost every time.
    # --- harmony: chord CHANGES as musical boundaries ----------------------
    # Agreed use: "chords are good for changes, not overall feel." So the key/
    # mode is deliberately NOT driving palette choice (all four songs measured
    # so far are minor, so a mode-driven palette would reduce cross-song
    # variety rather than add it). What IS used is where the harmony actually
    # moves -- those are real musical boundaries, and changing a big canvas's
    # pattern there instead of on an arbitrary bar count makes the change feel
    # motivated.
    from harmony import analyse_harmony

    harmony = analyse_harmony(MP3_PATH, GEN_DIR / "harmony.json")
    chord_changes = list(harmony.get("changes") or [])
    _key = harmony.get("key") or {}
    print(f"Harmony: key={_key.get('key', '?')} "
          f"(conf {_key.get('confidence', 0):.2f}), "
          f"{len(harmony.get('chords') or [])} chord segments, "
          f"{len(chord_changes)} change points")

    # The shared bar grid every large canvas lands on. Collected as spans are
    # placed, then used by the cohort-lock pass to pull any straggler onto the
    # same boundaries -- see the sync pass near the run-up dip.
    cohort_marks: set[int] = set()
    n_spd_swing = 0
    # (canvas, cycle-start) pairs already given their one span -- see hold_spans.
    _cycle_claimed: set[tuple] = set()

    def hold_spans(target, layer, key, section, palette, seat, energy_level):
        nonlocal n_spd_swing
        """Place a multi-bar HOLD as bar-aligned spans carrying a beat pulse.

        Every large canvas has to come through here, whatever ROLE it happened
        to draw that section. It previously did not, and that was the whole of
        "the matrices seem oddly aligned... sometimes some of them are pulsing
        and others aren't":

          * ROLE_FEATURE subdivided the section on change_marks() and gave
            every span a beat-locked brightness curve
          * ROLE_TEXTURE and ROLE_OUTLINE placed ONE flat span across the
            whole section with no curve at all

        Two matrices sitting side by side, running the SAME shader, would
        start and stop at different times and only one of them would breathe.
        Measured on Gooey's intro: Smoke Drift ran 7600-13500 on Fence Matrix
        1 with a pulse and 7225-13500 on Garage Roof without one. Role is an
        internal staging concept; nothing about it should be visible as two
        neighbouring canvases disagreeing.

        Returns the number of spans that actually got a pulse curve.
        """
        # In a drop/fade passage the CYCLE is the unit, not the section: one
        # shader from the kick that starts the drop until the kick that starts
        # the next one. The section containing a cycle's start owns it; every
        # other section overlapping that cycle places nothing, so the canvas
        # holds one pattern across the whole fade instead of switching partway
        # through it.
        # Inside an ambient passage the ambient pass owns the big canvases:
        # one slow hold per passage rather than a new pattern every few
        # seconds. Without this both were placing, and the section path's
        # faster cadence is what the eye followed.
        if target.get("large") and in_ambient(section.start_time):
            return 0
        cyc = next((c for c in drop_cycles
                    if c[0] < section.end_time and c[1] > section.start_time), None)
        if cyc is not None and target.get("large"):
            # Claimed by the FIRST section that reaches this cycle with this
            # canvas active -- not by the section containing the cycle's
            # start. Whether a canvas is active is decided per section, so
            # tying ownership to the start section meant that if that section
            # happened not to light this canvas, the entire 11-second cycle
            # got nothing and the span vanished instead of moving.
            # The dedicated drop-cycle pass owns these canvases here; the
            # per-section path must not also place on them. Claiming the cycle
            # opportunistically from whichever section happened to be active
            # was tried and covered only one of Gooey's five cycles, because
            # in a quiet intro the big canvases are rarely in the active set.
            return 0

        bars_per_change = 4 if target.get("large") else 2
        marks = change_marks(section, bars_per_change)
        if len(marks) < 2:
            return 0
        n = 0
        for j in range(len(marks) - 1):
            s_ms = int(marks[j] * 1000)
            e_ms = int(marks[j + 1] * 1000)
            if key in SHADER_SWAY:
                k = SHADER_SWAY[key][j % 2]
            else:
                k = key if j % 2 == 0 else REVERSE.get(key, key)
            # A DROP GETS A DIFFERENT SHADER, not the same one restarted.
            #
            # "The main last drop in World Is Mine, the shader changes before
            # the drop, then it drops out and comes back for the drop and it's
            # less impactful because it's the same thing we just saw a few
            # seconds ago."
            #
            # Measured there: a change at 213.60s (a section start) put Liquid
            # Chrome on the canvases, and the span beginning at the 218.87s
            # drop was Liquid Chrome again. Snapping a span boundary onto the
            # drop is not enough -- the boundary has to bring something new
            # through it, or the blackout resolves to the picture that was
            # already there.
            if any(abs(marks[j] - d) <= 0.35 for d in _drop_marks):
                _alt = [q for q in (PALETTE_DRIVEN_SHADERS or SHADER_KEYS)
                        if q != k and not is_centred(q)]
                if _alt:
                    k = _alt[hash(f"{SONG_NAME}|dropshader|{s_ms}") % len(_alt)]
            depth = 0.5 + 0.5 * min(1.0, energy_level / 0.7)
            if not k.startswith("shader_"):
                depth *= 0.6
            base_pal = palette
            if target.get("large") and is_palette_driven(k):
                base_pal = palette_variant(palette, seat)
            # PULSE ONE LAYER, NOT THE WHOLE MATRIX. "I notice sometimes we're
            # layering multiple shaders or effects on top of each other on the
            # matrices which looks great, maybe we could pulse just one of the
            # shaders and not have to pulse the whole matrix? ... for pulsing
            # the matrices, I think we're doing it a little too much."
            #
            # The feature layer takes the pulse; the base texture underneath
            # runs steady, so a canvas carrying both reads as one element
            # breathing against a stable ground rather than the whole prop
            # breathing as a block -- and it cuts how much pulsing there is
            # overall, which was the other half of the note.
            #
            # Even on the feature layer, not every span: a deterministic 30%
            # are held flat so a canvas alternates between breathing and
            # running steady instead of pulsing continuously.
            # SPEED SWING ON THE BEAT. "In Meridian at 0:10 when the drums
            # come in the diagonal stripes go across the house really fast on
            # the first beat and then suddenly on the next beat they go really
            # slowly -- that's a fantastic effect! I wish it did that again at
            # other points on the beat, fast movement, slow movement, fast,
            # slow."
            #
            # That happened by accident, where two spans of the same shader at
            # different speeds happened to abut on a beat. Made deliberate
            # here: a span is cut on the beat grid and the same shader
            # alternates between a fast and a slow variant across the cuts, so
            # the motion itself keeps time instead of a brightness pulse doing
            # it. Applied to a minority of spans on the big canvases -- the
            # whole point is that it is a surprise.
            # Always on a span that BEGINS at a structural moment, plus a
            # share of the rest. "I think we can use it a bit more, it doesn't
            # need to be only a surprise, it's a great focal point when
            # there's a strong rhythm or when a drop happens or on section
            # changes and such." Tying it to the moment makes it read as a
            # reaction to the music rather than a random flourish.
            _on_moment = any(abs(s_ms / 1000.0 - d) <= 0.35 for d in _drop_marks) or \
                any(abs(s_ms / 1000.0 - sec.start_time) <= 0.35
                    for sec in analysis.sections)
            if (target.get("large") and k.startswith("shader_")
                    and (_on_moment
                         or random.Random(f"{SONG_NAME}|spdswing|{target['name']}|{s_ms}"
                                          ).random() < 0.40)):
                # REPETITION MAKES IT INTENTIONAL. "I think it needs a little
                # bit more repetition to look intentional, sometimes it does
                # it once or twice and then does a different thing and it can
                # almost look like it's just glitching... So maybe if it's
                # quarter or a regular beat we stick with the same change
                # effect for a few bars, or if it's on a new section or drop
                # then we only do it once and in some impactful way."
                #
                # Two shapes, chosen by where the span begins. On a section or
                # drop it swings ONCE, big, as part of the moment. Anywhere
                # else it needs at least four alternations -- two bars of
                # fast/slow -- or it is not placed at all, because a couple of
                # swings in isolation reads as a fault rather than a figure.
                _sec_beats = [b for b in beats if s_ms / 1000.0 <= b < e_ms / 1000.0]
                if _on_moment:
                    _cuts = _sec_beats[:2]
                else:
                    _cuts = _sec_beats[::2][:12]
                    if len(_cuts) < 5:
                        _cuts = []
                if len(_cuts) >= 2:
                    _en, _base_set = EFFECTS[k]
                    for _vi, _mult in ((0, 2.4), (1, 0.35)):
                        _vk = f"{k}__spd{_vi}"
                        if _vk not in EFFECTS:
                            _st2 = dict(_base_set)
                            _st2["E_SLIDER_Shader_Speed"] = str(max(15, int(
                                int(_st2.get("E_SLIDER_Shader_Speed", 400)) * _mult)))
                            EFFECTS[_vk] = (_en, _st2)
                    _bounds = [s_ms / 1000.0] + _cuts[1:] + [e_ms / 1000.0]
                    for _ci in range(len(_bounds) - 1):
                        _a, _b2 = int(_bounds[_ci] * 1000), int(_bounds[_ci + 1] * 1000)
                        if _b2 - _a < 200:
                            continue
                        add_target(target, layer, f"{k}__spd{_ci % 2}",
                                   _a, _b2, base_pal)
                    cohort_marks.update((s_ms, e_ms))
                    n_spd_swing += 1
                    continue
            _skip = (layer != L_MOTION
                     or random.Random(f"{SONG_NAME}|pulseskip|"
                                      f"{target['name']}|{s_ms}").random() < 0.30)
            if _skip:
                cohort_marks.update((s_ms, e_ms))
                add_target(target, layer, k, s_ms, e_ms, base_pal)
                continue
            pal_k = beat_pulse_palette(base_pal, beats, s_ms, e_ms, depth)
            if pal_k is not base_pal:
                n += 1
            cohort_marks.update((s_ms, e_ms))
            add_target(target, layer, k, s_ms, e_ms, pal_k)
        return n

    def change_marks(section, bars_per_change: int) -> list[float]:
        """Bar boundaries at which a big canvas should change its pattern,
        preferring downbeats that coincide with a real chord change.

        Falls back cleanly to a fixed every-N-bars cadence when a song has
        little harmonic movement to work with -- measured, Gooey has one
        54-second chord block, so chord-driven boundaries would leave it with
        almost no changes at all. Genre determines how much this contributes,
        and both outcomes are correct for their song.
        """
        sec_db = [d for d in downbeats
                  if section.start_time <= d < section.end_time]
        if not sec_db:
            return []
        half_bar = bar_s / 2.0
        harmonic = [d for d in sec_db
                    if any(abs(d - c) <= half_bar for c in chord_changes)]
        # Only trust chord-aligned marks if they're neither too sparse (the
        # canvas would sit static for ages) nor so dense they'd out-pace the
        # fixed cadence they're replacing.
        floor_n = max(2, len(sec_db) // (bars_per_change * 2))
        ceil_n = max(2, len(sec_db) // max(1, bars_per_change // 2))
        # ONE SHADER PER DROP, where the song is built out of drops.
        #
        # "In the Gooey intro we're still changing shaders too much and
        # starting/stopping them at odd times. It seems like on every
        # drop/fade cycle we use two shaders and the timing of them switching
        # seems random. Let's use just one shader for each loud drop until it
        # fades out all the way and switch shaders on the next drop."
        #
        # A bar cadence cannot express that: the drops are ~11s apart on
        # Gooey's intro and a 4-bar mark lands every ~11s too, so the two
        # grids beat against each other and a canvas changes shader somewhere
        # in the middle of a fade. When a section contains real hit-then-fade
        # events, THEY are the boundaries -- the shader is held for the whole
        # cycle and changes on the next hit.
        # Measured on Gooey: the hits land at 4.8, 26.9, 37.9, 49.0 and the
        # section boundaries at 7.2, 26.0, 38.0, 49.8 -- i.e. the detector has
        # ALREADY split the intro into one section per drop/fade cycle. So the
        # extra shader change is the SUBDIVISION inside a section, not a
        # misplaced boundary: two 4-bar marks inside an 11s cycle is exactly
        # the "on every drop/fade cycle we use two shaders" report. When a
        # section starts on a hit, it gets one shader for its whole length.
        starts_on_hit = any(abs(h - section.start_time) <= 1.2 for h in big_hits)
        hits_in = [h for h in big_hits
                   if section.start_time + 1.2 < h < section.end_time - 1.2]
        if starts_on_hit and not hits_in:
            return [section.start_time, section.end_time]
        if hits_in:
            marks = [section.start_time] + hits_in
        elif floor_n <= len(harmonic) <= ceil_n:
            marks = harmonic
        else:
            marks = sec_db[::bars_per_change]
        # A DROP MUST LOOK DIFFERENT FROM WHAT CAME RIGHT BEFORE IT.
        #
        # "I notice we often switch to a new shader and color scheme before
        # the drop, and the drop itself is just the lights going off and then
        # back on with the same thing they showed before. A drop needs to look
        # different from what came right before."
        #
        # Bar marks and drops are independent grids, so a change lands a few
        # seconds early about as often as not -- and then the blackout hits and
        # the SAME pattern resumes, which is the one thing a drop must not do.
        # Any mark in the run-up is pulled out and the drop time itself becomes
        # the mark, so the pattern the drop reveals is genuinely new.
        # SNAP TO THE STRUCTURAL MOMENT, don't just avoid it.
        #
        # "Harmony has the same 'shader changes at the wrong time' issue, it
        # changes at 1:14 for example but the section change is at 1:17." And
        # on Cruise: "the shader changes more dramatically a few seconds later
        # after the drop, but that change would have been more effective right
        # at the drop."
        #
        # Deleting marks in the run-up fixed the early case only, and only for
        # drops. A change landing just AFTER the moment is the same failure
        # mirrored, and section boundaries matter as much as drops -- they are
        # where a viewer expects the look to turn over. So every mark within
        # 4s either side of a structural moment is MOVED onto it: the change
        # cannot happen early, cannot straggle in late, and the moment itself
        # always gets one.
        # Where a section boundary and an audible entrance disagree, the
        # ENTRANCE wins. The boundary is inferred by a self-similarity
        # detector; the entrance is a measured event in the audio. On
        # WhatsApp Shakespeare the detector put a boundary at 6.48s while the
        # drums actually arrive at 8.68s, and snapping to the nearest moment
        # sent the shader to 6.48 -- further from the drums than where it
        # started. A boundary within 3s of an entrance is dropped in its
        # favour so there is only one moment there to snap to.
        _ent = set(_entrances)
        _secs_m = {sec.start_time for sec in analysis.sections
                   if not any(abs(sec.start_time - e) <= 3.0 for e in _ent)}
        _moments = sorted(set(_drop_marks) | _ent | _secs_m)
        if _moments:
            snapped = []
            for m in marks:
                # An ENTRANCE beats a section boundary whenever both are in
                # range, not just when they nearly coincide. Harmony's single
                # kick at 77.31s sits 3.12s from the boundary at 74.19s -- far
                # enough apart that both survived as moments, so the mark
                # snapped to the nearer one and the colour changed three
                # seconds before the hit it was supposed to mark. The audible
                # event is the one a viewer is waiting for.
                near_e = (min(_ent, key=lambda d: abs(d - m)) if _ent else None)
                if near_e is not None and abs(near_e - m) <= 4.0:
                    snapped.append(near_e)
                    continue
                near = min(_moments, key=lambda d: abs(d - m))
                snapped.append(near if abs(near - m) <= 4.0 else m)
            for d in _moments:
                if section.start_time + 0.5 < d < section.end_time - 0.5:
                    snapped.append(d)
            marks = sorted({round(x, 3) for x in snapped
                            if section.start_time - 0.01 <= x <= section.end_time + 0.01})
        if marks and marks[-1] < section.end_time - 0.2:
            marks.append(section.end_time)
        return marks

    # --- vocals: phrase timing + presence as an energy signal --------------
    # Agreed use: "for vocals use them more for timing or energy." Word
    # IDENTITY is unreliable on sung material (see lyrics.py), so nothing here
    # depends on what is being sung -- only on WHEN, and how much.
    from lyrics import analyse_lyrics

    lyric_data = analyse_lyrics(SHOW_PATH / "stems" / SONG_NAME / "vocals.wav",
                                GEN_DIR / "lyrics.json")
    vocal_phrases_list = lyric_data.get("phrases") or []
    print(f"Vocals: {len(vocal_phrases_list)} sung phrases")

    def vocal_fraction(start_s: float, end_s: float) -> float:
        """How much of a window actually has singing in it."""
        span = max(1e-6, end_s - start_s)
        covered = sum(max(0.0, min(p["end"], end_s) - max(p["start"], start_s))
                      for p in vocal_phrases_list)
        return min(1.0, covered / span)

    melody_cache = GEN_DIR / "melody.json"
    if melody_cache.exists():
        melody_notes = json.loads(melody_cache.read_text())
    else:
        melody_notes = extract_melody_notes(stem_dir, analysis.sections, beat_ms)
        melody_cache.write_text(json.dumps(melody_notes))
    print(f"Melody notes detected: {len(melody_notes)}")

    _beats_arr = np.asarray(beats, dtype=float) if beats else np.array([0.0])

    def beat_phase(t: float, every: int = 1) -> int:
        """Which side of the swing a moment falls on, counted in BEATS.

        Movement direction is chosen from this rather than from a running
        per-hit counter. Direct feedback: "rhythmic movements like that should
        probably be mostly on quarter notes or maybe 8th notes at the fastest.
        Anything on off beats or on the melody seems to just get lost and
        seems random" -- and separately, "changing the direction of the
        movement on quarter notes, especially when there's a four on the floor
        style kick drum."

        Keying direction to the beat NUMBER means every prop reverses at the
        same musical instant no matter which voice triggered it or how many
        hits that voice happened to have. A per-hit counter, which is what
        this replaces, drifts out of phase the moment two props have different
        hit counts -- so the house stopped swaying together and each prop
        looked like it was flipping on its own schedule.

        `every=2` flips per half-note (every 2 beats), `every=4` per bar.
        """
        idx = int(np.searchsorted(_beats_arr, t))
        idx = max(0, min(idx, len(_beats_arr) - 1))
        return (idx // max(1, every)) % 2

    def kick_times_in(start_s: float, end_s: float) -> list[float]:
        """Real kick onsets in a time window, falling back to the generic
        beat grid if the kick isn't reliably present (e.g. a quiet verse) so
        callers always get SOME rhythmic timing source."""
        if kick_persistent and len(kick_times_arr):
            hits = kick_times_arr[(kick_times_arr >= start_s) & (kick_times_arr < end_s)]
            if len(hits):
                return hits.tolist()
        return [b for b in beats if start_s <= b < end_s]

    # The moment the kick genuinely enters, if there's a real kick pattern
    # this song. Direct feedback: "the very beginning of the song is too
    # busy, the song starts slowly and then when the kick drum starts
    # that's a great moment where the song suddenly really jumps into
    # gear" -- the section-level energy banding below was already treating
    # the intro as moderately active (this song's intro sat at energy=0.30,
    # just barely into the SECOND activity tier) because "energy" is an
    # aggregate over the whole section, not "has the song's rhythmic engine
    # actually started yet." Anything entirely before this timestamp gets
    # forced into a deliberately sparser band than the energy-based tiers
    # allow, and the moment itself gets an explicit accent, so the
    # transition reads as a real gear-shift regardless of where the
    # nearest section boundary happens to fall.
    first_kick_time = float(kick_times_arr[0]) if kick_persistent and len(kick_times_arr) else None

    # --- merge sub-phrase section slivers --------------------------------
    # Every section boundary re-rolls the palette, the shared large-canvas
    # shader, the active target set and all the roles. That's the intended
    # unit of change -- but only if a section is a musical phrase. On a
    # harder electronic track the structure detector can shred a passage
    # into ~1-second fragments, and then the whole show re-rolls every
    # second, which is indistinguishable from randomness.
    #
    # Measured, and it's strongly song-specific rather than universal:
    # Gooey has 57 sections of which 47 (82%) are shorter than two bars,
    # concentrated exactly in the first ~60s and last ~60s -- precisely the
    # regions reported as "random shader changes and some other small
    # details that seem somewhat random... the middle part seems good."
    # Its healthy middle is made of proper 25-37s sections. For comparison
    # The WhatsApp Shakespeare and Too Slow have ZERO sub-2-bar sections, so
    # this never showed up on them.
    #
    # Done here rather than in the shared structure detector because tempo
    # is known here, so "too short" can be expressed in BARS (musical) rather
    # than in seconds (arbitrary across tempos).
    bar_s = (beat_ms / 1000.0) * 4.0
    min_section_s = bar_s * 2.0
    merged_sections = []
    for sec in analysis.sections:
        if merged_sections and (merged_sections[-1].end_time
                                - merged_sections[-1].start_time) < min_section_s:
            prev = merged_sections[-1]
            # Absorb into the previous section. Keep whichever label/energy
            # belongs to the longer of the two so a 1s sliver can't rename a
            # real phrase, and so the merged block's energy reflects the part
            # that actually dominates it.
            prev_dur = prev.end_time - prev.start_time
            cur_dur = sec.end_time - sec.start_time
            dominant = prev if prev_dur >= cur_dur else sec
            prev.label = dominant.label
            prev.energy_level = (prev.energy_level * prev_dur
                                 + sec.energy_level * cur_dur) / max(1e-6, prev_dur + cur_dur)
            prev.end_time = sec.end_time
        else:
            merged_sections.append(sec.model_copy())
    # A trailing sliver has nothing after it to absorb it -- fold it back.
    if len(merged_sections) > 1 and (merged_sections[-1].end_time
                                     - merged_sections[-1].start_time) < min_section_s:
        tail = merged_sections.pop()
        merged_sections[-1].end_time = tail.end_time
    if len(merged_sections) < len(analysis.sections):
        print(f"Sections: merged {len(analysis.sections)} -> {len(merged_sections)} "
              f"(min {min_section_s:.1f}s = 2 bars)")
    analysis.sections = merged_sections

    # --- per-section loudness, relative to the song's own loudest section ---
    # Deliberately measured from the audio here rather than reusing
    # `section.energy_level`: that value is normalised and quantised into the
    # coarse activity bands above, so it can't express "same instruments, but
    # noticeably more mellow for a few seconds." This is a continuous 0-1
    # ratio of actual RMS loudness, which is what makes a mellow-then-hard
    # transition visible to the generator at all.
    section_loudness: dict[int, float] = {}
    try:
        _y, _sr = librosa.load(str(MP3_PATH), sr=22050, mono=True)
        _rms = librosa.feature.rms(y=_y)[0]
        _rt = librosa.times_like(_rms, sr=_sr)
        per_sec = []
        for sec in analysis.sections:
            m = (_rt >= sec.start_time) & (_rt < sec.end_time)
            per_sec.append(float(np.mean(_rms[m])) if m.any() else 0.0)
        # Rank-normalise rather than divide by the peak.
        #
        # "The intensity or loudness measurement still seems not sensitive
        # enough, mild drops in energy should still have some effect but it
        # seems to only pick up large changes." Exactly what a peak-relative
        # ratio does wrong: modern masters are compressed, so a genuinely
        # noticeable dip might only move RMS from 0.92 to 0.83 of peak -- a
        # 9% change that the mapping then barely registers. Ranking each
        # section against the song's OWN distribution instead means the
        # quietest section always lands near 0 and the loudest near 1,
        # whatever the absolute spread happens to be, so mild-but-real
        # differences get full expression. Blended 70/30 with the peak ratio
        # so a genuinely uniform song doesn't get artificial swings invented
        # out of noise.
        peak = max(per_sec) or 1.0
        order = sorted(range(len(per_sec)), key=lambda i: per_sec[i])
        rank_of = {si: r / max(1, len(per_sec) - 1) for r, si in enumerate(order)}
        section_loudness = {
            i: 0.7 * rank_of[i] + 0.3 * (v / peak)
            for i, v in enumerate(per_sec)
        }
        quiet = sum(1 for v in section_loudness.values() if v < 0.6)
        print(f"Section loudness: {quiet}/{len(per_sec)} sections below 60% "
              f"of the song's loudest")
    except Exception as exc:                     # never block generation on this
        print(f"Section loudness unavailable ({exc}); using voice count only")

    # --- section instance numbering, for escalation across repeats ---
    label_seen: Counter = Counter()
    section_meta = []
    for s in analysis.sections:
        label_seen[s.label] += 1
        section_meta.append({"instance": label_seen[s.label]})

    # This song's fixed palette family -- see PALETTE_FAMILY_SIZE above.
    fam_rng = random.Random(f"{SONG_NAME}|palette-family")
    # The song's OWN family first (see SONG_PALETTE_FAMILY). Songs not in the
    # table fall back to the old random subset, so a newly added track still
    # generates -- it just doesn't have a hand-picked identity yet.
    if SONG_NAME in SONG_PALETTE_FAMILY:
        song_palette_cycle = list(SONG_PALETTE_FAMILY[SONG_NAME])
    else:
        song_palette_cycle = fam_rng.sample(
            PAL_CYCLE, min(PALETTE_FAMILY_SIZE, len(PAL_CYCLE)))
    print(f"Palette family: {', '.join(song_palette_cycle)}")

    # Identify the section(s) that ARE the song's real climax, to give them
    # deliberately more contrast and richness instead of whatever the
    # ordinary rotation would have handed them. Direct feedback: "identify
    # the big climaxes of the song where we can introduce more contrast and
    # more textures." A climax = the section immediately following the
    # single strongest pre-drop implosion (highest drop_energy of the
    # detected cutouts) -- that IS the moment the whole pre-drop buildup/
    # blackout was built to pay off, so it's the one place a generic flat
    # rotation choice does the most damage. Falls back to the single
    # highest-energy section if no implosion was detected at all.
    climax_sections: set[int] = set()
    top_drop_time: float | None = None
    cutouts_all = cues.get("predrop_cutouts", [])
    if cutouts_all:
        top_drop_time = max(cutouts_all, key=lambda ev: ev.get("drop_energy", 0))["drop_time"]
        for si, s in enumerate(analysis.sections):
            if s.start_time <= top_drop_time < s.end_time:
                climax_sections.add(si)
                break
    if not climax_sections and analysis.sections:
        climax_sections.add(max(range(len(analysis.sections)),
                                key=lambda i: analysis.sections[i].energy_level))

    # =====================================================================
    # PASS 1 -- orchestration: per-section role assignment, sparse staging
    # =====================================================================
    # --- which props will carry MOVEMENT (computed before PASS 1) ---------
    # Needed up here so the ambient layer can get out of their way. See the
    # note in ROLE_TEXTURE below: a full-coverage twinkle/wash underneath a
    # chase leaves the prop 100% lit, and a chase is only visible as the
    # CONTRAST between lit and unlit nodes -- so the movement renders
    # correctly and is still invisible.
    def _ens_rank(t: dict) -> tuple:
        nm = t["name"].lower()
        is_outline_line = any(h in nm for h in
                              ("vertical", "horizontal", "outline", "c9", "icicle"))
        is_sub_detail = nm.endswith("grp") or "/" in t["name"]
        return (not is_outline_line, is_sub_detail, -t["n"])

    ens_targets = sorted(
        (t for t in cast
         if t["cat"] in ("single_line", "icicle")
         and t["n"] >= 2
         and t["name"] not in prop_voice
         and not t.get("symmetric")),
        key=_ens_rank)[:10]
    movement_props = {t["name"] for t in ens_targets} | set(prop_voice)

    # BEAT ANCHORS, chosen up front so the ambient passes below can get out of
    # their way. A prop cannot read as flashing on the kick while a continuous
    # wash holds it at 75%+ lit the whole time: measured on World Is Mine at
    # 96-98s, Mini Tree 1 sat at 76-78% lit through every flash, so the hit
    # went 94 -> 166 and then simply STAYED there. That is a step, not a
    # flash, and it is why the beat was invisible despite hundreds of
    # correctly-timed placements. The anchor needs darkness to come out of.
    #
    # See the beat-accent block below for why there is an anchor at all.
    def split_lr(target: dict) -> tuple[list[str], list[str]] | None:
        """Split a group's members into left and right halves by real position,
        or None if it doesn't split usefully.

        Used for antiphonal beat marking -- one side answering the other every
        beat, which shows the pulse AND the width of the house at the same
        time. Needs a real physical spread (a tight cluster answering itself
        reads as flicker, not as call-and-response) and at least two members a
        side. Derived from WorldPos, so no new xLights groups are required.
        """
        if target["kind"] != "group" or target.get("synthetic") or target["n"] < 4:
            return None
        mem = [m for m in target["members"] if m in positions]
        if len(mem) < 4:
            return None
        xs = [positions[m]["x"] for m in mem]
        if max(xs) - min(xs) < 200:
            return None
        mid = (max(xs) + min(xs)) / 2.0
        left = [m for m in mem if positions[m]["x"] < mid]
        right = [m for m in mem if positions[m]["x"] >= mid]
        if min(len(left), len(right)) < 2:
            return None
        return (left, right)

    # Percentile rank of each section's measured loudness, 0-1. Drives the
    # activity budget above and the accent width below, so every pass agrees
    # on how big a given section is meant to feel.
    _ord = sorted(range(len(analysis.sections)),
                  key=lambda i: section_loudness.get(i, 0.0))
    section_intensity = {}
    _n = max(1, len(_ord) - 1)
    for rank, i in enumerate(_ord):
        section_intensity[i] = rank / _n

    accent_pool = [t for t in cast
                   if t["n"] >= 2
                   and not t.get("slow")          # floods: never the beat marker
                   and prominence_of(t) >= 2]     # recognisable props only
    accent_pool.sort(key=lambda t: t["cx"])       # left-to-right travel
    # The anchor must be a prop that is otherwise QUIET. Ranking the full
    # accent pool put anchors on props that already carry an instrument voice
    # or ensemble movement, and those run SATURATED: measured on World Is Mine,
    # Orb-1 (the bass voice carrier) sits pinned at byte 255 for seconds at a
    # stretch, so a kick flash placed on it changes literally nothing. A prop
    # cannot mark the beat and perform another instrument at the same time --
    # the flash needs somewhere to come FROM.
    # ANY recognisable prop can mark the beat -- "we should also be able to
    # pulse the orbs or any of the other props or verticals or c9s or candy
    # canes etc., any prop or group of props can be used to pulse the main
    # beat." The pool is therefore everything recognisable that is not a slow
    # wash prop, including single hero props (no n>=2 requirement here: a Mega
    # Tree or a Showstopper is perfectly capable of marking a beat on its own).
    # A 2D canvas is the WRONG thing to pulse, even with a pattern. "I don't
    # like pulsing the whole mega tree though, just like pulsing matrices
    # doesn't make sense, nor the tunnel matrix." A matrix or a cone is a
    # drawing surface -- its job is to show a pattern that develops, and
    # interrupting that four times a bar wastes it and looks cheap. Their
    # OUTLINES are fair game (thin, linear, and they read as a flash), and
    # those are separate cast targets, so nothing is lost by excluding the
    # faces here.
    # Big canvases are back IN as anchors -- the previous round excluded them
    # entirely and that was an over-correction: "I liked what the mega tree and
    # arches were doing earlier, I just don't want the mega tree pulsing with a
    # basic color." The objection was to a FLAT COLOUR FILL, not to the Mega
    # Tree keeping the beat. beat_hit_key() gives every large canvas, matrix
    # and cone a pattern-shaped hit (Shockwave / spiral burst) instead of a
    # solid `On`, which is the thing he actually asked for.
    # ...but NOT the upper window matrices. Krists has now said this twice
    # from two directions: "the three upper windows aren't great for flashing
    # because they're matrices", and then "the upper three window matrices
    # often show solid colors or they're mostly blank with just the outlines
    # doing things, when the other matrices all have effects on them."
    # Those are the same fact. An anchor has its ambient fill suppressed so
    # its flash has darkness to come out of -- which is precisely why they
    # looked plain. Off the anchor list, they carry patterns like every other
    # matrix, and their OUTLINES (separate targets) still mark the beat.
    def _is_window_face(t: dict) -> bool:
        n = t["name"].lower()
        return "window matri" in n and "/" not in t["name"]

    # THE BIGGEST PROPS ARE NEVER ANCHORS.
    #
    # An anchor has its ambient fill deliberately suppressed so its flash has
    # darkness to come out of. On a small prop that reads as a beat. On the
    # Mega Tree or the Tunnel Matrix it reads as the whole thing switching
    # off and on -- "it's full flashing from on to off and on again. The
    # biggest props shouldn't do that."
    #
    # Note this is NOT fixed by giving them a shaped hit instead of a flat
    # one: measured on Gooey v126, with beat_hit_key already substituting a
    # Shockwave, the Mega Tree still made 75 full dark-to-bright-to-dark
    # transitions and the Tunnel Matrix 33. What makes it a flash is the prop
    # being DARK between hits, not the shape of the hit. Keeping them out of
    # the anchor role leaves their content running, so the beat shows up as
    # the brightness pulse Krists does want ("I like when they gently pulse
    # brightness under another effect") rather than as a slab switching.
    def _too_big_to_flash(t: dict) -> bool:
        return bool(t.get("large")) or t["name"] in CONE_SHAPED

    anchor_cands = [t for t in cast
                    if not t.get("slow")
                    and not _is_window_face(t)
                    and not _too_big_to_flash(t)
                    and prominence_of(t) >= 2
                    and t["name"] not in movement_props]
    if not anchor_cands:                  # every candidate is already busy
        anchor_cands = [t for t in accent_pool]
    # Shuffled per song, then walked in order, so the anchors differ from song
    # to song instead of falling out of a fixed size ranking every time.
    anchor_rank = sorted(anchor_cands, key=lambda t: t["name"])
    random.Random(f"{SONG_NAME}|beat-anchors").shuffle(anchor_rank)
    # si -> the target that marks every kick in that section.
    section_anchor: dict[int, dict] = {}
    # Sections where the arches LEAP instead of anchoring. Decided here, before
    # anchor selection, so the anchor pass can leave the arches alone -- the
    # leap places on individual arches and the beat guard would otherwise see
    # them as already taken by the group anchor and drop every step.
    leap_sections: set[int] = set()
    _leap_rng = random.Random(f"{SONG_NAME}|leaping-arches")
    for si, section in enumerate(analysis.sections):
        # Gate on DRUMS, not on section energy. "In the quiet part in Fara
        # where there are no drums the arches were still flashing to the beat,
        # so that's a bit confusing." A leap marks the beat, so it needs a beat
        # to mark -- energy_level can be high on a loud passage with no kit at
        # all, which is exactly where it looked wrong.
        _drum = 0.0
        for _v in ("kick", "snare"):
            if _v in voices:
                _drum = max(_drum, voices[_v].density_in(
                    section.start_time, section.end_time))
        if _drum >= 0.7 and _leap_rng.random() < 0.55:
            leap_sections.add(si)

    # ENSEMBLE MODE per section: the outlines/verticals/C9s either MOVE or
    # FLASH, alternating. "I also didn't see the window outlines or verticals
    # doing any pulsing to the beat, they just do the up down or side to side
    # movement. We can alternate, sometimes flash them, sometimes do motion."
    # They were structurally excluded from ever flashing because carrying the
    # unison movement put them in `movement_props`, which the anchor pool
    # filters out. Now that exclusion is per-section instead of permanent.
    ens_flash_sections = set()
    _ens_rng = random.Random(f"{SONG_NAME}|ens-mode")
    for si, section in enumerate(analysis.sections):
        # Was 35% of active sections; raised to 60% because the verticals /
        # horizontals / outlines / C9s still read as movement-only: "I don't
        # see any verts or horizontals or outlines or c9s pulsing, they're
        # still only doing movement." High-energy sections always flash --
        # that's where the "pulsing like crazy" belongs.
        # Never in an ambient passage. This pass flashes the linear props on
        # the beat, and in a drumless passage there is no beat to flash to --
        # measured on This Version of You, the arches were firing 1,2,3,4,4,3,
        # 2,1 every 0.45s straight through the piano intro, which is Krists's
        # "there is something else happening too on some other rhythm and it's
        # distracting". The ambient clearing did not catch it because these
        # land on the motion layer, which that pass deliberately leaves alone.
        if in_ambient(section.start_time):
            continue
        if section.energy_level >= 0.65:
            ens_flash_sections.add(si)
        elif section.energy_level >= 0.40 and _ens_rng.random() < 0.60:
            ens_flash_sections.add(si)

    if anchor_rank:
        ens_names = {t["name"] for t in ens_targets}
        for si, section in enumerate(analysis.sections):
            if section.energy_level < 0.30:
                continue
            pool = anchor_rank
            if si in ens_flash_sections:
                # This section the linear props flash instead of chasing, so
                # they become eligible to anchor the beat.
                # NOT gated on prominence_of >= 2: the window/corner/left/
                # right verticals and horizontals all score 1 (they match the
                # outline/GRP naming heuristic), which is exactly why they
                # could never anchor. They are curated props Krists named
                # directly, and ens_targets is already a ranked shortlist.
                # HORIZONTALS NEVER FLASH. They are the calm layer -- long
                # runs across the top and bottom of each matrix whose whole job
                # is smooth travel. Measured on Aperture: `Other Horizontals`
                # carried 268 smooth 925ms chases AND 35 flashes at one per
                # beat once a flash section began, which is the "erratic,
                # flashing all over the place" the verticals never show.
                extra = [t for t in ens_targets
                         if "horizontal" not in t["name"].lower()]
                # C9s (and any other long linear VOICE carrier) get to flash
                # too. "The C9s only do movement but they could do the flashing
                # and pulsing too sometimes. They add a strong rhythmic effect
                # because they cross the whole house." They were structurally
                # excluded: carrying an instrument voice puts a prop in
                # movement_props, and the anchor pool filters that out -- so
                # unlike the outlines, they could never alternate.
                extra += [t for t in cast
                          if t["cat"] in ("single_line", "icicle")
                          and t["n"] >= 2
                          and t["name"] in prop_voice
                          and t not in extra]
                if extra:
                    pool = anchor_rank + sorted(extra, key=lambda t: t["name"])
            # SEVERAL anchors at once, all held for the whole section. "I want
            # to have multiple places where we're showing the main beat and not
            # rotate them too frequently so it's not disorienting, you want
            # viewers to notice what's pulsing to the beat and be able to see
            # it for a few bars."
            if si in leap_sections:
                pool = [t for t in pool if t["name"] != "Arches"] or pool
            n_anch = 2 + int(round(section_intensity.get(si, 0.5)))
            base = (si * 2) % len(pool)
            picks = [pool[(base + k) % len(pool)]
                     for k in range(min(n_anch, len(pool)))]
            # RESERVE ONE SLOT for a group that splits left/right, so the
            # call-and-response actually happens regularly instead of only
            # when a splitting group randomly lands in the 2-3 anchors. On the
            # first build it fired so rarely that Arches -- a 2/2 split -- got
            # 23 individual effects across a whole song.
            splitters = [t for t in pool if split_lr(t)]
            if splitters and not any(split_lr(t) for t in picks):
                picks[-1] = splitters[si % len(splitters)]
            section_anchor[si] = picks

    n_cast = len(cast)
    role_counts = Counter()
    # --- DROP-AND-FADE CYCLES ---------------------------------------------
    # "There is a single kick drum for each drop and everything between that
    # and the next single kick drum is the slow fade... It all needs to be
    # synced to the kick drum. I think there's a total of 4 or 5 drop and fade
    # cycles like that in the entire first minute."
    #
    # Measured on Gooey: the kick fires at 4.82, 15.85, 26.90, 37.92, 48.97
    # and 59.99 -- five cycles, 11.03s apart to within 20ms, with the last one
    # the main drop where the full track starts. Against that, the section
    # detector put THIRTY-TWO boundaries in the same 62 seconds. Driving big
    # canvases off sections in this passage is what produced "the matrices
    # come on sometimes on the kick but sometimes on other random times, turn
    # off at random times, and switch to different patterns at random times."
    #
    # An isolated kick -- one with several seconds of silence before it -- IS
    # the structure here, so it is the only grid worth using. Detected rather
    # than hardcoded, so any song with this shape gets it.
    def _drop_cycles() -> list[tuple[float, float]]:
        kv = voices.get("kick")
        if kv is None or len(kv) < 3:
            return []
        ons = sorted(kv.onsets)
        iso = [t for i, t in enumerate(ons)
               if (i == 0 or t - ons[i - 1] >= 6.0)]
        gaps = [b - a for a, b in zip(iso, iso[1:]) if 5.0 <= b - a <= 20.0]
        if len(gaps) < 2:
            return []
        period = sorted(gaps)[len(gaps) // 2]
        # Walk the grid forward from the first isolated hit rather than
        # chaining detected hits directly. The kick detector MISSES some of
        # these: on Gooey it finds 4.78, 26.90, 37.87 and 49.02 but not the
        # one at 15.85, leaving a 22-second hole that looks like two cycles
        # fused into one. The figure is far more regular than the detector --
        # measured period 11.03s, consistent to 20ms -- so predict the next
        # start and SNAP it to a real kick when one is there, which keeps
        # every boundary on an actual drum hit while surviving a missed one.
        out: list[tuple[float, float]] = []
        t = iso[0]
        while True:
            nxt = t + period
            near = min(ons, key=lambda o: abs(o - nxt), default=None)
            if near is not None and abs(near - nxt) <= 0.5:
                nxt = near
            if nxt > (ons[-1] if ons else 0) + period:
                break
            # The passage ends where the kick becomes continuous again -- that
            # last isolated hit IS the main drop, and past it the song is no
            # longer built of drop-and-fade cycles.
            dense = sum(1 for o in ons if t < o < nxt) >= 4
            if dense:
                break
            out.append((t, nxt))
            t = nxt
            if len(out) > 24:
                break
        # A DROP AND FADE HAS TO ACTUALLY FADE. Evenly spaced isolated kicks
        # are necessary but not sufficient: on a sparse, low-percussion track
        # the kick is naturally that thin all the way through, so this claimed
        # the entire back half of This Version of You (76.7-173.5s of a 172s
        # song) and the clearing pass then stripped 10,652 effects out of it,
        # leaving the song at 5,842 placements where its peers run 26-45k.
        #
        # What separates the two is measurable. Gooey's cycles fall 1.48-2.09x
        # from the hit to the tail; This Version of You's middle cycles sit at
        # 0.93-1.24x -- flat, because that passage is sustained rather than
        # decaying. Require a real decay, and require at least two consecutive
        # cycles to show it, so a single fade-out at the end of a song does
        # not turn the whole song into cycles.
        rms = np.asarray(analysis.spectrum.rms_energy)
        rt = np.asarray(analysis.spectrum.rms_times)

        def _decays(c0: float, c1: float) -> bool:
            head = rms[(rt >= c0) & (rt < c0 + 1.2)]
            tail = rms[(rt >= c1 - (c1 - c0) * 0.25) & (rt < c1)]
            if not len(head) or not len(tail):
                return False
            return float(head.mean() / max(tail.mean(), 1e-9)) >= 1.40

        runs, cur = [], []
        for c in out:
            if _decays(*c):
                cur.append(c)
            else:
                if len(cur) >= 2:
                    runs.extend(cur)
                cur = []
        if len(cur) >= 2:
            runs.extend(cur)
        return runs

    drop_cycles = _drop_cycles()
    if drop_cycles:
        print(f"Drop/fade cycles: {len(drop_cycles)} "
              f"({', '.join(f'{a:.1f}' for a, _ in drop_cycles)}s"
              f" -> {drop_cycles[-1][1]:.1f}s)")

    # Drop/fade cycle times, needed by change_marks() below -- computed here
    # rather than at the staging pass further down, which ran too late to
    # influence which shader a canvas holds.
    big_hits = detect_big_hits(
        analysis,
        kick_times=(voices["kick"].onsets if "kick" in voices else None),
        downbeats=downbeats)

    prev_pal_name = None
    # Per-song median section energy -- the reference point for "a lower
    # energy part". Absolute thresholds were tried for density earlier in this
    # project and did not transfer between songs, because each track sits at
    # its own overall level.
    _energies = sorted(sec.energy_level for sec in analysis.sections)
    median_energy = (_energies[len(_energies) // 2] if _energies else 0.0)
    # The QUIETEST sections go greyscale -- about one in five, at least one in
    # every song, never two in a row, and never the climax. Picking them up
    # front guarantees the look actually appears and puts it where Krists
    # asked for it ("maybe use it especially for lower energy parts").
    mono_sections: set[int] = set()
    _quiet = sorted(range(len(analysis.sections)),
                    key=lambda i: analysis.sections[i].energy_level)
    _want = max(1, round(len(analysis.sections) * 0.20))
    for _i in _quiet:
        if len(mono_sections) >= _want:
            break
        if _i in climax_sections:
            continue
        if (_i - 1) in mono_sections or (_i + 1) in mono_sections:
            continue
        mono_sections.add(_i)

    n_shader_pulse = 0
    for si, section in enumerate(analysis.sections):
        # The section's beat anchor keeps its ambient fill suppressed so it has
        # darkness to flash out of -- see section_anchor above.
        anchor_names_here = {t["name"] for t in section_anchor.get(si, [])}
        anti_names_here = {t["name"] for t in section_anchor.get(si, [])
                           if split_lr(t)}
        e = section.energy_level
        inst = section_meta[si]["instance"]
        escalation = min(inst - 1, 3)  # 0,1,2,3 -- later repeats get bolder

        # Absolute target-count ranges, matched directly to what was measured
        # across four human sequences (mean simultaneous elements: 40-112).
        # Deliberately NOT a fraction of n_cast -- once submodel groups
        # (window outlines, snowflake parts) are in the pool, n_cast can be
        # 2x+ the human target-vocabulary size, and a fixed fraction of that
        # inflates simultaneous activity right past the human range even
        # though the fraction number looks the same. Absolute counts stay
        # correct regardless of how big the cast pool grows.
        # The pre-kick starvation rule only applies to the RUN-UP, not to
        # everything before the kick. It exists so the kick's entrance reads as
        # a jump, and for that it only needs the bar or two immediately before.
        # Unbounded, it starves any song whose kick arrives late: measured,
        # World Is Mine's first kick is at 46.2s and Aperture's at 30.2s, so
        # 46 and 30 seconds respectively were pinned to 8 active targets --
        # 16% of one song rendering near-black over music that is playing.
        # (ENIGMA got away with it only because its intro is genuinely silent.)
        PRE_KICK_LEAD_S = 12.0
        pre_kick = (first_kick_time is not None
                    and section.end_time <= first_kick_time
                    and section.end_time >= first_kick_time - PRE_KICK_LEAD_S)
        if pre_kick:
            # Deliberately below the normal minimum tier -- see the
            # `first_kick_time` comment above. A few ambient touches, not a
            # full "intro" treatment, so the kick's entrance actually reads
            # as a jump rather than a continuation.
            n_active_target, n_feature = 8, 1
        else:
            # SECTION LIT BUDGET. Activity is apportioned from the section's
            # MEASURED loudness rank within this song, not from the detector's
            # `energy_level` label. Measured on ENIGMA v2, the label-driven
            # version produced inverted dynamics: the bass breakdown rendered
            # at 36.8% lit -- brighter than every chorus -- while the actual
            # climax sat at 22.7%, dimmer than the verse. "Lighting is all
            # about emphasizing the right moments in the song so the big
            # moments look big," and that was the opposite.
            #
            # Rank-based rather than absolute, so it is self-normalising: the
            # song's own loudest section always gets the top budget and its
            # quietest always gets the bottom, whatever the absolute levels.
            # This makes "the climax is the brightest thing in the song" true
            # by construction instead of by luck.
            _lo, _hi = 14, 58
            n_active_target = int(round(_lo + (_hi - _lo) * section_intensity[si]))
            n_feature = 1 + int(round(2 * section_intensity[si]))
        n_feature = min(n_feature + (1 if escalation >= 2 else 0), 4)

        # --- instrumentation headroom -------------------------------------
        # "Songs often have other small moments that get quieter and suddenly
        # louder even though it's not a big buildup or drop... it's like just
        # saving some lights when some instruments drop out so we have extra
        # lights to use when new instruments come in."
        #
        # Energy alone doesn't capture this: a section can stay loud while
        # half the arrangement drops out (a breakdown that's still driving),
        # and RMS barely moves. Counting how many INSTRUMENT VOICES are
        # actually playing does capture it directly -- and the voice table
        # already exists for the legibility work, so this is free signal
        # rather than another detector to tune.
        #
        # Deliberately gentle (scales activity between 0.65x and 1.0x, never
        # below): "This shouldn't be extreme like the big drops." The value
        # is in the CONTRAST it buys for the moment the instruments return,
        # not in the dip itself being noticeable on its own.
        if voices:
            playing = sum(1 for v in voices.values()
                          if v.density_in(section.start_time, section.end_time)
                          >= VOICE_MIN_DENSITY)
            instr_ratio = playing / max(1, len(voices))
            # Voice COUNT alone under-reacts. Direct correction: "the number
            # of voices isn't quite enough of a measurement since often the
            # voices are all there but more mellow for a second and then the
            # bass comes in really hard and that should be a bigger visual
            # moment." Exactly right -- an arrangement can keep every
            # instrument present while dropping a long way in intensity, and
            # counting voices sees no change at all.
            #
            # `loudness_ratio` is this section's own loudness as a fraction of
            # the song's loudest section, so it measures intensity rather than
            # presence. Combined 50/50 with the voice ratio: presence still
            # matters (a stripped-back section should dim even if what's left
            # is loud), but intensity now moves the needle on its own.
            loud_ratio = section_loudness.get(si, 1.0)
            # Vocal presence as a third energy input. A section where the
            # singer is actually singing is subjectively "fuller" than an
            # instrumental passage at the same loudness and instrument count,
            # and this is the agreed use for the lyric data -- timing and
            # energy, never word identity. Weighted lightly (15%) so it
            # nudges rather than dominates.
            voc = vocal_fraction(section.start_time, section.end_time)
            combined = (0.45 * min(1.0, instr_ratio / 0.8)
                        + 0.40 * loud_ratio
                        + 0.15 * voc)
            # Widened from 0.65-1.0 to 0.5-1.0 -- "the drops aren't quite
            # graduated enough" and "the lights aren't dynamic enough with
            # the changes in energy in the song" both point the same way.
            # Widened from 0.5-1.0 to 0.35-1.0. Direct feedback: "the dynamic
            # detection that would get a bit dimmer before a new phrase or
            # drop or fill isn't working, it's too static and always lit up."
            # The inputs were varying correctly but the output range was too
            # narrow to see -- a section at half the song's intensity still
            # got 75% of full activity.
            headroom = 0.35 + 0.65 * combined
            # A drum fill almost always sits at the end of a phrase, just
            # before a new section arrives. Pulling activity down across the
            # fill buys contrast for the downbeat that follows it.
            #
            # This alone fired far too rarely to be the "dip before a change"
            # the show needs: fills are genuinely scarce (1-4 per song, and
            # ZERO on Gooey), so most section changes had no run-up at all.
            # Every section now dips as it approaches its own end, whether or
            # not a fill happens to be there -- a section boundary is itself
            # the structural moment worth setting up.
            if any(section.end_time - 1.5 <= f0 <= section.end_time
                   for f0, _f1 in drum_fills):
                headroom *= 0.7
            section_meta[si]["headroom"] = headroom
            n_active_target = int(round(n_active_target * headroom))
            section_meta[si]["instr_playing"] = playing

        # DURING A DROP/FADE CYCLE, THE DISPLAY THINS OUT. "Everything between
        # that and the next single kick drum is the slow fade" -- everything,
        # not just the canvases carrying the shader. Measured with the cycle
        # holds and their decay envelope in place but the rest of the show
        # running normally, the kicks at 15.8/26.9/37.9 still lifted the
        # display only 1.2-1.5x, because ~450 other targets held the floor up
        # at 40-60 while the fade decayed underneath them. The two kicks that
        # DID read (6.2x and 6.5x) are the ones with genuinely quiet approaches.
        #
        # So the cast is cut back hard inside a cycle: the hit lands on a
        # near-empty display and the fade has somewhere to fall to.
        if any(c[0] <= section.start_time < c[1] for c in drop_cycles):
            n_active_target = max(3, int(round(n_active_target * 0.20)))
        n_active = min(n_cast, max(2, n_active_target))

        # Rotate which targets are active so different props lead over time.
        # Prominent large structures (Tunnel's 80 arches, Arches, Orbs...) get
        # a near-guaranteed slot rather than the same uniform rotation odds
        # as a 2-member submodel group -- otherwise the single biggest,
        # most visually significant feature in the whole cast can go quiet
        # for long stretches purely by chance, which read as "underused"
        # despite being structurally present and correctly configured.
        # `t["n"] >= 15` alone missed the biggest offenders: hero MODELS
        # (Mega Tree, Garage Roof, Sun Room Window Matrix, Tunnel Matrix
        # Real) are single-model cast entries with n==1 -- structurally
        # identical rotation odds to a 2-member submodel group. Measured
        # directly in a generated sequence: Sun Room Window Matrix and
        # Garage Roof both went dark for 26-28 CONTINUOUS SECONDS in one
        # section purely from losing the rotation lottery, matching "the
        # matrices are all blank" precisely. `large` is exactly the "biggest,
        # most visually significant" criterion this comment already claimed
        # to guarantee -- it just wasn't wired in.
        rng = random.Random(f"{SONG_NAME}|{si}|{section.label}")
        prominent = [i for i, t in enumerate(cast) if t["n"] >= 15 or t.get("large")]
        rest = [i for i in range(n_cast) if i not in prominent]
        rng.shuffle(rest)
        offset = (si * 7 + escalation * 3) % max(1, len(rest))
        active_idx = list(prominent) if e >= 0.30 else []
        # VOICE CARRIERS ARE NEVER ROTATED OUT while their instrument is
        # actually playing. The whole legibility system rests on a prop
        # representing one instrument consistently -- letting a generic
        # activity budget silence it defeats the point, and it bites hardest
        # in exactly the sparse sections where that instrument is the only
        # thing to look at. Measured on ENIGMA: Orbs (the bass carrier) drew
        # 3 effects and 0.8 effect-seconds across §7's twenty seconds and the
        # same across §8's twenty-four, because a quiet section's budget is
        # ~14 targets and Orbs lost the lottery. "There's also a repetitive
        # bass rhythm going on, is it possible to show that too?" -- it was
        # assigned, reinforced, and then rotated out.
        for i, t in enumerate(cast):
            vn = prop_voice.get(t["name"])
            if not vn or i in active_idx:
                continue
            vv = voices.get(vn)
            if vv is None:
                continue
            if (vv.density_in(section.start_time, section.end_time) >= VOICE_MIN_DENSITY
                    or vv.pattern_score_in(section.start_time, section.end_time,
                                           beats) >= VOICE_MIN_PATTERN):
                active_idx.append(i)
        remaining_budget = max(0, n_active - len(active_idx))
        active_idx += [rest[(offset + k) % len(rest)]
                       for k in range(min(remaining_budget, len(rest)))
                       if rest[(offset + k) % len(rest)] not in active_idx]

        sec_start, sec_end = section.start_time_ms, section.end_time_ms
        sec_downbeats = [d for d in downbeats if section.start_time <= d < section.end_time]
        sec_beats = [b for b in beats if section.start_time <= b < section.end_time]
        # Advance the palette EVERY section, and never repeat the previous
        # section's palette back to back. The old `si // 2` deliberately held
        # a palette for two sections at a time, which combined with the
        # escalation offset to give chorus #5 and #6 the same palette --
        # ~50 continuous seconds of one colour scheme on this song. On a
        # red-leading palette that was the entire "props are red for a while"
        # complaint; even on a good palette it's monotonous. Cycling is now
        # through `song_palette_cycle` (this song's fixed subset), not the
        # full PAL_CYCLE, for cohesion -- see PALETTE_FAMILY_SIZE above.
        if si in climax_sections:
            # The song's real climax overrides the rotation entirely with a
            # dedicated high-contrast palette -- direct feedback: "identify
            # the big climaxes... where we can introduce more contrast."
            pal_name = cycle_name = "climax"
        else:
            pal_idx = (si + escalation) % len(song_palette_cycle)
            if song_palette_cycle[pal_idx] == prev_pal_name:
                pal_idx = (pal_idx + 1) % len(song_palette_cycle)
            pal_name = cycle_name = song_palette_cycle[pal_idx]
            # A quiet section may go MONO instead -- greyscale kept as one of
            # the colour choices rather than lost to the colour conversion.
            # CHOSEN, not rolled: a per-section coin flip gave Fara zero mono
            # sections and most songs one or two, which is why Krists saw it
            # "in one part in Gooey and I think that's about it." A song that
            # never shows a variation does not have that variation.
            if si in mono_sections:
                pal_name = sorted(MONO_PALETTES)[si % len(MONO_PALETTES)]
        prev_pal_name = pal_name
        palette = PALETTES[pal_name]
        section_meta[si]["palette"] = palette
        if pal_name == "climax":
            accent_pal = PALETTES["gold"]
        else:
            # Accent follows the ROTATION, not the section's actual palette:
            # a mono section is not in the song's colour cycle, so looking it
            # up there is a ValueError. Its accent stays the colour the
            # section would otherwise have had, which keeps the mono passage
            # tied to the songs's palette family instead of floating free.
            accent_idx = (song_palette_cycle.index(cycle_name) + 1) % len(song_palette_cycle)
            accent_pal = PALETTES[song_palette_cycle[accent_idx]]
        # One shader/pattern choice for every large canvas active THIS
        # section, instead of each rolling its own -- direct feedback:
        # "we could share shaders across matrices... more cohesive
        # overall." Re-rolled every section so the look still evolves
        # over the song; held steady within a section so simultaneously-
        # visible hero canvases read as one coordinated look, not four
        # independent animations.
        # SECTION-INSTANCE LOOK. "There are places where I notice when we
        # switch from one section to another, like from one verse to the next,
        # lots of stuff stays the same, but it feels fresh if a new verse has
        # something different about the look."
        #
        # Deliberately NOT randomness -- the second verse should differ from
        # the first in a way a viewer can feel as a change, so it keys off the
        # section's INSTANCE number (verse 1 vs verse 2 vs verse 3) and flips
        # two big, visible things at once:
        #   * what the hero canvases are running -- shader-led or built-in
        #     pattern-led. These look genuinely different at a glance.
        #   * which way everything travels. A whole facade that was chasing
        #     left is now chasing right.
        # Held for the whole section, so it reads as "this verse is different"
        # rather than as churn.
        look = inst % 3
        if look == 1:
            _pool = [k for k in SAFE_LARGE_WEIGHTED if not k.startswith("shader_")]
        else:
            _pool = [k for k in SAFE_LARGE_WEIGHTED if k.startswith("shader_")]
        # Weight the shared choice toward palette-driven shaders, so when the
        # hero canvases do run a shader they all sit in the section's colour
        # world instead of each importing its own.
        # The shared choice must be PALETTE-DRIVEN, not merely preferred.
        #
        # Cohesion changed the stakes here. Now that every hero canvas runs the
        # same shader, that one shader covers the entire facade at once -- so a
        # stock shader, which generates its own colours and ignores the
        # palette, paints all seven canvases in whatever hue it likes.
        # Measured: SearchingForSomething across all seven took 12 to 12 to a
        # 39.8% red peak. The same choice was harmless when canvases each ran
        # something different, because no single shader dominated.
        # THE SHARED CHOICE MUST BE AN EVEN TEXTURE, never a radial one.
        # "You're often using the same shader across all matrices which works
        # when it's a more abstract shape or texture but when it's a circular
        # or radial pattern it looks a bit repetitive like showing the same
        # movie on 10 screens." A centred pattern repeated across the house
        # reads as ten copies of one picture; a texture reads as one material
        # covering everything. Radial shaders are still available -- they just
        # arrive via a single canvas's own roll rather than the shared slot.
        _pd = [k for k in _pool if k in PALETTE_DRIVEN_SHADERS
               and not is_centred(k)]
        if _pd:
            shared_large_key = rng.choice(_pd)
        else:
            _npd = [k for k in _pool if not k.startswith("shader_")]
            shared_large_key = rng.choice(_npd or _pool or SAFE_LARGE_WEIGHTED)
        look_reverse = look >= 1

        # Stable seat number per large canvas this section, so each one gets a
        # consistent rotation of the section palette instead of a random one.
        _large_seats = {t["name"]: i for i, t in enumerate(
            sorted((cast[c] for c in active_idx if cast[c].get("large")),
                   key=lambda t: t["name"]))}
        for rank, ci in enumerate(active_idx):
            target = cast[ci]
            # Stable per-canvas seat straight from the name, so a lookup miss
            # can never silently collapse every canvas onto seat 0.
            large_seat = _large_seats.get(
                target["name"], abs(hash(target["name"])) % 3)

            # Weighted so MOTION carries the show (human sequences are
            # chase-dominated); pulse is seasoning, not the main dish.
            n_motion = max(n_feature, int(n_active * 0.38))
            n_pulse = max(1, int(n_active * 0.18))
            n_texture = max(1, int(n_active * 0.24))
            if rank < n_motion:
                role = ROLE_FEATURE
            elif rank < n_motion + n_pulse:
                role = ROLE_PULSE
            elif rank < n_motion + n_pulse + n_texture:
                role = ROLE_TEXTURE
            else:
                role = ROLE_OUTLINE
            # A large canvas (or other `wants_motion` target -- see its
            # docstring) doing a flat full-brightness flash is jarring --
            # direct feedback: "flashing such a large prop is hard for the
            # eyes," and separately "candy canes... big enough to do
            # movements... not just flash and color fade." These targets
            # never get the flash-style PULSE role; redirect to FEATURE
            # (real motion/pattern) instead. Arches specifically: PULSE role
            # placed on "Arches" (a real group, not synthetic) flashes all 4
            # arches simultaneously, completely bypassing the dedicated
            # beat-paced cascade below -- direct feedback confirmed this.
            if role == ROLE_PULSE and wants_motion(target):
                role = ROLE_FEATURE
            # A prop that carries an instrument voice ALWAYS performs that
            # instrument -- it never gets handed a generic feature/texture
            # role that would overwrite its identity with unrelated motion.
            #
            # Measured why this is necessary rather than nice-to-have: with
            # voice assignment in place but the prop still eligible for every
            # other pass, only 16-37% of the effects landing on a voice prop
            # actually coincided with its own instrument's onsets. The other
            # 60-80% came from feature motion, whole-display sweeps, arpeggio
            # chases and the like. A prop following its instrument a third of
            # the time is not legible as that instrument at all, which is the
            # entire point of the exercise. See the matching exclusions from
            # the generic pools in PASS 2.
            if target["name"] in prop_voice:
                role = ROLE_PULSE
            role_counts[role] += 1

            if role == ROLE_FEATURE and target["name"] in anti_names_here:
                # A group answering itself side to side cannot also run a
                # group-wide chase: measured, the Arches GROUP carried 96
                # effects lighting all four members while the per-member
                # antiphonal placed 23, so left and right measured identical
                # and the call-and-response was invisible.
                pass
            elif role == ROLE_FEATURE and target["name"] in movement_props:
                # ONE chase per prop, ever. The unison pass below already gives
                # every ensemble prop a beat-gridded chase; letting the feature
                # role add a second, independently-timed one on another layer
                # is what produced "the motion in the verticals and horizontals
                # is too messy, there are always multiple chases and they seem
                # to change direction at odd times with overlapping chases."
                # Verified in the .xsq: Corner Verticals carried twelve chases
                # on layer 0 and a second overlapping chase on layer 1 across
                # the same twelve seconds.
                pass
            elif role == ROLE_FEATURE:
                # Was `cat == "arch"` only -- direct feedback: "we shouldn't
                # always do the kick drum on the arches, we can pick other
                # props too." Broadened to any category sharing the same
                # shape this cascade actually needs (a small sequence of
                # individually-addressable members with a real physical
                # order to travel across): arches AND arch-SIZED single_line
                # groups (C9s at 8, small snowflake outline groups at 4-8).
                # Capped at 10 members -- some `single_line` groups (the GE
                # decorated-prop spoke/petal families, via OUTLINE_NAME_HINTS)
                # run 20-50 members; the cascade's "one member advances per
                # kick hit, brief hold" mechanic was designed around a tight
                # 4-arch cycle and reads as thin/scattered stretched across
                # that many individual targets, not more rhythmic. Those
                # larger groups keep their native single_line chase/wave
                # treatment (already confirmed-good) instead. This doesn't
                # force rotation explicitly -- it rides on the cast's
                # existing per-section target rotation, so whichever eligible
                # group happens to draw FEATURE role that section gets the
                # cascade, arches included but no longer exclusively.
                if target["cat"] in ("arch", "single_line") and 3 <= target["n"] <= 10:
                    # Arches read as random/uncentered when driven as a single
                    # group-level Chase (buffer-style ambiguity we can't
                    # visually verify). Cross-model cascade across the
                    # INDIVIDUAL members instead -- fully explicit timing.
                    #
                    # Timed on the REAL kick pattern (kick_times_in), not the
                    # generic beat grid -- direct correction: "I liked the
                    # rhythm on the arches but it wasn't aligned with the kick
                    # drum." The generic librosa beat grid marks the felt
                    # PULSE of the song, which on a syncopated track (R&B,
                    # hip-hop) is not the same thing as where the kick drum
                    # itself actually hits. See detect_kick_pattern().
                    ordered = sorted(target["members"],
                                      key=lambda m: positions.get(m, {}).get("x", 0))
                    reversed_order = ordered[::-1]
                    # Alternate what the arches follow, section to section.
                    # "The voice on the arches... it's often snare drum or
                    # drum rolls which look good, but they can also be great
                    # for kick drum or pulsing beats or bouncing left right
                    # beats." Rotating between the kick and the snare (and
                    # between a travelling cascade and a bounce) keeps the
                    # four arches from settling into one fixed behaviour for
                    # the whole song, while both sources stay percussive so
                    # it never drifts onto something unreadable.
                    arch_mode = si % 3
                    if arch_mode == 1 and "snare" in voices:
                        src = voices["snare"].in_window(section.start_time,
                                                        section.end_time)
                    else:
                        src = kick_times_in(section.start_time, section.end_time)
                    bounce = (arch_mode == 2)
                    hold = max(260, min(int(beat_ms * 0.9), 550))
                    n_m = len(ordered)
                    for bi, t in enumerate(src):
                        if bounce:
                            # Ping-pong across the arches (1-2-3-4-3-2-...)
                            # instead of always restarting at one end.
                            period = max(1, 2 * (n_m - 1))
                            ph = bi % period
                            idx = ph if ph < n_m else period - ph
                            m = ordered[idx]
                            if in_ambient(t):
                                continue
                            t_ms = int(t * 1000)
                            add(m, L_MOTION, "on_hit", t_ms, t_ms + hold, palette)
                            continue
                        going_forward = (bi // n_m) % 2 == 0
                        m = (ordered if going_forward else reversed_order)[bi % n_m]
                        # Not in an ambient passage. This walks a group's
                        # members on EVERY BEAT -- measured on This Version of
                        # You, the arches ran 1,2,3,4,4,3,2,1 every 0.45s
                        # straight through a drumless piano intro. It is the
                        # last of the beat-driven passes that had no idea the
                        # passage had no beat.
                        if in_ambient(t):
                            continue
                        t_ms = int(t * 1000)
                        # on_hit (sharp attack), not on_swell (soft 150ms
                        # fade-in) -- a slow fade-in undercuts the sense of
                        # landing exactly on the beat, which is likely why
                        # this read as "not rhythmic" despite being timed to
                        # the actual beat grid.
                        add(m, L_MOTION, "on_hit", t_ms, t_ms + hold, palette)
                else:
                    key = feature_effect_for(target, rng, escalation, shared_large_key)
                    # Effect changes land on BAR boundaries, and are held for
                    # several bars rather than swapping every single one.
                    #
                    # Direct feedback on Gooey's intro: "the shaders keep
                    # changing and the changes are not on rhythm so it looks
                    # random, I would change them on rhythm like every couple
                    # of bars." The change points were already on downbeats
                    # (verified: detected downbeats sit at 3.96 beats apart,
                    # so the bar grid itself is accurate) -- the problem was
                    # the RATE. A shader restarted every bar (2.76s in Gooey)
                    # never gets far enough into its own animation to read as
                    # a developing pattern, so consecutive bars just look like
                    # unrelated churn. Large canvases hold 4 bars, everything
                    # else 2, so a pattern has room to actually go somewhere.
                    bars_per_change = 4 if target.get("large") else 2
                    # Prefer bar boundaries where the HARMONY moves -- see
                    # change_marks(). Falls back to the fixed cadence on songs
                    # without enough chord movement to drive it.
                    #
                    # The span placement itself lives in hold_spans(), shared
                    # with the TEXTURE/OUTLINE branches so that every large
                    # canvas lands on the same boundaries and carries the same
                    # beat pulse no matter which role it drew this section.
                    marks = change_marks(section, bars_per_change)
                    if len(marks) >= 2:
                        n_shader_pulse += hold_spans(target, L_MOTION, key,
                                                     section, palette,
                                                     large_seat, e)
                    elif not (target.get("large") and in_ambient(section.start_time)):
                        # The ambient deferral lives inside hold_spans, so this
                        # fallback -- taken when a section has too few bar
                        # marks -- was placing a full-section shader on the
                        # canvases regardless. That is what cut short the
                        # 1:17 hold on This Version of You: the ambient pass
                        # placed Herringbone from 77.10 to 114.78 and this put
                        # Aurora Curtain over it at 78.47, leaving the good
                        # shader 1.33s of screen time.
                        add_target(target, L_MOTION, key, sec_start, sec_end, palette)

                    # LAYER a cone-appropriate accent ON TOP of a shader.
                    # "Maybe we can layer things on top of the shaders we're
                    # using since they do look good" + "you can definitely try
                    # layering shaders now." Only for cone props running an
                    # actual shader: the shader keeps providing the texture,
                    # and a slow spiral/pinwheel over it adds the wrapping
                    # motion a cone wants, which a flat shader alone doesn't
                    # give. On its own layer so neither replaces the other.
                    if (target["name"] in CONE_SHAPED
                            and key.startswith("shader_")
                            and len(marks) >= 2):
                        overlay = "spirals" if (si % 2 == 0) else "pinwheel"
                        add_target(target, L_PULSE, overlay,
                                   int(marks[0] * 1000), int(marks[-1] * 1000),
                                   accent_pal)

            elif role == ROLE_PULSE:
                # A prop with an assigned instrument voice performs THAT
                # instrument, all song, and nothing else. This is the core of
                # the legibility fix -- see VOICE_PROP_PREFS and
                # instrument_voices.py. Everything else that used to drive
                # this role (the whole undifferentiated drums stem) is gone:
                # it produced accurate-but-unreadable activity, because "the
                # drums" is not an instrument a viewer can follow.
                #
                # A prop with NO voice assigned gets no pulse at all now
                # (redirected to texture below). Fewer props flashing, each
                # one clearly attributable, was the explicit trade asked for:
                # "I don't want to lose all the rhythm but need to tighten it
                # up so it's clear what instrument a given prop is
                # representing."
                vname = prop_voice.get(target["name"])
                voice = voices.get(vname) if vname else None
                # A voice performs if it is DENSE enough OR if its rhythm
                # REPEATS enough. Density alone silenced exactly the parts
                # Krists asked about: "why can't we extract the rhythm from the
                # synths in the quieter parts? There are clear repetitive
                # rhythms in both... any repetitive rhythm on any voice is
                # worth showing." Measured on ENIGMA, the chord voice scores
                # 0.43 repetition through the bass breakdown and 0.26 through
                # the near-beatless section -- real, legible figures that the
                # density gate was throwing away. See Voice.pattern_score_in.
                _dense = voice is not None and voice.density_in(
                    section.start_time, section.end_time) >= VOICE_MIN_DENSITY
                _patterned = voice is not None and voice.pattern_score_in(
                    section.start_time, section.end_time, beats) >= VOICE_MIN_PATTERN
                if voice is None or not (_dense or _patterned):
                    # This prop isn't a voice carrier, or its instrument isn't
                    # really playing in this section -- give it quiet presence
                    # instead of inventing a rhythm for it.
                    #
                    # ...unless it carries movement from the unison ensemble,
                    # in which case a full-section twinkle underneath would
                    # light every node and hide the chase. This was the SECOND
                    # source of that fill: suppressing it in ROLE_TEXTURE and
                    # ROLE_OUTLINE alone left the prop still fully lit,
                    # because ensemble props draw the PULSE role and fall
                    # through to here. Found by re-probing after the first
                    # fix produced byte-identical output.
                    if (target["name"] not in movement_props
                            and target["name"] not in anchor_names_here):
                        add_target(target, L_BASE, "twinkle_soft",
                                   sec_start, sec_end, palette)
                    continue

                hits = voice.in_window(section.start_time, section.end_time)
                if target.get("slow"):
                    # Wash lights (floods) thin to at most one hit every two
                    # beats regardless of what their instrument is doing --
                    # see SLOW_PROPS. Their job is drama and lift, not rhythm.
                    min_gap = SLOW_PROP_MIN_GAP_BEATS * beat_ms / 1000.0
                    thinned = []
                    for t in hits:
                        if not thinned or t - thinned[-1] >= min_gap:
                            thinned.append(t)
                    hits = thinned
                dur = min(int(beat_ms * 0.55), 380)
                _prefs = VOICE_PROP_PREFS[vname]
                if _prefs.get("keys"):
                    flash_key = random.Random(
                        f"{SONG_NAME}|voicekey|{vname}").choice(_prefs["keys"])
                else:
                    flash_key = _prefs["key"]
                if target["n"] < 3 and flash_key in ("sweep_right", "sweep_left"):
                    flash_key = "on_hit"  # As Pixel sweep needs several members

                # Carry the rhythm with MOVEMENT where the prop's shape and
                # the voice's pace allow it, not with a flash. "Movement can
                # also be rhythmic. Like when the mini trees flash to the kick
                # drum, they could also have an up down movement to the
                # rhythm." Each hit's effect SPANS to the next hit, so the
                # prop is continuously in motion in tempo rather than blinking
                # and going dark between hits; alternating direction per hit
                # makes it rock rather than restart.
                motion = VOICE_MOTION_BY_CAT.get(target["cat"])
                if target["cat"] in ("single_line", "icicle"):
                    motion = line_motion_for(target)
                    if motion and target["name"] in DEPTH_ORIENTED:
                        # See the ensemble pass: depth-oriented strands read
                        # reversed from the street.
                        motion = (REVERSE.get(motion[0], motion[0]),
                                  REVERSE.get(motion[1], motion[1]))
                # A single-member LINEAR target still moves. `n >= 2` exists
                # for cross-model chases (light stepping from prop to prop),
                # but a lone strand chases along its OWN length just fine --
                # and every `*/Horizontals` and `*/Verticals` matrix-outline
                # submodel is n=1. Measured on Aperture: those submodels had
                # 52-71 `On` flashes and ZERO movement effects, which is
                # exactly "the horizontals are all erratic and just flashing
                # all over the place, no smooth movement at all."
                can_move = motion is not None and (
                    target["n"] >= 2 or target["cat"] in ("single_line", "icicle"))
                # A long run needs its travel stretched over two hits instead
                # of one -- "the c9 chases are too fast too." Halving the rate
                # doubles the time each dot has to cross its share of the line.
                long_run = target.get("spread_x", 0) >= 40 or "c9" in target["name"].lower()
                if long_run:
                    # A LONG GUTTER RUN NEEDS A SLOW SOURCE. "I don't like the
                    # movement in the c9s on the horizontal gutters, they seem
                    # to follow high frequency beats or something and I think
                    # they'd be better if they followed the drum or bass
                    # rhythms that are simpler so you could follow the
                    # movement." A dot crossing forty feet cannot be followed
                    # if it restarts on every sixteenth -- the eye needs the
                    # travel to complete. Thinned to at most one start per
                    # beat, whatever the assigned voice is doing underneath.
                    # Every TWO beats, not one. "The C9 gutter movement is
                    # still just too fast. The gutters are big and movement
                    # across the whole thing that fast just doesn't look good.
                    # It needs to be slower and just move less distance."
                    _min = beat_ms / 1000.0 * 1.9
                    _thin = []
                    for _t in hits:
                        if not _thin or _t - _thin[-1] >= _min:
                            _thin.append(_t)
                    hits = _thin

                flashes, bursts = split_flashes_and_bursts(hits)
                for i, t in enumerate(flashes):
                    t_ms = int(t * 1000)
                    nxt = flashes[i + 1] if i + 1 < len(flashes) else None
                    gap = (nxt - t) if nxt is not None else (beat_ms / 1000.0)
                    if target.get("slow"):
                        # Long soft swell rather than a hit -- a flood should
                        # bloom, not blink.
                        add_target(target, L_PULSE, "wash", t_ms,
                                   t_ms + int(min(gap, bar_s) * 1000), accent_pal)
                    elif can_move and gap >= VOICE_MOTION_MIN_GAP_S:
                        # A long run only starts a new travel every OTHER hit,
                        # so each chase has twice as long to cross -- the fix
                        # for "the c9 chases are too fast."
                        if long_run and i % 2 == 1:
                            continue
                        span = min(gap * (2 if long_run else 1), bar_s * 1.5)
                        # Beat-keyed direction, same reasoning as the ensemble
                        # above -- a voice prop should swing in phase with the
                        # rest of the house, not on its own hit count.
                        k = motion[beat_phase(t, every=2)]
                        # Movement on L_MOTION, leaving L_PULSE free for the
                        # beat accent below -- "I think we had some more
                        # pulsing/flashing of props on the main quarter note
                        # or half note beats before, but when we added more
                        # side to side and up down movement we lost some of
                        # the pulsing beats. I'd like both if we can!"
                        add_target(target, L_MOTION, k, t_ms,
                                   t_ms + int(span * 1000), accent_pal)
                    else:
                        # Too fast for movement to read -- a short hit at this
                        # rate is shimmer/texture, which is correct for hats.
                        add_target(target, L_PULSE, flash_key, t_ms,
                                   t_ms + dur, accent_pal)
                for b0, b1 in bursts:
                    b0_ms, b1_ms = int(b0 * 1000), int(b1 * 1000) + dur
                    move_key = motion[0] if can_move else \
                        feature_effect_for(target, rng, escalation, shared_large_key)
                    add_target(target, L_MOTION if can_move else L_PULSE,
                               move_key, b0_ms, b1_ms, accent_pal)

                # --- beat accent ON TOP of the movement ---------------------
                # "I'd like both if we can!" Movement carries the groove; this
                # adds back the punctuation that was lost when movement
                # replaced flashing. Deliberately only on STRONG beats (the
                # voice's hits that land on a quarter note) and only for the
                # percussive/rhythm-driver voices -- pulsing a melody line
                # would be exactly the "gets lost and seems random" problem.
                # Short and on its own layer so it reads as an accent over the
                # motion rather than interrupting it.
                if can_move and vname in ("kick", "snare", "bass"):
                    acc = min(int(beat_ms * 0.30), 200)
                    for t in flashes:
                        # nearest beat within a 16th -- i.e. on the quarter
                        # note grid, not on syncopated in-between hits
                        bi = int(np.searchsorted(_beats_arr, t))
                        bi = max(0, min(bi, len(_beats_arr) - 1))
                        if abs(float(_beats_arr[bi]) - t) > (beat_ms / 1000.0) * 0.25:
                            continue
                        t_ms2 = int(t * 1000)
                        add_target(target, L_PULSE, "on_pop",
                                   t_ms2, t_ms2 + acc, palette)

            elif role == ROLE_TEXTURE:
                if target["name"] in movement_props:
                    # This prop carries a CHASE on a higher layer. A chase is
                    # only visible as the contrast between lit and unlit
                    # nodes, so a full-coverage ambient fill underneath makes
                    # it invisible no matter how correct the movement is.
                    #
                    # Measured with the new per-prop probe, which is what
                    # finally caught this: on `Vertical Strip Front Left` the
                    # composited output was 100% lit with the lit-node
                    # centroid frozen at exactly 50.0% -- the whole strip on
                    # at once, brightening and dimming. The .xsq showed the
                    # chase present and correct on layer 1, with a 27-second
                    # `Twinkle` on layer 0 underneath it. Four earlier rounds
                    # of "fixing" the movement failed because the movement
                    # was never the problem.
                    continue
                if wants_motion(target):
                    # A large canvas (or other `wants_motion` target) holding
                    # a flat Shimmer/Color Wash for an entire section (can run
                    # 15-30+ seconds) reads as dead weight next to anything
                    # actually moving -- measured directly: "Color Wash" ran
                    # 17 continuous seconds on Garage Roof spanning the drop,
                    # and on a green-led section palette that IS the "garage
                    # roof is a pretty plain green, not impactful" complaint.
                    # Give it real (if unhurried) motion here too, same safe
                    # pool FEATURE uses, instead of a static fill -- and the
                    # SAME shared choice as every other large canvas this
                    # section, for cohesion.
                    key = feature_effect_for(target, rng, escalation, shared_large_key)
                    if look_reverse:
                        key = REVERSE.get(key, key)
                    # Same family, different lead colour per canvas. This is
                    # the path the hero canvases actually take most of the time
                    # -- the variant was only wired into the bar-marks loop, so
                    # it reported success while every canvas still rendered
                    # seat 0. See palette_variant().
                    pal_b = palette
                    if target.get("large") and is_palette_driven(key):
                        pal_b = palette_variant(palette, large_seat)
                    # Large canvases take the shared bar grid + beat pulse
                    # here too, so a matrix in TEXTURE role stays locked to
                    # its neighbours in FEATURE role. See hold_spans().
                    if target.get("large"):
                        n_shader_pulse += hold_spans(target, L_BASE, key,
                                                     section, palette,
                                                     large_seat, e)
                    else:
                        add_target(target, L_BASE, key, sec_start, sec_end, pal_b)
                elif target["name"] not in anchor_names_here:
                    key = "shimmer" if escalation >= 2 and e > 0.6 else "twinkle_soft"
                    add_target(target, L_BASE, key, sec_start, sec_end, palette)

            else:  # ROLE_OUTLINE
                if target["name"] in movement_props:
                    continue          # same reason as ROLE_TEXTURE above
                if wants_motion(target):
                    key = feature_effect_for(target, rng, escalation, shared_large_key)
                    if look_reverse:
                        key = REVERSE.get(key, key)
                    # Same family, different lead colour per canvas. This is
                    # the path the hero canvases actually take most of the time
                    # -- the variant was only wired into the bar-marks loop, so
                    # it reported success while every canvas still rendered
                    # seat 0. See palette_variant().
                    pal_b = palette
                    if target.get("large") and is_palette_driven(key):
                        pal_b = palette_variant(palette, large_seat)
                    if target.get("large"):
                        n_shader_pulse += hold_spans(target, L_BASE, key,
                                                     section, palette,
                                                     large_seat, e)
                    else:
                        add_target(target, L_BASE, key, sec_start, sec_end, pal_b)
                elif target["name"] not in anchor_names_here:
                    # A full-section Color Wash on the beat anchor would hold it
                    # lit through every kick flash -- the anchor must stay dark
                    # between hits or the flash has nothing to contrast against.
                    add_target(target, L_BASE, "wash", sec_start, sec_end, palette)

        log(section.start_time, "section",
            f"{section.label} #{inst} energy={e:.2f} -> {n_active}/{n_cast} targets lit "
            f"({n_active/n_cast*100:.0f}%), {n_feature} feature, palette={pal_name}")

    # --- WHOLE-TRACK BUILD, and the dropout at the end ---------------------
    # "Really I think the whole track is really an intro, so it needs to build
    # and get bigger and bigger throughout the whole thing... and at the end
    # there's the last drum hit where most of the instruments drop out and it
    # should be a dramatic dropout for the lights too where only outlines stay
    # on or only floods, something minimal but still large scale."
    #
    # Both are detected rather than hardcoded to one song. A track "builds" if
    # its energy rises across most of its length -- This Version of You
    # measures 0.16, 0.37, 0.65, 0.95, 1.00, 0.55 by sixths, i.e. four
    # consecutive rises. A track has a final dropout if the last strong drum
    # hit is followed by a long, much quieter tail (that song: hit at 155.7s,
    # then 16s at 0.41x the song mean).
    _sixth = []
    _rms_s = np.asarray(analysis.spectrum.rms_energy)
    _rt_s = np.asarray(analysis.spectrum.rms_times)
    _dur_s = float(analysis.duration_seconds)
    for _i in range(6):
        _m = (_rt_s >= _dur_s * _i / 6) & (_rt_s < _dur_s * (_i + 1) / 6)
        _sixth.append(float(_rms_s[_m].mean()) if _m.any() else 0.0)
    _rises = sum(1 for _i in range(4) if _sixth[_i + 1] > _sixth[_i] * 1.05)
    song_builds = _rises >= 3
    if song_builds:
        print(f"Whole-track build: energy rises through {_rises} of 5 steps -- "
              f"the show grows with it")

    final_dropout = None
    _kv2 = voices.get("kick")
    _sv2 = voices.get("snare")
    _hits = sorted(list(_kv2.onsets if _kv2 else []) + list(_sv2.onsets if _sv2 else []))
    if _hits:
        _last_hit = _hits[-1]
        _tail = _rms_s[_rt_s > _last_hit]
        if (_dur_s - _last_hit >= 8.0 and len(_tail)
                and _tail.mean() < _rms_s.mean() * 0.65):
            final_dropout = _last_hit
            print(f"Final dropout: last drum hit at {_last_hit:.1f}s, "
                  f"{_dur_s - _last_hit:.0f}s of tail at "
                  f"{_tail.mean() / _rms_s.mean():.2f}x the song mean")

    # --- AMBIENT PASSAGES: sequence the SHAPE, not the beat ----------------
    #
    # "For the slower/quieter/intro/outro type sections we need a different
    # approach that is more holistic and looks at how the entire section
    # builds or develops... instead of just repetitive rhythm and placing
    # voices on props, instead we identify the important moments and any slow
    # key/synth/bass/kick rhythms that are often like a single hit or
    # something happening only once every couple of bars."
    #
    # Everything else in this generator assumes a beat to hang things on. That
    # is right for a chorus and wrong for a drumless intro, where it produces
    # "lights pulsing and being very rhythmic when they need to be more smooth
    # and flowy". So passages with no drum pulse get a separate vocabulary:
    #
    #   * one long hold per passage on the big canvases, its brightness curve
    #     built from the passage's OWN measured energy, so the house follows
    #     the music's contour instead of marking time (envelope_palette)
    #   * a slow BLOOM on each sparse event -- the single bass notes and chord
    #     changes that are the only rhythm such a passage has. Blooms, not
    #     flashes: 0.3s in, over a second out.
    #   * an ENTRANCE mark when a voice arrives, which is the most legible
    #     moment a quiet passage owns.
    #
    # Note a passage can be LOUD and still belong here: This Version of You
    # runs at 0.78-0.80 loudness with zero drums for 37 seconds at a stretch.
    # Drumless is not the same as quiet, so the treatment scales with measured
    # loudness rather than assuming darkness.

    n_amb = 0
    _ambient_ids: set[int] = set()
    # Tagging that SURVIVES clip_for_blackout. That pass rebuilds effects with
    # model_copy, so every id()-based tag is silently lost halfway through the
    # pipeline -- which is how the ambient displacement filter came to delete
    # the very 37-second hold it existed to protect. `settings` is carried
    # through the copy, so the tag lives there. xLights ignores unknown keys
    # (X_Effect_Locked is a real one), so this costs nothing in the file.
    AMB_TAG = "X_AmbientHold"

    def tag_ambient(new_effects):
        for _e in new_effects:
            _e.settings = {**_e.settings, AMB_TAG: "1"}
            _ambient_ids.add(id(_e))

    def is_ambient_eff(_e) -> bool:
        return _e.settings.get(AMB_TAG) == "1" or id(_e) in _ambient_ids


    if ambient_passages:
        _rms = np.asarray(analysis.spectrum.rms_energy)
        _rt = np.asarray(analysis.spectrum.rms_times)
        amb_rng = random.Random(f"{SONG_NAME}|ambient")
        big = [t for t in cast if t.get("large")]
        movers = [t for t in cast if prominence_of(t) >= 2 and not t.get("large")]
        # Every linear prop -- outlines, verticals, horizontals, gutters. These
        # are what make a whole-house hit read as one gesture: thin, bright,
        # and spread right across the facade.
        amb_lines = [t for t in cast
                     if t["cat"] in ("single_line", "icicle")
                     or any(h in t["name"].lower()
                            for h in ("outline", "vertical", "horizontal",
                                      "c9", "gutter"))]
        # The families the chord hits cycle through, in a fixed order so the
        # sequence repeats and reads as deliberate.
        def _fam(*words):
            # No cap. Capping at eight meant a chord that landed on a small
            # family lit three props while its neighbour lit twenty-five, so
            # the progression pulsed in size for no musical reason. "Using
            # them all but changing color on the beat" -- every member of the
            # family goes, and the colour is what distinguishes the hits.
            return [t for t in cast
                    if any(w in t["name"].lower() for w in words)]

        # Families under a quarter of the largest are folded into the next
        # one rather than used alone. Measured on This Version of You the
        # families ran 25, 17, 7 and 3 props, so consecutive chords lit wildly
        # different amounts of house and the progression appeared to swell and
        # shrink at random. Every chord should light a comparable group; the
        # COLOUR is what distinguishes them.
        _raw = [f for f in (_fam("vertical"), _fam("horizontal"),
                            _fam("outline"), _fam("c9", "gutter"),
                            _fam("arch"))
                if len(f) >= 2]
        # Balance by the number of PROPS a family actually lights, not by the
        # number of cast entries in it. Measured on This Version of You the
        # entry counts were 5/2/4 -- near enough even -- while the props they
        # expanded to were 17/7/3, so the merge never fired and consecutive
        # chords still lit 17 props then 3.
        def _famsize(f):
            return sum(len(_xgroups.get(t["name"], []) or [1]) for t in f)

        amb_cycle = []
        if _raw:
            _floor = max(_famsize(f) for f in _raw) * 0.4
            for _f in _raw:
                if _famsize(_f) < _floor and amb_cycle:
                    _seen = {t["name"] for t in amb_cycle[-1]}
                    amb_cycle[-1] = amb_cycle[-1] + [t for t in _f
                                                     if t["name"] not in _seen]
                else:
                    amb_cycle.append(_f)
        print("  ambient chord families (props lit): raw "
              + "/".join(str(_famsize(f)) for f in _raw)
              + " -> merged " + "/".join(str(_famsize(f)) for f in amb_cycle))
        n_big_hits = 0
        for pi, ap in enumerate(ambient_passages):
            t0, t1 = ap["t0"], ap["t1"]
            si_at = next((i for i, sec in enumerate(analysis.sections)
                          if sec.start_time <= t0 < sec.end_time), 0)
            _sm = section_meta[si_at] if si_at < len(section_meta) else {}
            pal = (_sm.get("palette") if isinstance(_sm, dict) else None) or PALETTES["ice"]
            # Intensity follows the passage's measured loudness: a loud
            # drumless synth passage should be big, a dark intro should not.
            loud = float(ap.get("loud", 0.5))
            # On a track that builds, WHERE the passage sits matters as much
            # as how loud it is: an intro passage should be smaller than the
            # same passage two thirds of the way in, so the show grows with
            # the song rather than restating one level.
            if song_builds:
                _pos = min(1.0, max(0.0, (t0 / max(_dur_s, 1e-6))))
                loud *= 0.55 + 0.85 * _pos
            # (a) the canvases hold ONE developing pattern for the whole
            #     passage, breathing with the section's own energy curve.
            key = amb_rng.choice([k for k in SHADER_KEYS
                                  if is_palette_driven(k) and not is_centred(k)]
                                 or SHADER_KEYS)
            # ...and run it SLOWLY. "The ambient sections should swell more
            # gently and slowly." A shader at song speed keeps putting motion
            # at the beat rate even when nothing is pulsing, which is a large
            # part of why these passages still read as rhythmic. Registered as
            # a separate slowed key so the same shader elsewhere is unaffected.
            _slow_key = key + "__ambslow"
            if _slow_key not in EFFECTS:
                _en, _sett = EFFECTS[key]
                _sett = dict(_sett)
                _sett["E_SLIDER_Shader_Speed"] = str(
                    max(20, int(int(_sett.get("E_SLIDER_Shader_Speed", 400)) * 0.30)))
                EFFECTS[_slow_key] = (_en, _sett)
            key = _slow_key
            # EVERY recognisable prop carries the contour, not just the big
            # canvases. Ten targets out of several hundred moved the measured
            # correlation between display brightness and music energy by 0.01
            # -- the shaped layer was sitting on an unshaped floor, exactly as
            # the drop/fade envelope did before the same fix.
            _st = {t: 0.0 for t in ap["sparse"]}
            for _vn in ("kick", "bass", "chord"):
                _v = voices.get(_vn)
                if _v is None:
                    continue
                for _t, _s2 in zip(_v.onsets, _v.strengths):
                    for _e3 in ap["sparse"]:
                        if abs(_t - _e3) < 0.45:
                            _st[_e3] = max(_st[_e3], float(_s2))
            _vals = sorted(_st.values())
            _big_cut = _vals[int(len(_vals) * 0.6)] if _vals else 0.0
            # SHOW THE HARMONY AT SCALE. "In TVOY there is a slow piano part
            # with one chord at a time and I want to show that on some large
            # scale because it's the main focus point. In Meridian there is
            # also a synth part that's relatively slow."
            #
            # The chord voice IS that part. Splitting the passage's hold at
            # chord changes gives every chord its own colour across the big
            # canvases, so the harmony is what the house is doing rather than
            # something buried under a texture. Thinned to 2.5s so a busy
            # moment cannot turn this into flicker, and it falls back to one
            # unbroken hold when a passage has no chord movement to show.
            _ch = []
            _cv = voices.get("chord")
            if _cv is not None:
                _last = -9.0
                for _t in _cv.in_window(t0, t1):
                    if _t - _last >= 2.5:
                        _ch.append(_t)
                        _last = _t
            marks_a = [t0] + [c for c in _ch if t0 + 1.5 < c < t1 - 1.5] + [t1]
            for seat, tgt in enumerate(big + movers):
                _k = key if tgt.get("large") else "wash"
                # THE HARMONY GOES ON PROPS, NOT THE CANVASES. "Showing
                # piano/bass/synth rhythms should be on props not on the whole
                # matrix canvases across the whole house." Splitting the big
                # canvases per chord turned the whole facade over on every
                # chord, which is far too much surface for a piano note. The
                # canvases hold ONE slow envelope across the passage and the
                # props carry the chord changes.
                # A big canvas holds one pattern per passage EXCEPT at the
                # passage's strongest hits, which get a new colour and shader
                # so the moment lands. Harmony's single kick at 1:17 is the
                # case: deferring the whole passage to one hold removed the
                # change entirely, and Krists wants that hit to be the
                # impactful thing in a minute of stillness. Capped at three so
                # the passage still reads as one held idea, not a sequence.
                if tgt.get("large"):
                    # Ranked by ISOLATION, not loudness. Harmony's single kick
                    # at 1:17 is the moment Krists wants marked, and it is not
                    # among the passage's loudest hits -- what makes it land is
                    # that nothing else happens for seconds either side of it.
                    # Strength still breaks ties.
                    # EVERY hit with real silence in front of it, not a
                    # top-N: ranking picked the three most isolated and they
                    # were all early in the passage, so Harmony's 1:17 kick --
                    # the one moment in a quiet minute -- still got nothing.
                    # A gap of three seconds or more is what makes a hit read
                    # as an event; capped only to stop a sparse passage
                    # turning into a sequence of changes.
                    # Isolation is judged on the DRUM and BASS hits alone,
                    # not on the combined event list. Harmony's intro carries
                    # a chord roughly every second, so nothing in the combined
                    # list ever looks isolated -- while in the drums the 1:17
                    # kick has seven clear seconds in front of it, which is
                    # exactly why it lands as a moment to the ear.
                    _drum_ev = sorted(t for vn in ("kick", "bass")
                                      if vn in voices
                                      for t in voices[vn].in_window(t0, t1))
                    _tops = [e for i, e in enumerate(_drum_ev)
                             if (e - _drum_ev[i - 1] if i else 9.0) >= 3.0][:6]
                    _marks = sorted({t0, t1} | {e for e in _tops
                                                if t0 + 4 < e < t1 - 4})
                else:
                    _marks = marks_a
                for mi in range(len(_marks) - 1):
                    a_ms, b_ms = int(_marks[mi] * 1000), int(_marks[mi + 1] * 1000)
                    # Each chord takes the next colour of the family, so the
                    # change is visible as a change and not just a re-trigger.
                    p_ = (palette_variant(pal, seat + mi) if is_palette_driven(_k)
                          else pal)
                    p_ = envelope_palette(p_, _rms, _rt, a_ms, b_ms,
                                          depth=0.70 + 0.50 * loud)
                    _n0 = len(effects)
                    add_target(tgt, L_BASE, _k, a_ms, b_ms, p_)
                    tag_ambient(effects[_n0:])
                    n_amb += 1
            # (b) THE SINGLE HITS ARE THE DRAMA, so they get the whole house.
            #
            # "When there is a single kick or single bass hit we need to make
            # those dramatic, not just a gentle bloom on some props, we need
            # something big like all outlines and horizontals and verticals do
            # a big bright pulse, like a sudden hit and fade. Or maybe a big
            # shockwave across the whole house... The ambient sections in the
            # songs I choose are usually not just calm or drumless, they are
            # dramatic and the important moments need to really show."
            #
            # The strongest events in the passage get a full-house gesture:
            # every outline, vertical and horizontal hit at once with a sharp
            # attack and a long decay, and a shockwave travelling across the
            # big canvases underneath. The rest keep the quieter bloom, so the
            # big ones stay big by comparison -- if every hit were huge, none
            # of them would read as a moment.
            hold = int(min(2400, max(900, (t1 - t0) / max(len(ap["sparse"]), 1) * 900)))
            for ei, et in enumerate(ap["sparse"]):
                ms = int(et * 1000)
                is_big = _st.get(et, 0.0) >= _big_cut and _big_cut > 0
                if is_big and amb_lines:
                    # The linear props carry the CHORD, so they take the bright
                    # accent; the canvases underneath stay in the section
                    # palette so there is still a background for it to read
                    # against. "The final 4 chords are not really visible,
                    # there's just a faint blue glow on the outlines but it
                    # needs to show the chords in a contrasting color."
                    hit_pal = accent_bright(contrast_palette(pal, ei), 0.44)
                    _acc = {"X_AmbientAccent": "1"}
                    # sudden hit, long fade -- across every linear prop at once
                    for tgt in amb_lines:
                        _n0 = len(effects)
                        add_target(tgt, L_MOTION, "wash", ms, ms + 2600, hit_pal,
                                   extra={"T_TEXTCTRL_Fadein": "0.05",
                                          "T_TEXTCTRL_Fadeout": "2.10", **_acc})
                        tag_ambient(effects[_n0:])
                        n_amb += 1
                    # ...and a shockwave crossing the big canvases under it
                    # The canvases SWELL, they do not sweep. A travelling
                    # gesture crossing a matrix reads as several flashes --
                    # measured on Harmony's intro, the Sun Room and front
                    # window matrices showed 2.9 and 3.3 brightness peaks per
                    # hit while the linear props showed one: "the sun room
                    # window matrix, three upper window matrices, and floods
                    # all seem to flash twice on each of those hits that
                    # should be a smooth hit and fade like the other props."
                    # One rise and one decay, matching what the outlines do.
                    for tgt in big:
                        _n0 = len(effects)
                        add_target(tgt, L_MOTION, "wash", ms, ms + 2400, pal,
                                   extra={"T_TEXTCTRL_Fadein": "0.06",
                                          "T_TEXTCTRL_Fadeout": "2.00"})
                        tag_ambient(effects[_n0:])
                        n_amb += 1
                    n_big_hits += 1
                elif amb_cycle:
                    # A REPEATING PATTERN, not a different prop every time.
                    #
                    # "The main piano chord progression, looks like you're
                    # getting the rhythm but every hit does something totally
                    # different... it should be more cohesive, like every
                    # chord hit should do the same thing but maybe just with a
                    # different color or shape. Like hit all the verticals
                    # then the horizontals, then verticals again, then all the
                    # outlines."
                    #
                    # Rotating one prop per hit out of dozens is unfollowable:
                    # nothing recurs, so there is no pattern for the eye to
                    # latch onto. The hits now cycle through a short, fixed
                    # sequence of prop FAMILIES, every member of the family
                    # lighting together, with the colour advancing each time.
                    # Same gesture, different colour -- which is exactly what
                    # makes a slow chord progression readable.
                    fam = amb_cycle[ei % len(amb_cycle)]
                    # Contrast, not a variation. palette_variant kept the hit
                    # inside the section family, so in a grey/blue passage the
                    # chords were grey/blue on top of grey/blue and simply did
                    # not show. The accent hue advances per hit, which is the
                    # "changing color on the beat" half of what was asked for;
                    # amb_cycle rotating the prop family is the other half.
                    fam_pal = accent_bright(contrast_palette(pal, ei))
                    for tgt in fam:
                        _n0 = len(effects)
                        add_target(tgt, L_MOTION, "wash", ms, ms + hold, fam_pal,
                                   extra={"T_TEXTCTRL_Fadein": "0.25",
                                          "T_TEXTCTRL_Fadeout": f"{hold / 1000.0 * 0.8:.2f}",
                                          "X_AmbientAccent": "1"})
                        tag_ambient(effects[_n0:])
                        n_amb += 1
            # (c) an entrance gets a wider reveal -- several props at once,
            #     held long. This is the moment the passage is "about".
            if ap.get("enters") and movers:
                for j in range(min(4, len(movers))):
                    tgt = movers[(pi * 5 + j) % len(movers)]
                    _n0 = len(effects)
                    add_target(tgt, L_MOTION, "wash", int(t0 * 1000),
                               int(t0 * 1000) + 3200, pal,
                               extra={"T_TEXTCTRL_Fadein": "0.80",
                                      "T_TEXTCTRL_Fadeout": "2.20"})
                    tag_ambient(effects[_n0:])
                    n_amb += 1
        _probe("after ambient pass")
        print(f"Ambient passages: {len(ambient_passages)} treated as shape, not "
              f"beat ({n_amb} placements, {n_big_hits} full-house hits on "
              f"{len(amb_lines)} linear props; "
              + ", ".join(f"{p['t0']:.0f}-{p['t1']:.0f}s" for p in ambient_passages[:5])
              + ")")

    # --- DROP/FADE CYCLE HOLDS ---------------------------------------------
    # One shader per cycle on every big canvas, from the kick that starts the
    # drop to the kick that starts the next one. A dedicated pass rather than
    # a rule inside the section loop, for the reason this project has hit
    # three times now: section-driven placement only fires when a canvas is in
    # that section's active set, and across a quiet intro that is nearly
    # never, so the gesture silently disappears instead of landing.
    #
    # "There is a single kick drum for each drop and everything between that
    # and the next single kick drum is the slow fade... It all needs to be
    # synced to the kick drum."
    n_cycle = 0
    # Protected from the clipping passes below. These holds ARE the intro's
    # structure -- a blackout or a run-up dip trimming one leaves the canvas
    # dark for part of a fade and puts the next change off the kick, which is
    # the whole complaint. Measured: two of Gooey's five cycles lost their
    # spans this way before the exemption.
    _cycle_ids: set[int] = set()
    if drop_cycles:
        # EVERY recognisable prop, not just the big canvases. Measured at
        # t=35s, deep inside a fade: the canvases were correctly down at ~40%
        # of nominal, and `Flake I 1` -- which is not "large" and so never got
        # the envelope -- was putting out 34% of ALL the light on the house by
        # itself, holding the floor up so the next kick could only lift 1.2x.
        # The gesture is the whole display decaying together.
        cyc_canvases = [t for t in cast if prominence_of(t) >= 2]
        cyc_rng = random.Random(f"{SONG_NAME}|drop-cycles")
        for ci, (c0, c1) in enumerate(drop_cycles):
            # One shader for the whole cycle, shared across the canvases so
            # they read as one surface, changing only on the next drop.
            key = cyc_rng.choice([k for k in SHADER_KEYS
                                  if is_palette_driven(k) and not is_centred(k)]
                                 or SHADER_KEYS)
            si_at = next((i for i, sec in enumerate(analysis.sections)
                          if sec.start_time <= c0 < sec.end_time), 0)
            _sm = section_meta[si_at] if si_at < len(section_meta) else {}
            pal = (_sm.get("palette") if isinstance(_sm, dict) else None) or PALETTES["ice"]
            s_ms, e_ms = int(c0 * 1000), int(c1 * 1000)
            for seat, tgt in enumerate(cyc_canvases):
                # Big canvases carry the shader; everything else simply glows
                # and decays with it, so the fade is one gesture across the
                # house rather than a few props doing it over a lit floor.
                _k = key if tgt.get("large") else "wash"
                p = palette_variant(pal, seat) if is_palette_driven(_k) else pal
                p = fade_palette(p)
                cohort_marks.update((s_ms, e_ms))
                # Track the actual effect OBJECTS, not (name, start, end).
                # add()/add_target snap times to the frame grid and later
                # passes rewrite them again, so a tuple key recorded here
                # stops matching and the exemption silently does nothing --
                # which is exactly how cycle 4 of Gooey's intro ended up
                # clipped to a 75ms fragment while the others survived.
                _n0 = len(effects)
                add_target(tgt, L_BASE, _k, s_ms, e_ms, p)
                for _e in effects[_n0:]:
                    _cycle_ids.add(id(_e))
                n_cycle += 1
            log(c0, "drop-cycle",
                f"cycle {ci + 1}/{len(drop_cycles)}: {key} held "
                f"{c1 - c0:.1f}s across {len(cyc_canvases)} canvases")
        # SOMETHING STAYS ON THROUGH THE FADE. "Maybe when everything is
        # fading in the intro we could leave something still on, like some
        # outlines or something, so it's not like completely everything fades
        # away."
        #
        # A steady, dim hold across the WHOLE passage rather than per cycle:
        # it should not breathe with the drops, it should be the constant the
        # drops happen against. Outlines are the right prop for it -- thin,
        # architectural, and they read as the house still being there.
        keepers = [t for t in cast
                   if (t["cat"] in ("single_line", "icicle")
                       or any(h in t["name"].lower() for h in OUTLINE_NAME_HINTS))
                   and "/" not in t["name"]][:6]
        if keepers:
            _k0, _k1 = int(drop_cycles[0][0] * 1000), int(drop_cycles[-1][1] * 1000)
            _dim = PALETTES["ice"].model_copy(update={"extra_parts": [
                "C_VALUECURVE_Brightness=Active=TRUE|Id=ID_VALUECURVE_Brightness|"
                "Type=Custom|Min=0.00|Max=400.00|RV=TRUE|Values=0.00:0.09;1.00:0.09"]})
            for tgt in keepers:
                _n0 = len(effects)
                add_target(tgt, L_BASE, "wash", _k0, _k1, _dim)
                for _e in effects[_n0:]:
                    _cycle_ids.add(id(_e))
            print(f"Drop/fade cycles: {len(keepers)} outlines held on through "
                  f"the fades ({', '.join(t['name'] for t in keepers[:3])}...)")
        print(f"Drop/fade cycle holds: {n_cycle} "
              f"({len(drop_cycles)} cycles x {len(cyc_canvases)} canvases)")

    print(f"Staging: {dict(role_counts)}")
    print(f"After orchestration: {len(effects)} effects")

    # =====================================================================
    # PASS 2 -- signature musical cues
    # =====================================================================
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for t in cast:
        by_cat[t["cat"]].append(t)

    # Physical orders for whole-display sweeps. `wants_motion` targets
    # excluded -- display_sweep gives each stop a brief flat on_pop/on_swell
    # hit, and a large canvas or big decorative group (Candy Canes...) is now
    # guaranteed real animated screen time every section (see the
    # TEXTURE/OUTLINE handling above), so it doesn't need a flash-style hit
    # riding on top of that too. Was `large`-only; Candy Canes (custom, n=8,
    # not large) kept showing up here with the tell-tale scattered irregular
    # on_pop/on_swell timing of this + stab/detail pools, dominating its
    # effect mix with "On" despite the PULSE-role redirect already covering
    # the ROLE-based flash path -- this is a separate pass entirely.
    cast_by_y = sorted((t for t in cast if not wants_motion(t)
                        and t["name"] not in prop_voice),
                       key=lambda t: t["cy"])                 # bottom -> top
    cast_by_x = sorted((t for t in cast if not wants_motion(t)
                        and t["name"] not in prop_voice),
                       key=lambda t: t["cx"])                 # left -> right

    # --- 2-pre. Persistent kick-locked body pulse -----
    # Direct feedback on a house track: "it's a dance song that's 4 on the
    # floor, I don't feel the 4 on the floor in the lights... we need some
    # persistent lighting patterns that match that with pulsing or movement,
    # ideally in multiple places at once."
    #
    # Everything else in this generator is EVENT-driven (a riser here, a stab
    # there), which is why the constant underlying pulse never appeared:
    # nothing was modelling "the thing the whole track is built on". This
    # pass is deliberately different in kind -- an unbroken heartbeat running
    # underneath the song wherever there's real energy.
    #
    # Placed on the REAL kick onset times (kick_times_in), not an assumed
    # uniform quarter-note grid -- direct correction on an R&B track that
    # scored persistent-kick coverage (0.84) well above the house-validated
    # threshold: "it's not really four on the floor, it's more of an r&b
    # beat." The coverage gate was right (a kick IS clearly present), but
    # placing hits on a uniform beat grid regardless of the real pattern is
    # what made it read as generic dance-music four-on-the-floor rather than
    # the track's actual groove. See detect_kick_pattern().
    #
    # Rate safety: `split_flashes_and_bursts` (the same helper the PULSE role
    # uses) still caps this at the ~3.5Hz flash ceiling -- a real kick can
    # double-time in a fill, and this must not turn a fast run of real hits
    # into strobing.
    # EVERY song gets a beat marker. This used to be gated on
    # `kick_persistent` (>=80% kick coverage), which was the right test for
    # "is this four-on-the-floor" -- a GENRE question -- but was being used to
    # decide "should anything mark the beat at all", which is a UNIVERSAL
    # need. Measured across the library: Gooey 58.8%, World Is Mine 66.2%,
    # The Ambition 64.1%, Fara 77.7% all fell below the gate, so FOUR OF
    # EIGHT songs had no beat pulse whatsoever. That is exactly "there's
    # almost nothing pulsing or flashing to the main beat of each song".
    #
    # Krists asked for the shape of the fix directly: "Pick something to flash
    # on the kick drum, or pick a group and rotate through on each kick."
    # So one prop flashes per kick and the assignment ROTATES around the
    # house, which marks the beat unmistakably without any single prop
    # blinking repetitively -- the specific complaint about the floods.
    n_pulse = 0
    def beat_hit_key(target: dict) -> str:
        """How a given prop should MARK THE BEAT, by its shape.

        A flat `On` is right for a discrete object (a star, an orb, a candy
        cane) -- the whole thing lights, which is exactly what you want. It is
        wrong for a 2D canvas: "the three upper windows aren't great for
        flashing because they're matrices, we can flash a pattern or shader or
        flash the outline but flashing the whole window matrix green isn't the
        best use of a matrix." A matrix flashing flat is a coloured rectangle,
        and it wastes the only props on the house that can draw a shape.

        So matrices get a radial burst that still lands on the beat but reads
        as a pattern, linear props get a fast bidirectional chase from the
        centre (a beat you can see travel), and everything else flashes.
        """
        if target["name"] in CONE_SHAPED:
            # A cone wants motion that wraps it, not a flat ring. Keeps the
            # Mega Tree marking the beat without "pulsing with a basic color."
            return "chase_mid_out"
        # (see also cone_safe_key -- nothing anywhere may flat-flash a cone)
        if target.get("large") or target["cat"] == "matrix":
            # Shockwave: a ring expanding from the centre on each hit. Sharp
            # enough to read as rhythmic, but it draws rather than fills.
            return "shockwave"
        if target["cat"] in ("single_line", "icicle"):
            # A LINEAR PROP THAT IS ALREADY CHASING FLASHES INSTEAD.
            #
            # Measured on Aperture: ten props carried two chase layers at very
            # different rates -- a smooth 925ms ensemble chase plus a 200ms
            # staccato one from this beat hit, on the same prop. Two chase
            # speeds at once is what reads as "fast and erratic" rather than
            # the smooth slow motion the verticals show when they only have
            # one. The beat still lands; it just lands as a pulse on a prop
            # that is already moving, instead of a second competing travel.
            if target["name"] in movement_props:
                # A burst is not a second TRAVEL. The rule here is that a prop
                # already carrying a chase must not get a competing chase at a
                # different rate -- that stands. But a flat `On` was not the
                # only way to satisfy it: a shockwave expands from the middle
                # rather than running along the strand, so it marks the beat
                # without racing the motion already there, and on a 639-node
                # candy cane it draws the cane instead of filling it.
                return random.Random(f"pulse|{target['name']}").choice(
                    ["on_pop", "shockwave", "on_pop", "shockwave", "on_pop"])
            # C9 gutters and other free linear props: a burst travelling out
            # from the middle marks the beat AND shows the prop's length.
            return "chase_mid_out"
        if target["cat"] == "arch":
            return "on_hit"
        # A DECORATIVE PROP WITH GEOMETRY MOVES; IT DOES NOT JUST FLASH.
        #
        # Everything that reached here got a flat `on_pop`, which made the
        # small props 58-79% flat flashes. Measured against Krists's own 2025
        # sequences on the same families:
        #
        #   showstopper   his Shockwave 46% / On 36%   mine On 58%, Shockwave 11%
        #   snowflake     his Shockwave 45% / On 10%   mine On 69%, Shockwave  5%
        #   mini tree     his SingleStrand 32%, Curtain 14%, Shockwave 14%,
        #                 On 5%                        mine On 79%
        #
        # These props have real shape -- arms, spokes, rings, a cone -- and he
        # marks the beat by drawing that shape rather than filling it. The
        # choice is seeded by the prop name so a given prop keeps its own
        # signature through the song instead of flickering between gestures.
        _nm = target["name"].lower()
        _r = random.Random(f"beat-hit|{target['name']}")
        if "hanging sphere" in _nm:
            # A sphere is 8 strands of 30 -- it can travel along itself, not
            # just flash. Up/down fills, sparkle, a ring from the middle, and
            # the alternating-strand stripes, so the seven read as objects
            # doing something rather than seven lamps switching.
            return _r.choice(["fill_up", "fill_down", "sphere_sparkle",
                              "shockwave", "strand_stripes", "ripple_hex_in",
                              "chase_mid_out", "on_pop"])
        if any(h in _nm for h in ("mini tree", "tree")):
            # Cone-shaped: travel along it, or shut it like a curtain.
            return _r.choice(["chase_mid_out", "chase_bounce_drift",
                              "curtain_mid", "shockwave", "on_pop"])
        if any(h in _nm for h in ("flake", "snowflake", "showstopper",
                                  "spinner", "sphere", "star", "wreath")):
            # Radial by construction: a ring or polygon expanding from the
            # middle traces the prop instead of blanking it.
            return _r.choice(["shockwave", "shockwave", "shockwave",
                              "ripple_hex_in", "ripple_hex_out", "ripple_rays",
                              "pinwheel", "on_pop", "on_pop",
                              # Bars that reverse on the beat, and that open
                              # from the middle rather than sweeping past.
                              "bars_alt_up", "bars_expand", "bars_three",
                              "pinwheel_4", "pinwheel_6_slow", "pinwheel_twist"])
        # THE DEFAULT IS MOVEMENT, NOT A FLASH.
        #
        # This fallback produced 81% of every `On` in the song, and what it
        # was flashing is not point-like: candy canes at 600-639 nodes, canes
        # at 567, presents at 150. A 639-node cane lit flat is a coloured
        # shape; a chase along it draws the cane.
        #
        # Measured across 17 of Krists's own sequences: On is 6% of his
        # effects and SingleStrand 30%, against 71% and 15% here. He has 46
        # distinct effect types in play; this generator had 14. The flash stays
        # in the pool -- it is what makes the stepping read as rhythm rather
        # than as continuous motion -- but it is no longer the whole answer.
        return _r.choice(["chase_mid_out", "chase_bounce_pp", "chase_spark",
                          "shockwave", "on_pop", "on_pop"])

    # accent_pool / anchor_rank / section_anchor are all computed before the
    # staging passes above, so those passes can suppress the ambient fill on
    # whichever prop is anchoring this section.
    # PURE ROTATION IS NOT A BEAT MARKER. Rotating one prop per kick through
    # the whole pool means each individual prop fires once every N kicks --
    # measured on World Is Mine, 695 flashes over 9 props at ~123 BPM works out
    # to one flash per prop every 4.5 SECONDS. Nothing is marking the beat;
    # what a viewer sees is scattered single flashes wandering around the
    # house. "Still not seeing kick drum flashing clearly anywhere... I don't
    # see anything going with the very strong dance beat."
    #
    # So: an ANCHOR flashes on EVERY kick for a whole section -- that is the
    # thing that "always shows that beat clearly" -- and the rotation continues
    # underneath it as travel and variety. The anchor changes each section, so
    # no single prop blinks for the entire song, which was the original reason
    # for rotating in the first place ("just pulsing the floods on the upbeats
    # all the time... gets repetitive too"). Both asks are satisfiable at once;
    # the rotation alone was not.
    # WHAT DOES THE PULSE FOLLOW -- the quarter note, or the kick?
    #
    # Both answers are right, for different songs, and Krists gave both in
    # consecutive rounds: "I want more of that rhythmic pulsing with the main
    # beat which is usually just quarter notes" (general), then, on ENIGMA,
    # "this song isn't 4 on the floor, it's got more of a hip hop beat so we
    # could pulse on the kick drum, not quarter notes."
    #
    # So the generator decides per song, from how the kick actually behaves.
    # A four-on-the-floor track has a kick on nearly every beat; a hip-hop or
    # bass-music track has a sparse, syncopated kick carrying the groove --
    # measured, ENIGMA's kick sits 5.0 beats apart at the median while its
    # backbeat comes from kick+snare together. Pulsing a syncopated song on
    # rigid quarter notes flattens exactly the thing that makes it groove.
    kick_v = voices.get("kick")
    _kick_cov = kick_coverage
    # Coverage alone put The WhatsApp Shakespeare (84%) in quarter mode, but
    # it is an R&B track and wants the groove -- Krists flagged it directly.
    # A four-on-the-floor kick is not just PRESENT on most beats, it is EVENLY
    # spaced; an R&B kick covers many beats while syncopating between them.
    # Median spacing near one beat is what actually distinguishes them.
    _even = kick_v is not None and kick_v.median_ipi_beats <= 1.35
    # ...AND the song has to actually drive. So Easy (To Fall in Love) passes
    # both kick tests -- 94% coverage, median spacing 1.25 beats, and its
    # tempo is genuinely 143.6 (checked for half-time detection: its beats
    # show no strong/weak alternation) -- yet Krists heard it as wrong:
    # "it seems like it's in 4-on-the-floor mode but it's a mellow song."
    #
    # He is right, and the kick pattern cannot see it. What separates that
    # song is how PERCUSSIVE it is: its drums stem carries 0.441 of the mix's
    # RMS, the lowest in the library by a clear step (next is Gooey at 0.548,
    # up to Aperture at 0.773). A soft kick landing on every beat of a ballad
    # is not four-on-the-floor in the sense that matters here -- flashing the
    # house on all 143 of those beats per minute is.
    _drive = percussive_ratio() >= 0.50
    if _kick_cov >= 0.75 and _even and _drive:
        pulse_mode = "quarter"
    elif kick_v is not None and len(kick_v) >= 24:
        pulse_mode = "groove"          # kick (+ snare backbeat) drives it
    elif voices.get("bass") is not None and len(voices["bass"]) >= 24:
        # NO USABLE DRUMS IS NOT A REASON TO IMPOSE A GRID. Falling back to
        # quarter here was backwards: This Version of You has EIGHT kick
        # onsets and no detected snare at all, so it took the metronome
        # branch -- the most rigid option in the generator -- on the least
        # rhythmic track in the library. Where there are no drums the bass is
        # the pulse, and following it is what "they don't really have a steady
        # quarter kick and are more dynamic" asks for.
        pulse_mode = "groove"
    else:
        pulse_mode = "quarter"         # nothing rhythmic at all -- last resort
    # Per-song override. Meridian measures as a steady four-on-the-floor (93%
    # kick coverage, evenly spaced, drums 0.580 of the mix) and Krists hears
    # it the other way: "they don't really have a steady quarter kick and are
    # more dynamic." The measurement is not wrong about the kick, it just
    # cannot hear that the kick keeps stopping -- the song has 20s drumless
    # stretches. Recorded as an explicit taste call rather than bent into the
    # gate, which would have flipped Fara and 12 to 12 along with it.
    pulse_mode = PULSE_MODE_OVERRIDE.get(SONG_NAME, pulse_mode)

    def pulse_times_in(t0: float, t1: float) -> list[float]:
        if pulse_mode == "quarter":
            return [b for b in beats if t0 <= b < t1]
        # Groove mode: the kick, plus the snare backbeat, which together are
        # what a listener hears as "the beat" on a hip-hop feel. Merged and
        # thinned so two hits 40ms apart don't double-fire on one prop.
        src = list(kick_v.in_window(t0, t1))
        sn = voices.get("snare")
        if sn is not None:
            src += list(sn.in_window(t0, t1))
        src.sort()
        out: list[float] = []
        min_gap = beat_ms / 1000.0 * 0.4
        for t in src:
            if not out or t - out[-1] >= min_gap:
                out.append(t)
        return out

    # ONE BEAT PER PROP AT A TIME, across every pass that marks the beat.
    #
    # "It's messy when the same prop or group shows multiple different beats at
    # the same time... maybe we can make it so a prop or group (like the arches
    # especially) only shows one beat or anchor within the same beat or two.
    # Fine if the beats light up at different parts of the bar, but often they
    # overlap and one takes away from the other."
    #
    # The per-section claim added earlier stopped a prop drawing two ROLES, but
    # not two beat EVENTS landing within a beat of each other from different
    # passes. This is the temporal version: every beat placement registers the
    # time it lit that prop (and its group/members/base model, since lighting a
    # group lights its members), and a later pass backs off if it would land
    # too close. Different parts of the bar are fine -- that's the point of the
    # window rather than a blanket exclusion.
    beat_guard: dict[str, float] = {}
    GUARD_BEATS = 1.5

    def _guard_keys(target_or_name) -> list[str]:
        if isinstance(target_or_name, str):
            return [target_or_name, _base_model(target_or_name)]
        t = target_or_name
        return [t["name"], _base_model(t["name"]), *t["members"]]

    def beat_free(target_or_name, t_s: float) -> bool:
        gap = GUARD_BEATS * beat_ms / 1000.0
        return all(t_s - beat_guard.get(k, -1e9) >= gap
                   for k in _guard_keys(target_or_name))

    def beat_take(target_or_name, t_s: float) -> None:
        for k in _guard_keys(target_or_name):
            beat_guard[k] = t_s

    n_anchor = 0
    n_beat_skipped = 0
    anchor_names: list[str] = []
    if accent_pool:
        hold = max(140, min(int(beat_ms * 0.45), 300))
        for si, section in enumerate(analysis.sections):
            if section.energy_level < 0.30:
                continue          # let genuinely quiet passages breathe
            pal = section_meta[si].get("palette") or PALETTES["gold"]
            flashes = pulse_times_in(section.start_time, section.end_time)
            # One anchor per section, walked through the ranked list so
            # consecutive sections never reuse the same prop.
            anchors = section_anchor.get(si) or [anchor_rank[si % len(anchor_rank)]]
            # ONE BEAT TREATMENT PER PROP PER SECTION. "There are some places
            # where the same prop has two beats at the same time, like the
            # arches sometimes share multiple beats and that doesn't work as
            # well if they overlap." A prop carrying two different beat
            # patterns at once reads as mush -- the viewer can't attribute
            # either one. Everything a prop covers (its own name, its members,
            # its base model) is claimed here so no later pass can double up.
            claimed_beat: set[str] = set()
            for _a in anchors:
                claimed_beat.add(_a["name"])
                claimed_beat.update(_a["members"])
                claimed_beat.add(_base_model(_a["name"]))
            for _a in anchors:
                if _a["name"] not in anchor_names:
                    anchor_names.append(_a["name"])
            # ANTIPHONAL: split one anchor left/right and alternate halves on
            # odd/even beats. "Using the left candy canes on beats 1 and 3 and
            # the right candy canes on beats 2 and 4 for example, could be
            # another trick to add to the repertoire." Nine groups on this
            # layout split cleanly (Candy Canes 2/2, Orbs 5/4, Stars 6/4,
            # Arches 2/2, C9s 3/4...), all derived from WorldPos -- no new
            # xLights groups needed.
            anti = None
            for _a in anchors:
                halves = split_lr(_a)
                if halves:
                    anti = (_a, halves)
                    break
            for bi, bt in enumerate(flashes):
                t_ms = int(bt * 1000)
                on_downbeat = any(abs(bt - d) <= 0.09 for d in downbeats)
                # Every held anchor hits, so several places in the house are
                # marking the beat at once rather than one lonely prop.
                # HOW it hits depends on what the prop IS. "The three upper
                # windows aren't great for flashing because they're matrices,
                # we can flash a pattern or shader or flash the outline but
                # flashing the whole window matrix green isn't the best use of
                # a matrix." A flat `On` turns a 2D canvas into a coloured
                # rectangle, which is both ugly and a waste of the only props
                # that can draw anything.
                dur = hold if not on_downbeat else int(hold * 1.4)
                for _a in anchors:
                    # Anchors are EXEMPT from the guard. An anchor's whole job
                    # is to hit every beat for the section; running it through
                    # a "no two beats close together" filter blocks it from
                    # doing the one thing it exists for. It still REGISTERS
                    # each hit, so every other pass keeps clear of it.
                    beat_take(_a, bt)
                    if anti is not None and _a["name"] == anti[0]["name"]:
                        # This one answers itself side to side instead of
                        # flashing whole: left half on 1 and 3, right on 2 and 4.
                        side = anti[1][bi % 2]
                        for m in side:
                            add(m, L_PULSE, "on_pop", t_ms, t_ms + dur, pal)
                        n_anchor += 1
                        continue
                    add_target(_a, L_PULSE, beat_hit_key(_a), t_ms, t_ms + dur, pal)
                    n_anchor += 1
                # Rotation continues as travel: a different prop each kick, and
                # on the downbeat two more spread across the house so the bar
                # start reads heavier than the beats inside it.
                # "Some of these songs are really dancey songs, they should
                # be pulsing like crazy on the main drops and choruses!" The
                # number of props hitting together now scales with the
                # section's energy instead of being a fixed 1 (3 on downbeats):
                # a quiet verse still gets a single clean marker, a chorus gets
                # a broadside across the house.
                width = 1 + int(round(4 * section_intensity.get(si, 0.5)))
                if on_downbeat:
                    width += 2
                # HOLD THE ROTATING SET FOR TWO BARS. Advancing it every beat
                # meant a prop marked the beat for a single bar and vanished --
                # "there are places I notice where a prop or group is showing
                # the main beat but only for a bar, so maybe it could stay a
                # bit longer." Now the same props carry it for eight beats
                # before the set moves on.
                # Hold the focal set for FOUR bars. "There are some great beats
                # that sometimes only last a few beats or 1 bar, maybe we could
                # make them last a bit longer so you can have the focal point a
                # little longer before it changes, unless the section changes."
                # Section changes reset it anyway, since bi restarts per
                # section -- so a short section still gets one stable set.
                # Count BARS FROM TIME, not pulse events. `bi` indexes pulse
                # events, and in groove mode those are merged kick+snare hits
                # -- roughly two per beat -- so `bi // 16` was about two bars,
                # not four, and varied with the drum pattern. Krists spotted it
                # exactly: "maybe you're counting beats as bars?" Deriving from
                # elapsed time makes four bars mean four bars on every song.
                rot = int((bt - section.start_time) / max(bar_s * 4.0, 1e-6))
                extras = [accent_pool[(rot + k * len(accent_pool) // max(1, width))
                                      % len(accent_pool)]
                          for k in range(min(width, len(accent_pool)))]
                for tgt in extras:
                    # Skip anything already carrying a beat this section --
                    # including via group membership, which is how the Arches
                    # ended up with two overlapping patterns (the group as an
                    # anchor, individual arches via the rotation).
                    if (tgt["name"] in claimed_beat
                            or _base_model(tgt["name"]) in claimed_beat
                            or any(m in claimed_beat for m in tgt["members"])):
                        continue
                    if not beat_free(tgt, bt):
                        n_beat_skipped += 1
                        continue
                    beat_take(tgt, bt)
                    add_target(tgt, L_PULSE, beat_hit_key(tgt), t_ms,
                               t_ms + hold, pal)
                    n_pulse += 1
        log(0.0, "beat-accent",
            f"anchor per section ({len(anchor_names)} anchors) + rotation "
            f"through {len(accent_pool)} props "
            f"(kick coverage {kick_coverage * 100:.0f}%)")
    # --- 2-pre-a2. SUBMODEL BEAT MOVEMENT -----------------------------------
    #
    # Step light THROUGH a prop's own submodel family on each pulse. Stepping a
    # spoke family in order reads as circular motion; a ring family reads as
    # radial in/out. Both are movement the prop was physically built for, and
    # both make the layout look far more intricate than flashing the whole
    # thing -- which is the ask: "we could add movement to those submodels with
    # the main beat, like a left right or up down or circular movement that
    # changes direction on the beat."
    #
    # Direction reverses every bar, so the motion breathes instead of spinning
    # forever in one direction.
    n_submodel = 0
    submodel_props: list[str] = []
    if SUBMODEL_FAMILIES:
        # Only props actually in the cast, so nothing lands on a model that
        # build_cast deliberately excluded (disabled, container, etc).
        cast_names = {t["name"] for t in cast} | {m for t in cast for m in t["members"]}
        cands = sorted(n for n in SUBMODEL_FAMILIES if n in cast_names)
        sm_rng = random.Random(f"{SONG_NAME}|submodel-beat")
        if cands:
            for si, section in enumerate(analysis.sections):
                if section.energy_level < 0.30:
                    continue
                # Held for the whole section --
                # same "let the viewer notice it" rule as the beat anchors.
                # MORE OF THE HOUSE MOVING. "I think I want more submodel
                # usage. The flakes and rosa grande and baby illusion and all
                # those props have tons of submodels, I'm sure we can use
                # more." Two to four props per section instead of one or two,
                # and each gets a DIFFERENT family so they aren't all doing the
                # same gesture.
                # More props per section. With 36 candidates and ~12 sections,
                # 2-4 per section gave each prop about ONE section for the
                # whole song -- which is why the EFL flakes measured 35-82
                # effects while Showstopper Spinner had 618 and only the busy
                # ones were noticeable. "I still don't really see any submodels
                # being used in the Snowflake EFL 36 props... the baby grand
                # illusion also has barely any submodel use."
                k = 4 + int(round(3 * section_intensity.get(si, 0.5)))
                props = [cands[(si * 3 + j) % len(cands)]
                         for j in range(min(k, len(cands)))]
                pal = section_meta[si].get("palette") or PALETTES["gold"]
                hits = pulse_times_in(section.start_time, section.end_time)
                if len(hits) < 4:
                    continue
                for prop in props:
                    fams = SUBMODEL_FAMILIES[prop]
                    # Prefer families whose geometry MEANS something when
                    # stepped: spokes/arms/petals read as rotation, rings and
                    # circles as radial in/out. Alphabetical order picked
                    # "Arrow" and "Stick" first, which are just arbitrary
                    # groupings on these props.
                    def _fam_rank(st: str) -> tuple:
                        low = st.lower()
                        if any(w in low for w in ("spoke", "arm", "petal", "whirl",
                                                  "spinner", "flower")):
                            return (0, st)          # rotation
                        if any(w in low for w in ("ring", "circle")):
                            return (1, st)          # radial in/out
                        return (2, st)
                    ordered_f = sorted(fams, key=_fam_rank)
                    stem = ordered_f[(si + props.index(prop)) % len(ordered_f)]
                    parts = fams[stem]
                    if prop not in submodel_props:
                        submodel_props.append(f"{prop}/{stem}")
                    step_ms = max(120, min(int(beat_ms * 0.8), 420))
                    for hi, ht in enumerate(hits):
                        # Reverse on the bar so it swings rather than spins.
                        bar_i = int((ht - section.start_time) / max(bar_s, 1e-6))
                        idx = hi % len(parts)
                        if bar_i % 2 == 1:
                            idx = len(parts) - 1 - idx
                        t_ms = int(ht * 1000)
                        # SHAPE, not just a pop. "Doing single strand chases
                        # and fans and radials and stuff like that on the
                        # submodels instead of the whole model can look cool
                        # because it traces a certain shape."
                        #
                        # The part is stepped with an effect chosen for what
                        # that part IS: spokes/arms/petals get a fan opening
                        # or a rotation, rings and circles get a radial burst,
                        # anything else a chase along its length. `on_pop`
                        # stays in the rotation so the plain hit is still
                        # available -- it is what makes the stepping read as
                        # rhythm rather than as continuous motion.
                        _low = parts[idx].lower()
                        # Widened with the ripple shapes measured in Krists's
                        # sequences: a six-point polygon imploding is a flake
                        # closing on its own geometry, and rays are the spoke
                        # version of the same idea. Both are things he puts on
                        # exactly these parts and this generator could not
                        # express at all.
                        if any(w in _low for w in ("spoke", "arm", "petal",
                                                   "whirl", "spinner", "flower")):
                            _shape = ["chase_mid_out", "pinwheel", "on_pop",
                                      "ripple_rays", "chase_bounce_pp"]
                        elif any(w in _low for w in ("ring", "circle")):
                            _shape = ["shockwave", "ripple_explode", "on_pop",
                                      "ripple_hex_in", "ripple_hex_out"]
                        else:
                            _shape = ["chase_lr", "chase_rl", "on_pop",
                                      "chase_spark", "ripple_hex_in"]
                        add(f"{prop}/{parts[idx]}", L_PULSE,
                            _shape[(hi + idx) % len(_shape)],
                            t_ms, t_ms + step_ms, pal)
                        n_submodel += 1
    print(f"Submodel beat movement: {n_submodel} steps through "
          f"{len(submodel_props)} families"
          + (" (" + ", ".join(submodel_props[:4]) + ")" if submodel_props else ""))

    # --- 2-pre-a3. LEAPING ARCHES -------------------------------------------
    #
    # "Big arches are really great for rhythmic movement if you just do a chase
    # back and forth, arches were originally often called 'leaping arches'
    # because it looks like the lights are leaping. Let's add that option so
    # they sometimes leap right and left, I'm guessing having all four of them
    # do it would be best, and only for beats that are simple enough that
    # people can follow and know which beat it's showing."
    #
    # One arch per beat, walking 1-2-3-4-3-2-1, so a full there-and-back takes
    # six beats and the eye can ride it. Deliberately on the QUARTER-NOTE grid
    # rather than the groove pulse: a leap you can't predict isn't a leap.
    # Gated to steady sections -- if the beat isn't simple, this reads as
    # scatter rather than motion.
    n_leap = 0
    arch_t = next((t for t in cast if t["name"] == "Arches"
                   and t["kind"] == "group" and t["n"] >= 3), None)
    if arch_t:
        arch_order = sorted((m for m in arch_t["members"] if m in positions),
                            key=lambda m: positions[m]["x"])
        if len(arch_order) >= 3:
            n_a = len(arch_order)
            # Reach the LAST arch, then turn around AT it. The previous cycle
            # [0,1,2,3,2,1] turned in mid-air: it never played arch 4 twice, so
            # the reversal appeared to start from the third arch. "They should
            # make it all the way to the end of the last arch and then bounce
            # back starting at the end of that last arch." Playing each
            # endpoint twice -- once outbound, once inbound -- puts the turn
            # where the eye expects it.
            # THREE LEAP MODES, rotated per section. "In addition to just the
            # single chase going back and forth across all four arches, could
            # we also add a mode that has 2 chases going back and forth on two
            # arches each, as well as a mode with 4 chases each just going back
            # and forth on a single arch? It would provide more variety."
            #
            # Each mode is a list of GROUPS, and every group runs its own
            # there-and-back cycle simultaneously:
            #   "sweep"  one chase over all four arches (the original)
            #   "pairs"  two chases, each over half the arches
            #   "solo"   one chase per arch, all bouncing at once
            # Playing each endpoint twice keeps the turn where the eye expects
            # it, exactly as the single sweep already does.
            def _cycle(idxs):
                return list(idxs) + list(reversed(idxs))

            half = max(1, n_a // 2)
            leap_modes = {
                "sweep": [_cycle(range(n_a))],
                "pairs": [_cycle(range(0, half)), _cycle(range(half, n_a))],
                "solo": [_cycle([i]) for i in range(n_a)],
            }
            _mode_names = ["sweep", "pairs", "solo"]
            _leap_mode_offset = random.Random(f"{SONG_NAME}|leapmode").randrange(3)
            cycle = list(range(n_a)) + list(range(n_a - 1, -1, -1))
            for si, section in enumerate(analysis.sections):
                # Only where the pulse is genuinely steady and the section has
                # some drive; and not every eligible section, so it stays a
                # gesture rather than a default.
                if si not in leap_sections:
                    continue
                sec_b = [b for b in beats
                         if section.start_time <= b < section.end_time]
                if len(sec_b) < 8:
                    continue
                pal = section_meta[si].get("palette") or PALETTES["gold"]
                hold = max(160, min(int(beat_ms * 0.9), 460))
                # Which mode this section uses. Deterministic per song and
                # section so a song moves through the three rather than
                # settling on one, and so re-running gives the same show.
                mode = _mode_names[(si + _leap_mode_offset) % len(_mode_names)]
                groups = leap_modes[mode]
                log(section.start_time, "leaping-arches",
                    f"mode={mode} ({len(groups)} simultaneous chase"
                    f"{'s' if len(groups) > 1 else ''})")
                for bi, bt in enumerate(sec_b):
                  for _g in groups:
                    m = arch_order[_g[bi % len(_g)]]
                    # Authoritative in its own sections. Anchors are already
                    # kept off the arches here (see leap_sections), so the
                    # remaining claims come from passes whose arch content this
                    # gesture is meant to replace -- deferring to them left
                    # Fara and Gooey with zero steps while Too Slow got 27.
                    beat_take(m, bt)
                    t_ms = int(bt * 1000)
                    # A CHASE ALONG the arch, not a flash OF the arch. "Leaping
                    # arches are where each arch has a chase from its start to
                    # its end, just like what the verticals and horizontals do
                    # for movement." The light travels the arc, and because
                    # consecutive arches fire in sequence the whole run reads
                    # as one leap moving across the house. Alternating
                    # direction on the turn-around keeps it leaping back.
                    # Direction is per GROUP now: each simultaneous chase
                    # turns around at its own end, so in "solo" mode all four
                    # arches bounce independently rather than sharing the
                    # single sweep's turn.
                    fwd = (bi % len(_g)) < max(1, len(_g) // 2)
                    add(m, L_PULSE, "chase_lr" if fwd else "chase_rl",
                        t_ms, t_ms + hold, pal)
                    n_leap += 1
    print(f"Leaping arches: {n_leap} steps ({len(_mode_names)} modes rotating)")

    # --- 2-pre-a4. LONG LINEAR PROPS FLASH TOO ------------------------------
    #
    # "The C9s only do movement but they could do the flashing and pulsing too
    # sometimes. They add a strong rhythmic effect because they cross the whole
    # house." Two earlier attempts failed: adding them to the anchor pool did
    # nothing (they rarely won a slot), and suppressing their movement made it
    # WORSE -- flashes fell from 33 to 14 -- because the branch I expected to
    # take over never fired. This is explicit instead of emergent: in flash
    # sections, the long gutter runs flash on the pulse, full stop.
    n_gutter = 0
    # (the beat anchor's contrast is measured after staging -- see below)
    # C9 GUTTER RUNS ONLY. `spread_x >= 40` also caught the window/matrix
    # HORIZONTALS, so they picked up beat flashes on top of their slow sway --
    # "the horizontals at the top and bottom of each matrix don't seem to do
    # smooth slow motion anymore, but are doing something more fast and erratic
    # now." Those props are meant to be the calm layer.
    # Long runs in EITHER direction. `spread_x >= 40` alone meant only the
    # horizontal gutter runs ever qualified -- the verticals are tall, not
    # wide, so they were structurally excluded and could only ever move:
    # "I only see the c9s and verticals/horizontals flashing or pulsing in
    # groove mode or only in a couple of songs, most songs I don't see it
    # happening and the c9s etc only do movement."
    #
    # The matrix HORIZONTALS stay out on purpose. They are the calm layer --
    # putting beat flashes on top of their slow sway is exactly what produced
    # "the horizontals don't seem to do smooth slow motion anymore, but are
    # doing something more fast and erratic now."
    gutter_props = [t for t in cast
                    if t["cat"] in ("single_line", "icicle")
                    and t["n"] >= 3
                    and (t.get("spread_x", 0) >= 40 or t.get("spread_y", 0) >= 40)
                    and "horizontal" not in t["name"].lower()]
    for si, section in enumerate(analysis.sections):
        if si not in ens_flash_sections or not gutter_props:
            continue
        pal = section_meta[si].get("palette") or PALETTES["gold"]
        hits = pulse_times_in(section.start_time, section.end_time)
        hold = max(150, min(int(beat_ms * 0.5), 320))
        for gi, tgt in enumerate(gutter_props):
            for hi, ht in enumerate(hits):
                # Alternate halves of the bar between gutters so two long runs
                # answer each other rather than firing as one slab -- EXCEPT in
                # the loudest sections, where they fire together.
                #
                # The beat lift on The WhatsApp Shakespeare fell from 1.352x to
                # 1.113x in its RHYTHMIC sections (not its quiet ones) as the
                # matrix pulsing was dialled back, the pulse skip added and big
                # props stopped flashing -- all things Krists asked for, and all
                # of which take brightness off the beat. "It could use more
                # impact for the main beat, I think we lost a bit of that."
                #
                # Rather than undo any of that, the beat goes where flashing is
                # still welcome: the long linear runs, hitting as one when the
                # section is loud enough to carry it.
                if section.energy_level < 0.60 and (hi + gi) % 2:
                    continue
                if not beat_free(tgt, ht):
                    continue
                beat_take(tgt, ht)
                t_ms = int(ht * 1000)
                # L_SIG, not L_PULSE. Measured in the .xsq: the C9s carry
                # continuous SingleStrand movement on the same layers the pulse
                # was landing on, and a movement effect starting 25ms after the
                # flash squeezed it to a single frame -- which is why three
                # attempts "worked" in the file and were invisible on the
                # house. The signature layer sits above all of it.
                add_target(tgt, L_SIG, "on_hit", t_ms, t_ms + hold, PALETTES["white"])
                n_gutter += 1
    print(f"Gutter flashes: {n_gutter} on {len(gutter_props)} long runs")

    print(f"Beat accent [{pulse_mode}]: {n_anchor} anchor flashes "
          f"(anchors: {', '.join(anchor_names[:6])}"
          f"{'...' if len(anchor_names) > 6 else ''}) "
          f"+ {n_pulse} rotating through {len(accent_pool)} props "
          f"({n_beat_skipped} overlaps suppressed) "
          f"(kick coverage {kick_coverage * 100:.0f}%)")
    print(f"Beat-locked brightness: {n_shader_pulse} wash spans pulse on the beat")
    print(f"Speed swing: {n_spd_swing} shader spans alternate fast/slow on the beat")

    # --- 2-pre-b. Whole-house shader as an underlying texture bed ----------
    # "We can also experiment with using shaders across the whole house,
    # especially as a lower layer that doesn't cover up the other existing
    # effects."
    #
    # Placed on L_BASE (layer 0) on the real 504-member "ALL Display" group
    # with Per Preview buffering, so the shader is computed across the whole
    # facade as ONE canvas rather than repeating per prop -- that's what
    # makes it read as a single texture the house is sitting inside.
    #
    # Two deliberate restraints, both because this underlays literally
    # everything and so is the single easiest way to wash the show out:
    #   * only in sections with real energy (>=0.5) -- quiet passages keep
    #     their darkness, which is where the show's dynamic range lives;
    #   * held 8 bars per change, far slower than any foreground effect, so
    #     it behaves like a slowly-breathing backdrop rather than competing
    #     for attention with the props on top of it.
    HOUSE_GROUP = "ALL Display"
    n_house = 0
    if SHADER_KEYS and any(g.name == HOUSE_GROUP for g in show_config.model_groups):
        house_rng = random.Random(f"{SONG_NAME}|house-shader")
        for si, section in enumerate(analysis.sections):
            # Gate the bed on measured LOUDNESS, not the coarse energy band.
            #
            # "The shaders somehow don't always seem to follow the intensity
            # changes, they often are on when the other lights go off, or when
            # all the lights come on suddenly the shader stays the same as
            # before so it's not as effective." Root cause: the bed keyed off
            # `section.energy_level` while everything else had moved to the
            # continuous loudness measure, so the two could disagree -- the
            # bed staying lit through a quiet passage is precisely the
            # "shader on when the other lights go off" case. Now it uses the
            # same signal as the activity budget, so they move together.
            loud = section_loudness.get(si, 1.0)
            if loud < 0.45:
                continue
            pal = section_meta[si].get("palette") or PALETTES["ice"]
            # Never start the bed before the kick enters. Measured: on The
            # WhatsApp Shakespeare the bed began at 6.97s while the kick
            # arrives at 8.82s -- 1.85s of full-house shader ALREADY RUNNING
            # before the moment that's supposed to be the song kicking into
            # gear. "The shaders suddenly come on the wrong beat too early,
            # and makes the main beat drop when the kick drum kicks in less
            # impactful." The bed is the single largest visual element, so it
            # arriving early is the one thing most able to spoil that entrance.
            bed_from = section.start_time
            if first_kick_time is not None:
                bed_from = max(bed_from, first_kick_time)
            # The bed swaps its shader on harmonic movement too, just far more
            # slowly than a foreground canvas (every 8th eligible mark).
            marks = [d for d in change_marks(section, 8) if d >= bed_from - 0.05]
            if not marks:
                continue
            if marks[-1] < section.end_time - 0.2:
                marks.append(section.end_time)
            # STOCK SHADERS ONLY for the house-wide bed. The bed spans every
            # prop at once, so whatever bias a shader has in coverage or
            # palette response is multiplied across the entire facade. Adding
            # the four new high-coverage shaders to this pool took 12 to 12
            # from a 14% red peak to 46% -- over the human ceiling -- in one
            # build. They stay available for individual large canvases, where
            # the same bias is contained to one prop.
            key = house_rng.choice(HOUSE_BED_SHADERS or SHADER_KEYS)
            # Slow the shader WAY down for the house-wide bed.
            #
            # "The whole house shaders look great, except they move too fast
            # most of the time. When shaders move quickly across the whole
            # house it makes individual props look like they're strobing."
            # That's a scale effect, not a speed-value problem: the same
            # shader speed that reads as gentle motion inside one matrix is
            # sweeping across ~50 feet of facade here, so any given prop sees
            # the pattern rush past it. Overriding Shader_Speed on a copy of
            # the registered settings keeps the per-song parameter variation
            # from round 15 while fixing the pace.
            slow_key = f"{key}__housebed"
            if slow_key not in EFFECTS:
                ename, esettings = EFFECTS[key]
                slowed = dict(esettings)
                # 0.12, down from 0.22 -- the house bed spans ~50 feet, so
                # apparent speed at any single prop is far higher than the
                # same value inside one matrix. Reported still too fast on
                # two separate songs after the first reduction.
                slowed["E_SLIDER_Shader_Speed"] = str(
                    max(60, int(int(slowed.get("E_SLIDER_Shader_Speed", 400)) * 0.12)))
                slowed["X_AmbientSlowed"] = "1"
                EFFECTS[slow_key] = (ename, slowed)
            for j in range(len(marks) - 1):
                s_ms, e_ms = int(marks[j] * 1000), int(marks[j + 1] * 1000)
                # The bed breathes on the beat too, but shallowly (depth 0.4).
                # It is deliberately the slowest-moving thing in the show, and
                # a slow shader with no beat relationship at all is exactly
                # what "the shaders are not tied to the beat" describes at the
                # largest scale. A gentle house-wide swell on each beat gives
                # the whole facade a pulse under everything else -- pumping it
                # any harder would read as the house strobing.
                # Inside a drop/fade cycle the bed steps aside. It spans EVERY
                # prop at once, so it -- not the section staging -- is what
                # holds the floor up while a fade is supposed to be decaying:
                # with the cycle holds and the cast cut back, the display
                # still only fell to 40-62 between kicks. The cycle holds are
                # the bed for that passage, and they carry the decay envelope.
                if any(c[0] - 0.3 <= s_ms / 1000.0 < c[1] for c in drop_cycles):
                    continue
                _bed_pal = beat_pulse_palette(pal, beats, s_ms, e_ms, 0.4)
                add(HOUSE_GROUP, L_BASE, slow_key, s_ms, e_ms, _bed_pal, SPAN)
                n_house += 1
                # The house bed is ONE spanning effect across every prop, so
                # the depth-oriented ones get a single column of it and light
                # as a slab. A same-span copy on each of those props, rendered
                # per-model, overrides the group there and draws the shader
                # properly. This is the "whole house effects render correctly
                # on those props as well" fix.
    print(f"Whole-house shader bed: {n_house} spans")

    # --- 2-pre-b2. FLOODS: sparing, and only for what floods are good at -----
    #
    # "The floods do a slow pulse too often, seems like in every song. Floods
    # are really for when lights are off in a section of the house then those
    # floods could be on to show some contrast, or if they're doing a beat then
    # it looks better to do upper floods then lower floods, back and forth,
    # like lowers on beats 1 and 3 and uppers on beats 2 and 4. Or floods on
    # the left of the house and floods on the right. Floods should be used more
    # sparingly."
    #
    # Floods are no longer instrument carriers at all (see the voice-candidate
    # filter). They now do exactly two things, in a minority of sections:
    #
    #   CONTRAST -- when the rest of the house is quiet, floods wash the dark
    #     facade. This is the use Krists describes first and it only makes
    #     sense when there IS darkness to fill, so it is gated on the section's
    #     own activity level rather than fired on a schedule.
    #   ALTERNATION -- lower floods on beats 1 and 3, upper floods on 2 and 4
    #     (or left/right), which reads as a big slow rocking pulse across the
    #     whole house rather than one indistinct bloom.
    #
    # The upper/lower and left/right sets are derived from the layout's own
    # WorldPos coordinates -- no new xLights groups needed. Measured split:
    # upper floods sit at y~433-456, lower at y~53-81, so the gap is enormous
    # and the threshold is not delicate.
    # Only ENABLED floods: build_cast already dropped Active=0 models, so the
    # cast's own Floods group is the authority. (Flood-5 is disabled, and the
    # group also carries two non-floods -- Black Santa, Gingerbread Man --
    # which the name filter excludes.)
    _enabled = {m for t in cast for m in t["members"]} | {t["name"] for t in cast}
    flood_models = sorted(m for m in _enabled
                          if m.lower().startswith("flood") and m in positions)
    n_flood = 0
    if flood_models:
        ys = sorted(positions[m].get("y", 0) for m in flood_models)
        y_split = (ys[0] + ys[-1]) / 2
        upper = [m for m in flood_models if positions[m].get("y", 0) >= y_split]
        lower = [m for m in flood_models if positions[m].get("y", 0) < y_split]
        xs = sorted(positions[m].get("x", 0) for m in flood_models)
        x_split = (xs[0] + xs[-1]) / 2
        left = [m for m in flood_models if positions[m].get("x", 0) < x_split]
        right = [m for m in flood_models if positions[m].get("x", 0) >= x_split]
        flood_rng = random.Random(f"{SONG_NAME}|floods")
        for si, section in enumerate(analysis.sections):
            if section.end_time - section.start_time < bar_s * 2:
                continue
            quiet = section_loudness.get(si, 1.0) < 0.5 or section.energy_level < 0.35
            if quiet:
                # CONTRAST: one long soft wash over the whole quiet stretch.
                # No rhythm at all -- the point is presence while the detailed
                # props are dark.
                for m in flood_models:
                    add(m, L_PULSE, "wash", section.start_time_ms,
                        section.end_time_ms,
                        section_meta[si].get("palette") or PALETTES["ice"])
                    n_flood += 1
                log(section.start_time, "floods",
                    f"contrast wash across {len(flood_models)} floods (quiet section)")
                continue
            # ALTERNATION, and only in a MINORITY of loud sections -- this is
            # the "used more sparingly" half of the note. One section in three.
            if si % 3 != 0:
                continue
            pair = ((lower, upper) if flood_rng.random() < 0.6 else (left, right))
            if not pair[0] or not pair[1]:
                continue
            pal = section_meta[si].get("palette") or PALETTES["gold"]
            hold = int(min(beat_ms * 0.9, 700))
            sec_bt = [b for b in beats
                      if section.start_time <= b < section.end_time]
            # HALF-NOTES, NOT BEATS, and one palette held across the section.
            # "There's a point where the floodlights are really strobing while
            # changing colors... I don't want strobing like that, not so fast
            # and not with anything big, it can give people migraines." Floods
            # wash the entire facade, so alternating them every beat is ~1.7
            # alternations per second across the whole house -- the worst
            # possible prop to do that on. Every other beat reads as a rock.
            hold = int(min(beat_ms * 1.7, 1400))
            for bi, bt in enumerate(sec_bt[::2]):
                grp = pair[bi % 2]        # bar 1&3 -> one side, 2&4 -> other
                t_ms = int(bt * 1000)
                for m in grp:
                    add(m, L_PULSE, "wash", t_ms, t_ms + hold, pal)
                    n_flood += 1
            log(section.start_time, "floods",
                f"alternating {len(pair[0])}/{len(pair[1])} floods on the beat")
    # --- 2-pre-b3. SILENT INTRO: minimal, but never near-dark ---------------
    #
    # Measured on ENIGMA v2: the first 27.6 seconds rendered at 1.1% lit --
    # effectively black -- because every rhythm pass keys off instrument
    # onsets and this intro has literally zero onsets on all six voices, and
    # the whole-house bed is held until the first kick. Krists: "Nearly dark is
    # probably too little for that whole intro, but there's no drums and it's
    # an intro, so just something dramatic. Maybe a single whole house shader,
    # or something with floods and most props off, or just outlines... The
    # first sound of that intro is a big drum drop sort of thing though, so the
    # very first sound does need something bright."
    #
    # Generalises to any song opening with atmosphere before the drums.
    n_intro = 0
    first_onset = min((v.onsets[0] for v in voices.values() if len(v)),
                      default=None)
    if first_onset is not None and first_onset > 6.0:
        intro_pal = PALETTES["ice"]
        # (a) the very first sound gets a real hit -- this is a moment, not a
        #     fade-in, and it sets up the emptiness that follows.
        y0 = 0.0
        for t in cast:
            if prominence_of(t) >= 2:
                add_target(t, L_SIG, "on_hit", int(y0 * 1000), int(y0 * 1000) + 320,
                           PALETTES["white"])
                add_target(t, L_SIG, "wash", int(y0 * 1000) + 320, int(y0 * 1000) + 2600,
                           intro_pal,
                           extra={"T_TEXTCTRL_Fadein": "0.0",
                                  "T_TEXTCTRL_Fadeout": "2.2"})
                n_intro += 2
        # (b) then a single slow whole-house shader plus outlines only -- the
        #     "minimal but not dark" bed, holding until the drums arrive.
        bed_from = 2.8
        if HOUSE_GROUP and bed_from < first_onset - 1.0:
            key = SHADER_KEYS[0] if SHADER_KEYS else "plasma_organic"
            slow_key = f"{key}__introbed"
            if slow_key not in EFFECTS:
                ename, es = EFFECTS[key]
                sl = dict(es)
                sl["E_SLIDER_Shader_Speed"] = str(
                    max(20, int(int(sl.get("E_SLIDER_Shader_Speed", 400)) * 0.10)))
                EFFECTS[slow_key] = (ename, sl)
            add(HOUSE_GROUP, L_BASE, slow_key, int(bed_from * 1000),
                int(first_onset * 1000), intro_pal, SPAN)
            n_intro += 1
        for t in cast:
            if t["cat"] in ("single_line", "icicle") and t["n"] >= 2:
                add_target(t, L_MOTION, "chase_mid_out", int(bed_from * 1000),
                           int(first_onset * 1000), intro_pal)
                n_intro += 1
        log(0.0, "intro", f"first-sound hit + minimal bed to {first_onset:.1f}s")
    print(f"Silent intro: {n_intro} placements "
          f"(first instrument onset at {first_onset:.1f}s)"
          if first_onset is not None else "Silent intro: n/a")

    print(f"Floods: {n_flood} placements "
          f"({len(flood_models)} floods, upper/lower and left/right sets from layout)")

    # --- 2-pre-c. Unison rhythmic movement across the linear props ---------
    # "In previous revisions there was a lot of movement in the verticals and
    # horizontals and gutters/c9s which I liked that seems to be missing.
    # Having many of the verticals or horizontals or c9s or window outlines
    # doing the SAME rhythmic movement can be super effective visually."
    #
    # This is deliberately NOT the same thing as the per-prop voice system
    # above: there, one prop = one instrument, for attribution. Here, MANY
    # props share ONE instrument and move identically, for scale. Both are
    # legible; they just do different jobs, and the unison version is the one
    # that makes the whole roofline move as a single object.
    #
    # Driven by the slowest-moving available voice (chord, then snare, then
    # kick) precisely because it has to stay at quarter/eighth pace -- "they're
    # too random and stroby on anything faster."
    # Drive the ensemble from the song's RHYTHMIC DRIVER, not its harmony.
    #
    # Was chord > snare > kick, i.e. the most melodic option first. Direct
    # feedback: "the vertical and horizontal movements... seem to change
    # directions on somewhat random seeming beats, I'm not sure what voices
    # they're choosing... anything on off beats or on the melody seems to just
    # get lost and seems random. I want to put more attention on like the kick
    # drum, snare, and any obvious synth stabs or piano chord stabs."
    # A chord line lands wherever the harmony moves, which is frequently
    # off the felt pulse -- exactly the "random seeming beats" description.
    # Percussion first now, harmony only as a last resort.
    ensemble_voice = next((v for v in ("kick", "snare", "chord") if v in voices), None)
    n_unison = 0
    if ensemble_voice:
        # Prefer the props this gesture is actually ABOUT -- the window/matrix
        # outline verticals and horizontals, the gutter C9s, the icicles.
        #
        # A plain `[:10]` over cast order was silently excluding all of them:
        # the first ten single_line/icicle entries were four single-member
        # `Sun Room Window Matrix/*` submodels (n=1, so they can't chase at
        # all) plus the big GE decorated-prop spoke families, while the real
        # "Window Verticals", "Corner Verticals", "Other Verticals" and
        # "Right Verticals" sat at indices 11-21 and never made the cut. That
        # is exactly "I don't see any of the movement in the window and matrix
        # outline verticals and horizontals like we had before."
        # (ens_targets is selected before PASS 1 -- see movement_props there.)
        ev = voices[ensemble_voice]
        for si, section in enumerate(analysis.sections):
            if section.energy_level < 0.45 or not ens_targets:
                continue
            # In a genuinely dancey section the linear props do BOTH: they
            # keep the chase AND flash the beat. Suppressing movement wherever
            # they flash cut unison placements from 920 to 240 on Gooey, which
            # throws away the movement Krists had just said was working
            # ("they're still only doing movement but the movement is
            # better"). Only the mid-energy alternating sections trade one for
            # the other.
            if si in ens_flash_sections and section.energy_level < 0.65:
                continue
            # RUN ON THE BEAT GRID, not on the driver's raw hit times.
            #
            # Driving off kick/snare onsets makes every chase span the gap to
            # the NEXT onset, and those gaps are syncopated: measured on Gooey,
            # Corner Verticals got chases of 350ms, then 1050ms, then 1400ms,
            # back to back. A chase whose speed changes every repetition cannot
            # be followed, which is the whole complaint -- "having a single
            # chase should work best, it can also often follow the main beat
            # and you can follow the chase clearly."
            #
            # So: one chase per half note, every one the same length, starting
            # exactly on a beat. The driver voice still decides WHETHER the
            # ensemble plays in this section (it only moves where the rhythm
            # section is actually playing), but not the individual timings.
            sec_beats_here = [b for b in beats
                              if section.start_time <= b < section.end_time]
            hits = ev.in_window(section.start_time, section.end_time)
            if len(hits) < 4 or len(sec_beats_here) < 4:
                continue
            spaced = sec_beats_here[::2]          # half notes
            pal = section_meta[si].get("palette") or PALETTES["ice"]
            for i, t in enumerate(spaced):
                nxt = spaced[i + 1] if i + 1 < len(spaced) else None
                # Uniform span by construction now -- half notes are evenly
                # spaced, so every chase crosses at the same speed and the eye
                # can lock onto it.
                span = min((nxt - t) if nxt else 2 * beat_ms / 1000.0, bar_s)
                if span < VOICE_MOTION_MIN_GAP_S:
                    continue
                # Long strips traverse in the same time as short ones once the
                # buffer is per-model, so they LOOK faster. "The three smaller
                # window matrices have the right speed... but the bigger
                # matrices like the sun room, garage, and fence, the movement
                # is too fast." Doubling the span on the (Large) split halves
                # the apparent speed, matching what the small ones already do.
                big_span = span * 2.0
                t_ms = int(t * 1000)
                # Every prop moves on the SAME beat in the SAME phase -- that
                # simultaneity is what makes the ensemble read as one gesture.
                # But each prop moves along ITS OWN axis: a vertical strip
                # travels up/down, a gutter run travels along its length.
                #
                # This pass previously hardcoded `chase_dots_lr/rl` for every
                # member regardless of orientation, and since it is the main
                # source of movement on the verticals and outlines, that is
                # why they still only ever moved side to side -- the
                # orientation logic added last round lived in the voice path,
                # which these props mostly don't go through.
                # Direction from the BEAT NUMBER, so every prop reverses on
                # the same musical instant and the whole facade sways
                # together. Half-note flips (`every=2`) -- a reversal on every
                # single quarter note reads as agitation rather than sway.
                # Direction flips every HALF BAR (2 beats), so the reversal
                # itself lands on a musical boundary you can hear coming --
                # "we can also change the direction of the c9 movements on the
                # half notes (every half bar) for example."
                ph = beat_phase(t, every=2)
                for tgt in ens_targets:
                    _sp = big_span if "(Large)" in tgt["name"] else span
                    mot = line_motion_for(tgt, seed=f"{si}")
                    # A FRONT-TO-BACK STRAND CHASES THE OTHER WAY ROUND.
                    # Its own buffer runs along Z, so "left to right" in model
                    # space reads as back-to-front from the street -- opposite
                    # to every strand that runs across the facade. Krists:
                    # "single line 13 and 14 have movement going the opposite
                    # way of the other single lines." This is a rendering
                    # consequence of the geometry, not a layout mistake, so it
                    # is corrected here rather than by flipping the models.
                    if (tgt["name"] in DEPTH_ORIENTED
                            or all(m in DEPTH_ORIENTED for m in tgt["members"])):
                        mot = (REVERSE.get(mot[0], mot[0]),
                               REVERSE.get(mot[1], mot[1]))
                    add_target(tgt, L_MOTION, mot[ph], t_ms,
                               t_ms + int(_sp * 1000), pal)
                    n_unison += 1
        if n_unison:
            log(0.0, "unison-lines",
                f"{ensemble_voice} drives {len(ens_targets)} linear props in unison")
    print(f"Unison line movement: {n_unison} placements"
          + (f" on {ensemble_voice}" if ensemble_voice else ""))

    # --- The kick's actual entrance: an explicit "gear-shift" accent -----
    # Independent of section boundaries (see the `first_kick_time` comment
    # above) -- a sharp hit across a real subset of the cast right at the
    # moment the kick enters, so "suddenly jumps into gear" reads as a
    # single clear event even when the nearest section label starts a few
    # seconds earlier or later than the kick actually does.
    if first_kick_time is not None:
        t_ms = int(first_kick_time * 1000)
        gear_targets = [t for t in cast if t["n"] >= 3][:24] or cast[:24]
        for i, t in enumerate(gear_targets):
            add_target(t, L_SIG, "on_hit", t_ms, t_ms + 300, PALETTES["white"])
        log(first_kick_time, "kick-enters",
            f"gear-shift accent across {len(gear_targets)} targets")

    def target_name(t, idx=0):
        """A single addressable name for one-off accents. Real groups/models
        can be targeted directly; synthetic (non-xLights) targets have no
        valid name of their own, so rotate through their real members."""
        if t.get("synthetic"):
            return t["members"][idx % len(t["members"])]
        return t["name"]

    def display_sweep(order, start_ms, end_ms, palette, key="on_swell", max_t=26):
        seq = order
        if len(seq) > max_t:
            step = len(seq) / max_t
            seq = [seq[int(i * step)] for i in range(max_t)]
        n = len(seq)
        if n < 2:
            return
        # A whole-display sweep is a fast directional gesture that follows the
        # synth zips, not the chords -- the opposite of what a smooth passage
        # wants. Measured on This Version of You, this pass alone put 420
        # off-chord placements into a piano intro that should carry nothing
        # but chord swells.
        if in_ambient(start_ms / 1000.0):
            return
        span = max(end_ms - start_ms, n * 25)
        hold = max(160, int(span / n * 2.2))
        for k, t in enumerate(seq):
            s = start_ms + int(span * k / (n - 1))
            _tn = target_name(t, k)
            # A PROP WITH GEOMETRY BURSTS AS THE WAVE REACHES IT, rather than
            # simply lighting. This pass is 37% of every flat `On` on the
            # flakes and spinners, and the sweep is the same gesture either
            # way -- one prop after another down the facade -- but a flake
            # that opens outward as the wave arrives shows the flake, where a
            # flat fill just proves it is switched on. Krists's own sequences
            # put Shockwave on those props 45-64% of the time against On at
            # 10-17%; this generator was the other way round.
            _k = key
            if key in ("on_swell", "on_pop", "on_hit") and any(
                    h in _tn.lower() for h in ("flake", "snowflake", "spinner",
                                               "rosa", "grand illusion",
                                               "showstopper", "sphere")):
                _k = random.Random(f"sweep|{_tn}").choice(
                    ["shockwave", "ripple_hex_out", "shockwave", key])
            add(_tn, L_SIG, _k, s, s + hold, palette)

    # --- 2a. Fast upward synth zips -> quick sweep up the whole display ---
    fast = sorted(cues.get("fast_risers", []), key=lambda r: -r["rise_ratio"])
    strong_fast = [r for r in fast if r["rise_ratio"] >= 2.0][:40]
    strong_fast.sort(key=lambda r: r["start_time"])
    for r in strong_fast:
        s_ms = int(r["start_time"] * 1000)
        e_ms = int(r["end_time"] * 1000)
        display_sweep(cast_by_y, s_ms, e_ms, PALETTES["white"], key="on_pop")
        log(r["start_time"], "fast-sweep-up",
            f"upward zip across display ({r['duration']:.2f}s, x{r['rise_ratio']:.1f})")
    print(f"Fast upward sweeps: {len(strong_fast)} (from {len(fast)} candidates)")

    # --- 2b. Long risers -> slower directional builds ---
    long_risers = [r for r in cues.get("mix_risers", [])
                   if r["rise_ratio"] >= 1.7 and r["duration"] >= 1.2]
    for r in long_risers[:24]:
        s_ms, e_ms = int(r["start_time"] * 1000), int(r["end_time"] * 1000)
        order = cast_by_y if r.get("kind") == "tonal" else cast_by_x
        display_sweep(order, s_ms, e_ms, PALETTES["ice"], key="on_swell", max_t=30)
        if in_ambient(r["start_time"]):
            continue          # a build inside a smooth passage fights the chords
        log(r["start_time"], "riser",
            f"{'vertical' if r.get('kind')=='tonal' else 'horizontal'} build ({r['duration']:.1f}s)")
    print(f"Long riser builds: {min(len(long_risers), 24)}")

    # --- 2c. Synth stabs -> fast pop on a rotating subset ---
    # Drop any stab that doesn't land near the beat/half-beat grid -- direct
    # feedback: an isolated flash "tied to some random one-off transient"
    # with no rhythm to it looks glitchy rather than musical, and should be
    # removed rather than shown.
    stabs = [s for s in cues.get("synth_stabs", []) if near_beat_grid(s["time"])]
    n_stabs_raw = len(cues.get("synth_stabs", []))
    # Symmetric ensembles (window matrix trio, garage curtain panels) are
    # excluded -- rotating through their individual members here is exactly
    # what broke the symmetry the ambient layer otherwise maintains.
    # `wants_motion` targets excluded for the same reason PULSE role is: a
    # brief isolated pop is "hard on the eyes" on a huge canvas and just
    # "flash and color fade" on a big decorative group -- this pool (matrix +
    # custom cats) was `large`-only and caught Garage Roof/Sun Room Window
    # Matrix directly (fixed round 10), then Candy Canes (custom, n=8, not
    # large) directly (fixed this round) -- both times because "matrix" and
    # "custom" as raw categories don't distinguish "huge canvas"/"decorative
    # group deserving real motion" from "small prop a brief pop suits fine".
    stab_pool = [t for t in (by_cat.get("matrix", []) + by_cat.get("custom", []))
                 if not t.get("symmetric") and not wants_motion(t)
                 and t["name"] not in prop_voice] or \
                [t for t in cast if not t.get("symmetric") and not wants_motion(t)
                 and t["name"] not in prop_voice]
    for i, s in enumerate(stabs):
        t_ms = int(s["time"] * 1000)
        picks = [stab_pool[(i * 3 + k) % len(stab_pool)] for k in range(3)]
        for t in picks:
            add(target_name(t, i), L_SIG, "on_pop", t_ms, t_ms + 240, PALETTES["violet"])
    if stabs:
        log(stabs[0]["time"], "synth-stabs",
            f"{len(stabs)} filter-open stabs -> fast pops on rotating props")
    print(f"Synth stabs: {len(stabs)} (of {n_stabs_raw}, off-rhythm ones dropped)")

    # --- 2d. Pre-drop implosion into a drop hit ---
    # Only the strongest drops get the whole-display treatment; doing it at
    # every detected cutout makes the biggest gesture in the show routine.
    climax_drop_rng = random.Random(f"{SONG_NAME}|climax-drop")

    _climax_shared_key = climax_drop_rng.choice(SAFE_LARGE)

    def drop_hit_key(t, is_climax=False):
        """Vary the drop payload by shape instead of a flat On-burst
        everywhere -- direct feedback wanted the peak moment to show
        "complex submodels and such", not just uniform brightness.

        At the song's ONE real climax (is_climax=True), large canvases draw
        from the full shader-inclusive pool instead of the plain pinwheel/
        spirals binary -- direct feedback: "introduce more contrast and more
        textures with the shaders... the biggest drop should be the climax
        visually too." Deliberately NOT the same shared-key-for-cohesion
        used for ordinary sections: the climax is exactly the one moment
        where every large canvas showing something DIFFERENT reads as rich
        rather than disjointed, since it's a single ~2s burst, not an
        ongoing section the eye has time to notice as inconsistent."""
        if t.get("large"):
            if is_climax:
                # ONE shader across the canvases, not one each. The climax was
                # deliberately given per-canvas variety on the theory that a
                # ~2s burst is too brief to read as inconsistent. On a quiet
                # song it is not: "at :51 all the matrices have different
                # effects which looks a bit odd." Measured there -- Spirograph
                # Spectrum, Rain on Glass, Black Cherry Cosmos, Liquid Chrome
                # and Stained Glass, one per matrix, all at once. The matrices
                # are one surface and should agree.
                return _climax_shared_key
            return "pinwheel" if t["cat"] == "tree" else "spirals"
        if t["cat"] == "arch":
            return "chase_mid_out"
        if t["cat"] == "matrix":
            return SHADER_KEYS[0] if SHADER_KEYS else "plasma_organic"
        if t["cat"] == "single_line":
            return "chase_bounce"
        if t["cat"] == "icicle":
            return "chase_mid_out"
        return "shockwave"

    # Drops come from TWO detectors, because they are two different musical
    # gestures and each is blind to the other:
    #   * predrop_cutouts -- the build goes near-silent, then slams. Classic
    #     EDM. Gives a real cutout window to hold black through.
    #   * energy_drops    -- the level simply STEPS UP and stays up, with no
    #     silence at all (detect_energy_drops in detect_cues.py).
    # Relying on cutouts alone meant songs that don't use the silence
    # convention got no drop staging whatsoever: World Is Mine has zero
    # cutouts, so its two biggest moments -- which Krists identified by ear at
    # 1:35 and 3:39, and which measure as 37%->101% and 42%->73% of peak -- had
    # nothing staged on them at all. "They don't look like drops at all."
    #
    # An energy drop has no natural dark window, so one is manufactured: a
    # short run-up dark of about a beat before the hit. Deliberately much
    # shorter than a cutout blackout, because the music here is NOT quiet --
    # holding the house black for two seconds over a loud build fights the
    # song instead of setting up the payoff.
    cutouts = list(cues.get("predrop_cutouts", []))
    # INSTRUMENT DROPOUTS come first and outrank the other two detectors: a
    # bass-only hole before the drop is the strongest set-up gesture there is,
    # and it carries a REAL dark window (the length of the hole) rather than a
    # manufactured one. "From 2:48 to 2:53 every instrument drops out and it's
    # just the bass, and then the big drop is at 2:53, so right now we don't
    # have that impact."
    n_dropouts = 0

    def _snap_drop(t: float) -> float:
        """Pull a detected drop onto the nearest BAR line, then the nearest
        beat. Band-energy timing is inherently a little loose -- measured on
        ENIGMA the real drop came back at 173.55s while the downbeat is at
        173.20s, half a beat late, and a drop that lands half a beat off reads
        as sloppy no matter how big it is: "the timing seems just a bit off
        with the double drop, it's not quite on the drop, especially the second
        time." The energy-drop path already did this; the dropout path did not.
        """
        best = t
        if downbeats:
            d = min(downbeats, key=lambda x: abs(x - t))
            if abs(d - t) <= beat_ms / 1000.0 * 0.75:
                return d
        if beats:
            b = min(beats, key=lambda x: abs(x - t))
            if abs(b - t) <= beat_ms / 1000.0 * 0.5:
                best = b
        return best

    for ev in cues.get("instrument_dropouts", []):
        _fd = ev.get("false_drop")
        cutouts.append({
            "cutout_start": float(ev["dropout_start"]),
            "cutout_end": float(ev["dropout_end"]),
            "drop_time": _snap_drop(float(ev["drop_time"])),
            # Ranked at the top so the 6-slot cap never drops one of these in
            # favour of an ordinary cutout.
            "drop_energy": 1.5,
            "dropout": True,
            "false_drop": (_snap_drop(float(_fd)) if _fd else None),
        })
        n_dropouts += 1
    n_energy_drops = 0
    MINOR_DROPS: list[float] = []
    for ev in cues.get("energy_drops", []):
        dt = float(ev["drop_time"])
        # Snap to the nearest downbeat within half a beat -- a drop lands on
        # a bar line, and RMS-window timing can sit slightly off it.
        if downbeats:
            near = min(downbeats, key=lambda d: abs(d - dt))
            if abs(near - dt) <= beat_ms / 2000.0:
                dt = near
        # Skip anything the cutout detector already staged.
        if any(abs(dt - c["drop_time"]) < 6.0 for c in cutouts):
            continue
        # A MINOR DROP GETS A PROPORTIONATE RESPONSE, not the full staging.
        # "Some smaller drops that dip only a little and aren't necessarily big
        # drops but are still important moments to accentuate." Staging one
        # like a major -- two beats of darkness, a full-house hit -- would make
        # the song lurch every twenty seconds and steal from the real drops.
        # It gets a shorter breath and takes its place as an anchor, so the
        # look changes ON it, which is what makes a moment read as intended.
        _minor = ev.get("tier") == "minor"
        if _minor:
            MINOR_DROPS.append(dt)
            continue
        dark = min(beat_ms / 1000.0, 0.6)
        cutouts.append({
            "cutout_start": dt - 2 * beat_ms / 1000.0,
            "cutout_end": dt - dark,
            "drop_time": dt,
            "drop_energy": float(ev.get("drop_energy", 0.8)),
            "energy_drop": True,
        })
        n_energy_drops += 1
    # A cutout that an ENERGY DROP also fired on is a real drop, and outranks
    # a louder-but-uncorroborated one for the six staged slots. Too Slow's
    # first drop at 0:30 measured a 4.59x sustained level lift -- unmistakable
    # -- but its cutout carried drop_energy 0.61 and lost the cap to weaker
    # candidates, so the biggest moment in the first minute got no staging at
    # all: "the first drop at :30 is not picked up, there is no impact."
    for _c in cutouts:
        for _e in cues.get("energy_drops", []):
            if abs(float(_e["drop_time"]) - _c["drop_time"]) < 3.0:
                _c["drop_energy"] = _c.get("drop_energy", 0) + 0.5 * float(
                    _e.get("lift", 1.0))
                break
    # A DROPOUT OWNS ITS WHOLE MOMENT. The two-stage return spans the hole,
    # the fake drop and the real drop; another event staged anywhere inside
    # that span fires between them and wrecks the gesture. Measured on ENIGMA:
    # the corroboration boost above promoted a cutout to 172s, one second
    # before the real drop at 173.5s, which is exactly "it seems to drop
    # between when the drums start and when everything else comes in."
    _drop_spans = [(c["cutout_start"] - 1.0, c["drop_time"] + 2.0)
                   for c in cutouts if c.get("dropout")]
    cutouts = [c for c in cutouts
               if c.get("dropout")
               or not any(a <= c["drop_time"] <= b for a, b in _drop_spans)]
    # ...and they always keep a staged slot, whatever the energy ranking says.
    _drops = [c for c in cutouts if c.get("dropout")]
    _rest = sorted((c for c in cutouts if not c.get("dropout")),
                   key=lambda ev: -ev.get("drop_energy", 0))
    # A CYCLE START IS NOT THE MAIN DROP. In a drop-and-fade passage every
    # cycle begins with a big hit, so the detectors see each one as a drop --
    # but only the LAST one is where the track actually opens up:
    #
    #   "The main drop after the intro where the full track starts is right at
    #    1:00, there is a false drop before that where bass starts but then it
    #    should cut out before the main big drop."
    #
    # Measured on Gooey: candidates at 48.95 and 59.93, both staged as full
    # implosions. 48.95 is the fifth cycle's opening kick -- the bass entering
    # over the fade -- and 59.99 is where the kick becomes continuous. Demoted
    # here rather than dropped, so the cycle still gets its hit; it just does
    # not get the whole-display blackout-and-slam reserved for a real drop.
    if drop_cycles:
        _cycle_starts = [c[0] for c in drop_cycles]
        _main = drop_cycles[-1][1]
        _demoted = [ev for ev in (_drops + _rest)
                    if any(abs(ev["drop_time"] - cs) < 1.5 for cs in _cycle_starts)
                    and abs(ev["drop_time"] - _main) >= 1.5]
        if _demoted:
            print(f"  false drops (cycle starts, not the main drop): "
                  f"{', '.join(f'{ev['drop_time']:.1f}s' for ev in _demoted)}")
        _keep = [ev for ev in _drops if ev not in _demoted]
        _rest = [ev for ev in _rest if ev not in _demoted]
        _drops = _keep
    # SNAP EVERY DROP TO THE HIT IT IS ABOUT.
    #
    # "The first drop at 1:23 is off the beat, the lights come in too early...
    # At 2:46 there's also a drop and the lights seem to hit too early... And
    # at 4:48 it's close but still a bit off."
    #
    # The energy detector reports where the level STARTS RISING, which is
    # before the downbeat that actually lands -- measured on The Last Goodbye,
    # 1.51s early at 1:23, 0.56s at 2:46 and 0.07s at 4:48, in exactly the
    # order Krists ranked them. A drop is a single moment and the lights have
    # to be on it, so each candidate is pulled forward to the strongest drum
    # onset just after it and then quantised to the beat grid.
    def _snap_drop(dt: float) -> float:
        # The FIRST strong hit in the window, not the strongest one. Taking
        # the strongest pushed The Last Goodbye's 4:48 drop out to 289.63s --
        # a louder bar a second and a half after the drums actually entered at
        # 288.07 -- which is the same error as being early, mirrored. A drop
        # is where the thing STARTS.
        # Widen ONLY if the near window is empty. 2.0s is right whenever the
        # band lands promptly after the detected rise -- Gooey's 1:00 drop has
        # candidates there and snapping to 60.00 is exactly right. But on a
        # long build there may be nothing for several seconds: This Version of
        # You's hit is at 77.10 against a detection at 74.37, and with a fixed
        # 2.0s window the drop could never reach it. Widening unconditionally
        # fixed that song and pushed Gooey's 1:00 out to 60.67, so the wider
        # search is a fallback rather than the rule.
        def _window(span: float) -> list[tuple[float, float]]:
            out: list[tuple[float, float]] = []
            for vn in ("kick", "snare"):
                v = voices.get(vn)
                if v is None:
                    continue
                out += [(float(t_), float(s_))
                        for t_, s_ in zip(v.onsets, v.strengths)
                        if dt <= t_ <= dt + span]
            return out

        cands = _window(2.00) or _window(3.50)
        for vn in ():
            v = voices.get(vn)
            if v is None:
                continue
            cands += [(float(t_), float(s_)) for t_, s_ in zip(v.onsets, v.strengths)
                      # Never earlier than the detection: the energy detector
                      # reports the rise, so the true hit is always at or
                      # after it, and allowing a backward snap put 1:23 at
                      # 83.51 when the drums enter at 83.80.
                      # 3.5s, not 2.0. The energy detector reports the RISE,
                      # and on a long build the band can land well after it:
                      # This Version of You's big hit is at 77.10 while the
                      # detector fires at 74.37, 2.73s earlier and outside the
                      # old window, so the drop could never snap to it. Still
                      # "the first hit above 55% of the window peak", so a
                      # wider window finds the entry rather than a louder bar
                      # after it.
                      if dt <= t_ <= dt + 3.50]
        if not cands:
            return dt
        _mx = max(s_ for _t, s_ in cands)
        best_t = min(t_ for t_, s_ in cands if s_ >= _mx * 0.55)
        nb = min(beats, key=lambda b: abs(b - best_t)) if beats else best_t
        # Trust the beat grid only when it agrees with the hit closely; a
        # drum entry that lands slightly ahead of the grid is still the drop.
        return nb if abs(nb - best_t) <= 0.12 else best_t

    _snapped = 0
    SNAPPED_DROPS: list[float] = []
    for _ev in _drops + _rest:
        _new = _snap_drop(float(_ev["drop_time"]))
        SNAPPED_DROPS.append(_new)
        if abs(_new - _ev["drop_time"]) > 0.05:
            _snapped += 1
            # A snap of more than half a second means the detector fired on an
            # EARLIER arrival than the one the band lands on -- which is the
            # definition of a double drop. Harmony's 3:18 is detected at 198.6
            # and snaps to 199.7, and those are exactly the two hits Krists
            # describes ("at 3:18 nothing happens, it's only on the second
            # drop at 3:19-3:20 that the lights all come on"). Keeping the
            # original time gives stage one for free, instead of hunting back
            # through onsets for it and landing a bar early.
            if _new - _ev["drop_time"] >= 0.5:
                _ev["stage1_time"] = float(_ev["drop_time"])
            _ev["drop_time"] = _new
    if _snapped:
        print(f"  snapped {_snapped} drop times onto the drum hit they mark")

    cutouts = (_drops + _rest)[:6]
    cutouts.sort(key=lambda ev: ev["cutout_start"])
    blackout_windows = []
    ci_drop = 0
    # per-prop implosion / one house-wide collapse / a directional wipe
    _collapse_modes = ("prop", "house", "wipe")
    _stage1_times: set[float] = set()
    for ev in cutouts:
        c0, c1 = int(ev["cutout_start"] * 1000), int(ev["cutout_end"] * 1000)
        drop = int(ev["drop_time"] * 1000)
        # For a classic cutout the audio is silent from c0 to c1 and the dark
        # hold belongs AFTER it. For an instrument DROPOUT the hole IS the
        # gesture -- the bass keeps playing through it -- so the darkness has
        # to span the whole thing. Using cutout_end here gave ENIGMA's 2.2s
        # dropout a 320ms dark window and a measured contrast of 1.1x, i.e.
        # no visible set-up at all.
        if ev.get("dropout"):
            dark_start = c0 + 260
        else:
            dark_start = max(c1, c0 + 260)
        # TWO-STAGE RETURN: hold black only until the DRUMS come back, not all
        # the way to the real drop. On ENIGMA the drums return at 2:50.9 over a
        # still-quiet bass and the full band arrives at 2:53.5; blacking out
        # across both would swallow the drums' re-entry, which is itself a
        # moment. So the hole ends at the fake drop, a deliberately MODEST hit
        # marks it, and the full-display treatment is saved for the real one.
        false_ms = None
        if ev.get("false_drop"):
            false_ms = int(float(ev["false_drop"]) * 1000)
            if false_ms <= dark_start + 200 or false_ms >= drop - 400:
                false_ms = None
        # CAP THE BLACKOUT AT THE HOLE, not at the drop. For a dropout the
        # dark stretch IS the hole; once the instruments are back the lights
        # have to come back with them. When the drop lands well after the hole
        # ends, blacking through to it kills music that is already playing --
        # measured: Lose My Cool's hole ends at 5.1s with its drop at 12.0s and
        # rendered 0.0% lit for those seven seconds, and World Is Mine sat at
        # 1.3% for its first twenty. Both at the very top of the song, where it
        # reads as a fault rather than a gesture.
        black_end = false_ms or drop
        if ev.get("dropout"):
            black_end = min(black_end, c1 + int(beat_ms * 0.5))
        # On a double drop the darkness must END at stage one, or it blacks
        # out the very hit stage one exists to mark. Measured on Harmony: the
        # display sat at 0.3-0.6 from 198.5 to 199.0, straight through the
        # 198.6 hit, so "at 3:18 nothing happens".
        _s1_ms = ev.get("stage1_time")
        if _s1_ms is not None:
            black_end = min(black_end, int(float(_s1_ms) * 1000))
        blackout_windows.append((dark_start, max(black_end, dark_start + 200)))
        is_climax_drop = top_drop_time is not None and abs(ev["drop_time"] - top_drop_time) < 0.01
        # Only the strongest drops earn a TOTAL blackout. The climax always
        # does; the rest alternate so the gesture doesn't become routine.
        # Keeping every one of them fully black was making the biggest moment
        # in the song look like all the others.
        # Grade the darkness by how big the drop actually is, rather than
        # alternating. "I think the drops aren't quite graduated enough."
        # Alternating gave only two levels and ignored drop strength entirely,
        # so a modest drop could go pitch black while a bigger one didn't.
        # Ranking by measured drop energy means the depth of the blackout
        # tracks the size of the payoff: the strongest third go fully dark,
        # the rest keep outlines alive.
        energy_rank = sorted(
            range(len(cutouts)),
            key=lambda i: -cutouts[i].get("drop_energy", 0)).index(ci_drop)
        # DOUBLE DROPS: the band arrives in two stages. "It's a weird one
        # because the kick/bass come in first and then a beat or two later the
        # rest of the instruments hit and it's detecting that as the main drop
        # but it's really the first one, or maybe we can treat it like a double
        # drop, add a bunch on the first drop and a beat or two later when the
        # other instruments hit add the rest."
        #
        # Measured at Harmony's 3:18: drums enter at 196.02s and bass at
        # 196.93s, while the full-band energy step lands at 199.0-200.0s --
        # seven beats later, and staging only the second is what made the
        # moment feel late. The first stage now gets its own gesture on the
        # linear props, so the house acknowledges the kick and bass arriving
        # and still has the full display in reserve for the real hit.
        _stage1 = ev.get("stage1_time")
        for _vn in (() if _stage1 else ("kick", "bass")):
            _v = voices.get(_vn)
            if _v is None:
                continue
            # The LAST hit before the drop, and close to it -- not the first
            # one in a four-second window. Measured on Harmony's 3:18: the
            # window caught a pickup at 196.0s and staged everything there,
            # while the hit Krists calls the first drop is at ~198.6 and the
            # full band lands at 199.7. Stage one belongs to the bar the drop
            # is in, not to the bar before it.
            _c = [t_ for t_ in _v.onsets if drop / 1000.0 - 2.2 <= t_
                  <= drop / 1000.0 - 0.5]
            if _c:
                _stage1 = max(_c) if _stage1 is None else max(_stage1, max(_c))
        if _stage1 is not None and t is cast[0]:
            log(_stage1, "double-drop",
                f"first stage {drop / 1000.0 - _stage1:.1f}s before the main hit")
        full_blackout = is_climax_drop or energy_rank < max(1, len(cutouts) // 3)
        # SAVE SOMETHING FOR LATER. "Don't go 100% on the first moment in the
        # song, save some stuff for later drops and moments."
        #
        # Ranking purely by measured energy will happily spend the biggest
        # gesture in the show on the first drop, which leaves the later ones
        # -- often the real payoff on these tracks -- with nowhere to go. The
        # first staged moment therefore never takes a full blackout unless it
        # genuinely IS the song's climax, and its hold is shorter. Everything
        # after it is unaffected, so the escalation comes from withholding at
        # the start rather than from inflating the end.
        _is_first = ci_drop == 0
        if _is_first and not is_climax_drop:
            full_blackout = False
        ci_drop += 1
        # The climax drop holds its pattern a little longer than an ordinary
        # implosion (2.6s vs 1.9s) -- the biggest moment in the song earning
        # slightly more screen time before ambient resumes is part of making
        # it read as the climax rather than one implosion among six.
        hold_end = drop + (2600 if is_climax_drop
                           else (1500 if _is_first else 1900))
        # collapse inward, hold TRUE black (ambient gets clipped out of this
        # window below), then everything punches on the drop with varied,
        # pattern-rich content -- not just a flat flash -- and holds briefly
        # before ambient resumes, so "almost no lights" to "almost all
        # lights, doing something" actually reads as the big contrast it's
        # meant to be.
        _cmode_now = _collapse_modes[ci_drop % len(_collapse_modes)]
        if _cmode_now == "house" and HOUSE_GROUP:
            add(HOUSE_GROUP, L_SIG, "ripple_implode", c0, dark_start,
                PALETTES["white"], SPAN)
        for t in cast:
            # Large canvases kept showing up in red-flash reports even after
            # the buffer-style fix and the feature-pool narrowing -- this
            # collapse phase was a separate code path that only exempted
            # tree-shaped large targets (Mega Tree) and still hardcoded
            # Ripple for every other large canvas (Garage Roof, Sun Room
            # Window Matrix), exactly matching reported red-flash locations
            # and counts (6 Ripple hits on each == exactly the implosion
            # count). Use Shockwave -- proven safe everywhere, every round --
            # for ALL large targets' collapse phase, not just tree-shaped
            # ones.
            # VARY THE COLLAPSE. "When we drop out all the lights before a
            # big drop, we always use the same effect that's the reverse
            # shockwave on every prop individually, we should sometimes use it
            # across the whole house as a group and sometimes use other
            # effects to drop out so it's not always the same."
            #
            # Three modes, rotating by drop index so a song's implosions do
            # not all look alike: the per-prop collapse (what it always did),
            # a collapse run across the WHOLE HOUSE as one spanning canvas so
            # the light retreats across the facade rather than inside each
            # prop, and a directional wipe. Chosen per drop, not per prop, so
            # every prop in one implosion still agrees.
            _cmode = _collapse_modes[ci_drop % len(_collapse_modes)]
            if _cmode == "house" and HOUSE_GROUP:
                pass          # placed once, outside this per-prop loop
            else:
                if _cmode == "wipe":
                    collapse_key = ("sweep_left" if (ci_drop // len(_collapse_modes)) % 2
                                    else "sweep_right")
                else:
                    collapse_key = ("shockwave" if t.get("large")
                                    else "ripple_implode")
                add_target(t, L_SIG, collapse_key, c0, dark_start,
                           PALETTES["white"])
            if full_blackout:
                # A double drop's darkness ends AT stage one. Running it to
                # the main drop blacked out the very hit stage one exists to
                # mark -- measured on Harmony, the display sat at 0.3-0.6 from
                # 198.5 to 199.0, straight through it, so the moment Krists
                # heard as the first drop showed nothing at all.
                _black_end = (int(_stage1 * 1000) if _stage1 is not None
                              else drop)
                add_target(t, L_SIG, "off", dark_start, _black_end,
                           PALETTES["white"])
            elif t["cat"] in ("single_line", "icicle") or "/" in t["name"]:
                # Softened pre-drop: outlines only. "Maybe it doesn't always
                # need to go completely dark, it could just go down to
                # outlines only or a sparse whole house shader or something,
                # so it's not just totally blank." Linear/outline props keep
                # a dim presence and everything else still goes to black, so
                # the shape of the house stays readable through the gap while
                # the contrast into the drop is nearly all still there.
                add_target(t, L_SIG, "twinkle_soft", dark_start, drop,
                           PALETTES["white"])
            else:
                add_target(t, L_SIG, "off", dark_start, drop, PALETTES["white"])
            # White for the instant of impact (maximum contrast against the
            # blackout that precedes it), then a warm gold pattern. Both were
            # `ember` when ember led with #FF2A00, which meant every drop in
            # the song flashed the ENTIRE display saturated red -- a big part
            # of the measured red fraction outside the xmas-palette sections.
            if false_ms is not None and (hash(t["name"]) % 3) == 0:
                # One prop in three, briefly -- the drums are back but the song
                # is not. Deliberately smaller than the real drop below so the
                # two moments stay distinguishable.
                add_target(t, L_SIG, "on_hit", false_ms, false_ms + 200,
                           PALETTES["ice"])
                add_target(t, L_SIG, "wash", false_ms + 200, drop, PALETTES["ice"],
                           extra={"T_TEXTCTRL_Fadein": "0.0",
                                  "T_TEXTCTRL_Fadeout": "0.8"})
            # The white hit hands over to the pattern ON THE NEXT BEAT, not at
            # a fixed +260ms. At 95.7 BPM that fixed offset put a 72-effect
            # wave 0.4 of a beat after the drop -- in the gap between beats --
            # which is what read as "there's an additional large effect that
            # comes in on the second or third beat, so it's feeling a little
            # off." A drop is two events (the hit, then the thing it becomes);
            # both have to be on the grid.
            nb = next((b for b in beats if b * 1000 > drop + 120), None)
            hand = int(nb * 1000) if nb and nb * 1000 - drop <= 900 else drop + 260
            add_target(t, L_SIG, "on_hit", drop, hand, PALETTES["white"])
            drop_pal = PALETTES["climax"] if is_climax_drop else PALETTES["gold"]
            add_target(t, L_SIG, drop_hit_key(t, is_climax_drop), hand, hold_end, drop_pal)
            # THE DROP HAS TO BE BRIGHT. "It just seemed like the drops lit up
            # brighter before somehow" -- they did: the no-flash guard was
            # converting the big canvases' hit into a travelling pattern, and
            # Gooey's 1:00 drop went from a peak of 175 to 89.7. A swell with a
            # 0.02s attack and a long decay is not a flash (it never switches
            # off), so it passes the guard and puts the brightness back.
            if _stage1 is not None and (t["cat"] in ("single_line", "icicle")
                                        or any(h in t["name"].lower()
                                               for h in ("outline", "vertical",
                                                         "horizontal", "c9",
                                                         "snowflake", "flake",
                                                         "candy", "cane"))):
                # STAGE ONE HAS TO READ. "It still only hits on the second part
                # of the drop, the first part has a very subtle sparkle that's
                # not impactful. I think the first drop should have all the
                # outlines come on or maybe just the snowflakes and canes."
                #
                # It was being CLEARED: stage one lands several seconds before
                # the drop (3.5s at Harmony's 3:18) and the ambient pass strips
                # signature-layer effects that are not within 600ms of a staged
                # drop. Its time is now registered so the clearing spares it,
                # the prop set includes the snowflakes and candy canes Krists
                # named, and it holds full brightness for a beat before
                # decaying instead of fading from the first frame.
                _s1 = int(_stage1 * 1000)
                _stage1_times.add(_stage1)
                add_target(t, L_SIG, "on_hit", _s1, _s1 + 2000, PALETTES["white"],
                           extra={"T_TEXTCTRL_Fadein": "0.03",
                                  "T_TEXTCTRL_Fadeout": "1.40"})
            if t.get("large") or prominence_of(t) >= 2:
                # `on_hit` (a plain On at full level), not a Color Wash: the
                # wash version only recovered the drop peak from 99 to 104
                # against 191 before the guard, because a wash is not full
                # brightness. The long fade-out is what keeps it a swell.
                add_target(t, L_SIG, "on_hit", drop, drop + 1600,
                           PALETTES["white"],
                           extra={"T_TEXTCTRL_Fadein": "0.02",
                                  "T_TEXTCTRL_Fadeout": "1.40"})
        log(ev["cutout_start"], "implosion",
            f"collapse to true black, hold, full-display hit + pattern on drop @{ev['drop_time']:.1f}s")
    print(f"Pre-drop implosions: {len(cutouts)} staged (of "
          f"{len(cues.get('predrop_cutouts', []))} cutout candidates + "
          f"{len(cues.get('energy_drops', []))} energy drops; "
          f"{n_energy_drops} energy drops used, {n_dropouts} instrument dropouts)"
          + (" -> " + ", ".join(f"{c['drop_time']:.0f}s"
                                + ("*" if c.get("energy_drop") else "")
                                for c in cutouts) if cutouts else ""))

    # Ambient (layers 0-2) effects that overlap a blackout window get clipped
    # out entirely -- without this, the ambient bed keeps quietly running
    # underneath layer 3's "Off" the whole time, so the "hold black" moment
    # never actually reads as near-total darkness, undercutting the contrast
    # with the hit that follows. This is why the drop wasn't dramatic enough.
    def clip_for_blackout(effects_list, windows):
        if not windows:
            return effects_list
        out = []
        for eff in effects_list:
            if id(eff) in _cycle_ids:
                out.append(eff)
                continue
            if eff.layer not in (L_BASE, L_MOTION, L_PULSE):
                out.append(eff)
                continue
            pieces = [(eff.start_time_ms, eff.end_time_ms)]
            for w0, w1 in windows:
                new_pieces = []
                for s, e in pieces:
                    if e <= w0 or s >= w1:
                        new_pieces.append((s, e))
                        continue
                    if s < w0:
                        new_pieces.append((s, w0))
                    if e > w1:
                        new_pieces.append((w1, e))
                pieces = new_pieces
            for s, e in pieces:
                if e - s >= 40:
                    # Re-snap: clipping computes new boundaries from blackout
                    # windows in seconds, which lands them off the render frame
                    # grid again -- see add()'s note on frame jitter.
                    s = int(round(s / FRAME_MS) * FRAME_MS)
                    e = int(round(e / FRAME_MS) * FRAME_MS)
                    if e <= s:
                        continue
                    out.append(eff.model_copy(update={"start_time_ms": s, "end_time_ms": e}))
        return out

    effects = clip_for_blackout(effects, blackout_windows)

    # The big canvases, as one group. Used by BOTH the cohort lock below and
    # the run-up dip further down, so it has to be defined ahead of the first
    # of them -- it originally lived inside the dip block, and moving the lock
    # above the dip left it referencing a name that did not exist yet.
    cohort_names = set()
    for _t in cast:
        if _t.get("large"):
            cohort_names.add(_t["name"])
            cohort_names.update(_t.get("members") or [])

    # --- cohort lock: pull straggler big canvases onto the shared grid -----
    # "There are other parts of Gooey also where the matrices either start or
    # end at slightly different times from each other... let's get them more
    # together."
    #
    # hold_spans() puts every large canvas on the same bar marks and the dip
    # now moves them as a group, which removes the two causes found by
    # measurement. This is the backstop for any third: a cohort hold that is
    # within ~a beat of a shared boundary is snapped onto it. Misalignments
    # that small are never intentional -- they are rounding, clipping or an
    # overlap nudge -- and at 40fps a 375ms stagger between two matrices
    # running the same shader is clearly visible as "oddly aligned".
    #
    # ORDER MATTERS: this runs BEFORE the run-up dip, not after. Run
    # afterwards it snapped dip-truncated ends back out to the full bar
    # boundary and silently undid the breath -- measured, ENIGMA's dip
    # median sat at 0.82-0.88 against 0.61 before the cohort work, and
    # raising the dip probability from 0.66 to 0.9 moved Gooey and Fara not
    # at all (0.817 and 0.900 both times), which is what gave it away. The
    # dip truncates every cohort member to the SAME window start, so running
    # it second keeps them aligned anyway.
    #
    # Guarded against undoing a blackout: clip_for_blackout() ran earlier, so
    # snapping an end LATER could push an effect back into darkness that was
    # deliberately cleared. Any snap that would newly overlap a blackout
    # window is dropped.
    if cohort_marks:
        import bisect
        # THE GRID ITSELF LANDS ON HITS. These marks come from cycle and hold
        # boundaries computed by several passes, and the lock then pulls every
        # big canvas onto the nearest one -- so a mark sitting a beat early
        # drags the whole facade there with it. Measured on Gooey: the intro's
        # second cycle is placed at 15.75, the grid held 15.16, and every big
        # canvas got pulled to 15.16 while the kick and bass land at 15.85.
        # "The shaders change before the kick/bass hits."
        #
        # Snapping the marks first means the cohort still moves as one -- which
        # is the whole point of it -- but onto the beat the music is on.
        _bt_g = np.asarray(beats, dtype=float)
        _beat_g = beat_ms / 1000.0

        def _mark_on_hit(ms: int) -> int:
            _x = ms / 1000.0
            _lo, _hi = _x - 0.12, _x + 1.15 * _beat_g
            _c = [float(t) for _vn in ("kick", "snare", "bass")
                  if (_v := voices.get(_vn)) is not None
                  for t in _v.onsets if _lo <= t <= _hi]
            _tgt = min(_c) if _c else _x
            if _bt_g.size:
                _tgt = float(_bt_g[int(np.argmin(np.abs(_bt_g - _tgt)))])
            return int(round(_tgt * 1000))

        grid = sorted({_mark_on_hit(m) for m in cohort_marks})

        def _snap(v: int) -> int:
            i = bisect.bisect_left(grid, v)
            best = None
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(grid) and (best is None
                                           or abs(grid[j] - v) < abs(best - v)):
                    best = grid[j]
            return best if best is not None and abs(best - v) <= 900 else v

        def _in_blackout(a: int, b: int) -> bool:
            return any(a < int(wb * 1000) and b > int(wa * 1000)
                       for wa, wb in blackout_windows)

        n_snapped = 0
        for eff in effects:
            # Ambient chord hits are placed ON a chord and must stay there.
            # The cohort grid was pulling 22 of the 25 props lit by the 36.0s
            # chord onto its own boundaries, which is exactly the complaint
            # that the chords were "messy and not on the beat of the piano
            # chords" -- the hits were not missing, they had been dragged off
            # the note. The grid exists to align long canvas holds with each
            # other, and a 2.6s chord wash is not one of those.
            if (is_ambient_eff(eff)
                    and eff.end_time_ms - eff.start_time_ms <= 6000):
                continue
            if eff.layer not in (L_BASE, L_MOTION):
                continue
            if eff.end_time_ms - eff.start_time_ms < 1200:
                continue
            nm = eff.model_name
            if not (nm in cohort_names or nm.split("/")[0] in cohort_names):
                continue
            a, b = _snap(eff.start_time_ms), _snap(eff.end_time_ms)
            if b - a < 600 or (a, b) == (eff.start_time_ms, eff.end_time_ms):
                continue
            if _in_blackout(a, b) and not _in_blackout(eff.start_time_ms,
                                                       eff.end_time_ms):
                continue
            eff.start_time_ms, eff.end_time_ms = a, b
            n_snapped += 1
        _probe("Cohort lock:")
        print(f"Cohort lock: {n_snapped} big-canvas holds pulled onto the "
              f"shared grid ({len(grid)} boundaries)")

    # --- run-up dip: thin the last beat of each section ---------------------
    # "The dynamic detection that would get a bit dimmer before a new phrase
    # or drop or fill isn't working, it's too static and always lit up."
    #
    # Section-level headroom can only make a whole section dimmer -- it cannot
    # create the short breath immediately BEFORE a change, which is what makes
    # the change land. This drops a fraction of the ambient/texture layers for
    # the last beat of each section, leaving the rhythm and signature layers
    # untouched so the groove never stutters. Deliberately partial (two thirds
    # of the eligible ambient effects, deterministic per section) -- a full
    # cut would read as a mistake rather than as a breath.
    dip_rng = random.Random(f"{SONG_NAME}|runup-dip")
    n_dipped = 0
    dip_windows = []
    for section in analysis.sections[:-1]:
        if section.end_time - section.start_time < bar_s * 2:
            continue
        dip_windows.append((section.end_time - beat_ms / 1000.0,
                            section.end_time))
    # ...AND BEFORE EVERY DROP. The dip was built from section ENDS only, so a
    # drop that does not coincide with one got no breath before it at all.
    # Measured on Gooey, seven of its ten drops had a run-up as bright as the
    # passage they came out of (dip ratio 0.96-1.24, where a real hush is
    # ~0.5): "sometimes there's not enough of a dropout first or a brightness
    # change on the drop." A drop with nothing taken away beforehand has
    # nowhere to rise from, however hard it is hit.
    #
    # A beat and a half rather than the section dip's one beat -- a drop is a
    # bigger moment and wants a longer inhale before it.
    for _dt in sorted(set(SNAPPED_DROPS)):
        if _dt > bar_s:
            dip_windows.append((_dt - 1.5 * beat_ms / 1000.0, _dt - 0.02))
    # Minor drops get a breath too, just a shorter one -- one beat rather than
    # a beat and a half.
    for _dt in sorted(set(MINOR_DROPS)):
        if _dt > bar_s:
            dip_windows.append((_dt - beat_ms / 1000.0, _dt - 0.02))
    # The dip is deliberately PARTIAL for ordinary props -- a full cut reads
    # as a mistake rather than as a breath. But rolling that dice per EFFECT
    # meant the big canvases disagreed with each other: at any given section
    # approach some matrices truncated into the dip and their neighbours ran
    # straight through, which is precisely "they turn off at odd times and not
    # all at the same time with each other." The cohort decides ONCE per
    # window and moves together; everything else keeps its own dice.
    # Near-always for the cohort, where the per-effect rate is 0.66. The two
    # numbers do different jobs: 0.66 PER EFFECT produces a partial thinning
    # within one prop's own stream, which is what keeps a dip from reading as
    # a mistake. Once the big canvases move as one, that same 0.66 becomes a
    # coin flip on whether the breath happens AT ALL, and the canvases carry
    # most of the lit area -- measured, the dip median went 0.61 -> 0.88 on
    # ENIGMA, i.e. the breath before a section change all but disappeared.
    # Synchronised means it should happen nearly every time and be felt.
    cohort_dip = [random.Random(f"{SONG_NAME}|cohort-dip|{i}").random() < 0.9
                  for i in range(len(dip_windows))]
    if dip_windows:
        kept = []
        for eff in effects:
            # Ambient holds are exempt, like cycle holds. The dip trims the
            # last beat before a section change, which is right for ordinary
            # ambient fill and catastrophic for a passage-length hold: on This
            # Version of You it cut the 37-second Herringbone at 1:17 down to
            # 1.33s, because the song has a section boundary just after it.
            # That hold IS the passage's content, not fill to be thinned.
            if (id(eff) not in _cycle_ids and not is_ambient_eff(eff)
                    and eff.layer in (L_BASE, L_MOTION)):
                s_s, e_s = eff.start_time_ms / 1000.0, eff.end_time_ms / 1000.0
                hit_i = next((i for i, (a, b) in enumerate(dip_windows)
                              if s_s < b and e_s > a), None)
                hit = dip_windows[hit_i] if hit_i is not None else None
                in_cohort = (eff.model_name in cohort_names
                             or eff.model_name.split("/")[0] in cohort_names)
                take = (cohort_dip[hit_i] if (hit is not None and in_cohort)
                        else dip_rng.random() < 0.66)
                if hit and take:
                    a, _b = hit
                    if s_s < a - 0.05:
                        # truncate into the dip, on the frame grid
                        eff.end_time_ms = int(round(a * 1000 / FRAME_MS) * FRAME_MS)
                        kept.append(eff)
                    n_dipped += 1
                    continue
            kept.append(eff)
        effects = kept
    _probe("Run-up dips:")
    print(f"Run-up dips: {len(dip_windows)} section approaches, "
          f"{n_dipped} ambient effects thinned")

    # --- 2d-bis. Big hit + slow fade (the repeating quiet-section drop
    # pattern -- e.g. Gooey's opening minute pulses roughly every 11s, each
    # decaying to a moderate baseline rather than silence, so the cutout
    # detector above only ever caught the first one). Without an explicit
    # "hit then fade" gesture here, whatever small amount of ambient/pulse
    # activity is already scheduled for a quiet section just looks like
    # unmotivated flicker instead of following the music's actual story.
    drop_times_used = {int(ev["drop_time"] * 1000) for ev in cutouts}
    # A moderate slice of the cast, not literally everyone -- this needs to
    # read as a clear "hit then fade" gesture, but shouldn't be the same
    # 100%-of-display scale as the true implosion/drop moments elsewhere, or
    # it stops being a special, bigger gesture by comparison.
    hit_rng = random.Random(f"{SONG_NAME}|big-hits")
    hit_targets = hit_rng.sample(cast, k=max(1, len(cast) // 2))
    placed_hits = 0
    for t in big_hits:
        t_ms = int(t * 1000)
        if any(abs(t_ms - d) < 2000 for d in drop_times_used):
            continue  # already covered by a pre-drop implosion
        for tgt in hit_targets:
            add_target(tgt, L_SIG, "on_hit", t_ms, t_ms + 300, PALETTES["white"])
            add_target(tgt, L_SIG, "wash", t_ms + 300, t_ms + 1900, PALETTES["ice"],
                       extra={"T_TEXTCTRL_Fadein": "0.0", "T_TEXTCTRL_Fadeout": "1.6"})
        log(t, "big-hit", "bright hit, then a deliberate slow fade across part of the display")
        placed_hits += 1
    print(f"Big hit + fade moments: {placed_hits} (of {len(big_hits)} candidates)")

    # --- 2e. Melody / bass arpeggio chases across a prop family ---
    def arp_chase(runs, pool, palette, tag):
        placed = 0
        if not pool:
            return 0
        for run in runs:
            onsets = run.get("onset_times", [])
            if len(onsets) < 3:
                continue
            fam = min(pool, key=lambda t: abs(t["n"] - run["period"]) if t["n"] > 1 else 99)
            members = [m for m in fam["members"]]
            if len(members) < 2:
                continue
            members = sorted(members, key=lambda m: positions.get(m, {}).get("x", 0))
            # Hold each note as long as the melody's own pace allows (up to
            # the gap to the next onset) so a slow arpeggio reads as clearly
            # held notes, not a quick flicker -- a fixed short 200ms made a
            # "somewhat slowly repeating" 4-note pattern nearly invisible.
            gaps = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
            typical_gap_ms = int((sorted(gaps)[len(gaps) // 2] if gaps else 0.4) * 1000)
            hold = max(220, min(typical_gap_ms - 40, 900))
            for k, t in enumerate(onsets):
                t_ms = int(t * 1000)
                add(members[k % len(members)], L_SIG, "on_swell", t_ms, t_ms + hold, palette)
                placed += 1
            log(run["start_time"], tag,
                f"{fam['name']} follows {'-'.join(run['note_names'])}")
        return placed

    mel_pool = [t for t in cast if t["cat"] in ("arch", "custom") and t["n"] >= 3
                and not t.get("symmetric") and t["name"] not in prop_voice]
    bass_pool = [t for t in cast if t["cat"] == "matrix" and t["n"] >= 3
                 and not t.get("symmetric") and t["name"] not in prop_voice]
    n_arp = arp_chase(cues.get("arpeggios", []), mel_pool, PALETTES["toxic"], "melody-chase")
    n_arp += arp_chase(cues.get("bass_arpeggios", []), bass_pool, PALETTES["ember"], "bass-chase")
    print(f"Arpeggio chase notes: {n_arp}")

    # --- 2e-bis. Legato string phrase in the chorus -> coordinated side-to-
    # side sway across verticals/curtains/window outlines, like the house
    # (or dancers) swaying. Short notes get a quick partial sway, long notes
    # a fuller held one, alternating direction note to note.
    #
    # The first implementation put an identical local Wave on all ~25 pool
    # members at once, every note. That is NOT what "the house swaying" looks
    # like -- 25 props each running their own little left-right wave in place
    # reads as generalised shimmer, and Krists reported not seeing the sway at
    # all. A sway is a gesture of the WHOLE FACADE: the lit mass itself has to
    # move bodily left, then right, in the rhythm of the phrase. So: sort the
    # pool by real X position, split it into zones, and light one zone per
    # note, walking the zone index back and forth like a metronome/pendulum.
    # Same principle as the arch cascade -- a travelling position, expressed
    # across props, beats any per-prop effect for legibility at house scale.
    sway_pool = [t for t in cast
                 if (t["cat"] == "single_line" or t["name"] in
                     ("Curtain Front", "Curtain North", "Curtain South"))
                 and t["name"] not in prop_voice]
    legato_notes = detect_legato_phrase(SHOW_PATH / "stems" / SONG_NAME, analysis.sections)
    if sway_pool and legato_notes:
        sway_pool.sort(key=lambda t: t["cx"])
        n_zones = max(3, min(5, len(sway_pool)))
        zones: list[list[dict]] = [[] for _ in range(n_zones)]
        for idx, t in enumerate(sway_pool):
            zones[min(idx * n_zones // len(sway_pool), n_zones - 1)].append(t)

        # Direction rule, from precise feedback on the real riff's rhythm:
        # "1-left-right-left-right-2-3-left-right-2-3-4 where numbers stay
        # moving the same direction." Read against the SHORT/LONG split
        # `detect_legato_phrase` already computes (long = a held beat note,
        # "1"/"2"/"3"/"4"; short = one of a quick alternating "left-right"
        # ornament pair): a LONG note continues whatever direction the sway
        # was already travelling -- it does not reverse. A SHORT note is
        # part of an alternating pair and DOES reverse, every time. The
        # previous version flipped direction on every note regardless of
        # long/short (a fixed pendulum keyed to note count), which is why it
        # didn't read as the described rhythm even once it became visible.
        z = n_zones // 2
        direction = 1
        for note in legato_notes:
            s_ms, e_ms = int(note["start"] * 1000), int(note["end"] * 1000)
            if not note["long"]:
                direction *= -1
            z = max(0, min(n_zones - 1, z + direction))
            # bounce off the end instead of getting stuck lighting the same
            # edge zone repeatedly if several notes in a row push that way
            if z in (0, n_zones - 1):
                direction *= -1
            active = [z] + ([z + direction] if note["long"] else [])
            for zi in active:
                if 0 <= zi < n_zones:
                    for t in zones[zi]:
                        add_target(t, L_SIG, "on_swell", s_ms, e_ms,
                                   PALETTES["violet"])
        long_n = sum(1 for n in legato_notes if n["long"])
        log(legato_notes[0]["start"], "string-sway",
            f"{len(legato_notes)} notes ({long_n} long) -> facade pendulum across "
            f"{n_zones} zones / {len(sway_pool)} verticals/curtains")
        print(f"String-sway notes: {len(legato_notes)} as a {n_zones}-zone facade "
              f"pendulum over {len(sway_pool)} targets")
    else:
        print("String-sway notes: none detected")

    # --- 2f. Drum fills roll across the house -------------------------------
    # The old pass here scattered isolated cymbal hits onto a rotating
    # decorative target -- i.e. a different prop each time, on hits that
    # weren't the song's focal point. That is precisely what was described as
    # "other random snare beats can be chaotic if they're not the song's
    # focal point", so it's gone: hats/cymbals now live on ONE dedicated prop
    # via the voice system, and the genuinely special drum moments get this
    # instead.
    #
    # "Drum fills are actually really great to highlight." A fill is a
    # rolling gesture, so it's rendered as one: each hit of the fill advances
    # across X-ordered prominent props, so the fill literally travels across
    # the facade and lands at the far end as the next section starts.
    fill_pool = [t for t in cast_by_x
                 if t["n"] >= 2 and not t.get("symmetric")
                 and t["name"] not in prop_voice]
    n_fill_hits = 0
    if drum_fills and fill_pool:
        for f0, f1 in drum_fills:
            hits = sorted({t for vn in ("snare", "hat", "kick")
                           if vn in voices
                           for t in voices[vn].in_window(f0, f1)})
            if len(hits) < 3:
                continue
            hold = max(120, min(int(beat_ms * 0.4), 240))
            for hi, t in enumerate(hits):
                tgt = fill_pool[int(hi * len(fill_pool) / max(1, len(hits)))]
                t_ms = int(t * 1000)
                add_target(tgt, L_SIG, "on_pop", t_ms, t_ms + hold,
                           PALETTES["white"])
                n_fill_hits += 1
            log(f0, "drum-fill",
                f"{len(hits)} hits rolling across {len(fill_pool)} props")
    print(f"Drum-fill highlights: {len(drum_fills)} fills, {n_fill_hits} hits")

    # --- 2g. Melody mapped to physical position across a multi-member prop --
    # "I want to try extracting some melodies too when they're clear, and try
    # showing them on something I have multiple of, so like if there are 8
    # orbs they can represent 8 notes."
    #
    # Distinct from the existing arp_chase, which walks a family's members in
    # order regardless of what the notes actually do. Here the note's PITCH
    # picks the member: lowest note of the phrase -> leftmost prop, highest
    # -> rightmost. Same note always lights the same prop, so a repeated
    # melodic phrase draws the same shape on the house every time.
    n_melody = 0
    _mel_played: list[str] = []
    if melody_notes:
        mel_candidates = [
            t for t in cast
            # The spheres and orbs may play the melody even if they already
            # carry an instrument voice. The rule exists so one prop is not
            # asked to represent two things at once, but these two are the
            # only families with enough discrete members to make pitch legible
            # -- "the orbs are also enough to do melodies" -- and excluding
            # them left the melody with nowhere good to go.
            if t["n"] >= 4 and not t.get("symmetric")
            and (t["name"] not in prop_voice
                 or t["name"] in ("Hanging Spheres", "Orbs"))
            and t["cat"] != "arch"
            and "/" not in t["name"]
            and not any(h in t["name"].lower() for h in OUTLINE_NAME_HINTS)
        ]
        if mel_candidates:
            # SEVEN SPHERES IS AN OCTAVE. "There are seven of them which is
            # essentially a full octave, you could use them to play a slow
            # melody or bass line as long as it's not too chaotic." A scale
            # degree maps straight onto one sphere, so a repeated phrase draws
            # the same shape in the air every time -- which is the whole point
            # of pitch-mapping and is much clearer on seven discrete objects
            # than on a row of mixed props.
            # ...and the Orbs carry one just as well: "the orbs are also
            # enough to do melodies, you can use both together or one at a
            # time or whatever too for variety." So the phrase, not the song,
            # picks who plays it -- spheres, orbs, or both at once -- which
            # keeps a long melodic song from becoming one prop's job.
            _mel_pref = [t for t in mel_candidates
                         if t["name"] in ("Hanging Spheres", "Orbs")]
            _mel_fallback = max(
                mel_candidates,
                key=lambda t: (t["name"] in MUST_HAVE_GROUPS, t["n"]))
            _mel_pool = _mel_pref or [_mel_fallback]

            def _members_of(t):
                return sorted(t["members"],
                              key=lambda m: positions.get(m, {}).get("x", 0))

            mel_target = _mel_pool[0]
            members = _members_of(mel_target)
            # Drop notes inside an ambient passage. The melody follows the
            # strings or vocal, a DIFFERENT rhythm from the piano chords such a
            # passage is built around -- "there is something else happening too
            # on some other rhythm... it should really be just smooth movement
            # flowing, with on/fadeout on each of those piano chords."
            melody_notes = [n for n in melody_notes
                            if not in_ambient(n.get("start", 0.0))]
            if not melody_notes:
                mel_candidates = []
            n_deg = (melody_notes[0].get("n_degrees", 1) or 1) if melody_notes else 1
            # Phrases: a gap of more than two seconds starts a new one, and
            # each phrase is assigned its player. Alternating WITHIN a phrase
            # would scatter one musical line across the house, which is the
            # "chaotic and random" he warned against.
            _phrase_of: dict[int, int] = {}
            _ph = 0
            _prev_end = None
            for _i, _n in enumerate(melody_notes):
                if _prev_end is not None and _n["start"] - _prev_end > 2.0:
                    _ph += 1
                _phrase_of[_i] = _ph
                _prev_end = _n.get("end", _n["start"])
            _n_phrases = _ph + 1

            def _players(ph: int) -> list[dict]:
                if len(_mel_pool) < 2:
                    return _mel_pool
                _r = random.Random(f"{SONG_NAME}|melody-player|{ph}").random()
                if _r < 0.25:
                    return _mel_pool            # both together
                return [_mel_pool[ph % len(_mel_pool)]]

            _played: set[str] = set()
            for _i, note in enumerate(melody_notes):
                s_ms = int(note["start"] * 1000)
                e_ms = max(s_ms + 160, int(note["end"] * 1000))
                for _t in _players(_phrase_of[_i]):
                    _mem = _members_of(_t)
                    if not _mem:
                        continue
                    # Map the note's pitch rank onto a member index. More
                    # notes than props is normal -- several adjacent semitones
                    # share a prop, which still preserves "melody goes up =
                    # light moves right" as the readable gesture.
                    _idx = int(note["degree"] * len(_mem) / max(1, n_deg))
                    _idx = max(0, min(len(_mem) - 1, _idx))
                    add(_mem[_idx], L_SIG, "on_swell", s_ms, e_ms,
                        PALETTES["gold"])
                    _played.add(_t["name"])
                    n_melody += 1
            if melody_notes:
                log(melody_notes[0]["start"], "melody",
                    f"{len(melody_notes)} notes over {n_deg} pitches across "
                    f"{_n_phrases} phrases -> {', '.join(sorted(_played))} "
                    f"by pitch height")
                _mel_played = sorted(_played)
    print(f"Melody notes mapped: {n_melody}"
          + (f" -> {', '.join(_mel_played)}" if n_melody and _mel_played else ""))

    # --- 2h. Vocal phrase entries ------------------------------------------
    # A small swell on whichever prop already carries the `vocal` voice, at
    # the START of each sung phrase. Phrase starts are where a listener's
    # attention lands, and they're sparse (~45 in a song) so this can't
    # become flicker. Uses only WHEN singing begins -- nothing depends on the
    # words, which measured too unreliable to build on.
    #
    # Deliberately on the prop the vocal voice already owns, and on smaller
    # props by construction (VOICE_PROP_PREFS puts `vocal` on custom/matrix,
    # never a large canvas): "use smaller props for following melodies or
    # vocals if they have a clear rhythm."
    n_phrase = 0
    vocal_target = voice_props.get("vocal")
    if vocal_target and vocal_phrases_list:
        for p in vocal_phrases_list:
            s_ms = int(p["start"] * 1000)
            e_ms = min(int(p["end"] * 1000), s_ms + int(bar_s * 1000))
            if e_ms - s_ms < 200:
                continue
            add_target(vocal_target, L_SIG, "on_swell", s_ms, e_ms,
                       PALETTES["white"])
            n_phrase += 1
        if n_phrase:
            log(vocal_phrases_list[0]["start"], "vocal-phrases",
                f"{n_phrase} phrase entries on {vocal_target['name']}")
    print(f"Vocal phrase accents: {n_phrase}"
          + (f" -> {vocal_target['name']}" if n_phrase else ""))

    # --- mirror the real tunnel onto its preview arches ---------------------
    #
    # "I enabled the tunnel arches, which remember are my way of seeing what
    # the tunnel would look like even though the tunnel matrix is the real prop
    # I put on the controller to drive those lights. Can you please make sure
    # the tunnel arches group shows the same effects as the tunnel matrix?"
    #
    # All 80 `Tunnel Arches N` models carry `Controller=No Controller` -- they
    # exist purely so the walk-through tunnel is visible in the preview. The
    # light that actually ships comes from `Tunnel Matrix Real`. Mirroring onto
    # the `Tunnel` GROUP (one placement each, rather than 80) keeps the preview
    # honest without inflating the file. The matrix is itself a member of that
    # group, so it receives each effect twice -- identical content at identical
    # times, so it renders the same; correctness costs nothing here.
    TUNNEL_MIRROR_SRC = "Tunnel Matrix Real"
    TUNNEL_MIRROR_DST = "Tunnel"
    # --- ambient passages: take the RHYTHM back out ------------------------
    # Placing smooth gestures is only half of it. The beat anchor, gutter
    # flashes, submodel stepping, leaping arches and ensemble flashes all run
    # off their own gates and will happily keep flashing underneath -- which
    # is the state Krists described: rhythmic when it should be flowing.
    #
    # Short hits on the rhythm layers come out; anything longer stays, because
    # that is a gesture rather than a beat mark. Staged drops are exempt: a
    # quiet passage often ENDS on one, and that moment is the payoff the whole
    # passage was building toward.
    if ambient_passages:
        _rms_all = np.asarray(analysis.spectrum.rms_energy)
        _rt_all = np.asarray(analysis.spectrum.rms_times)
        _drop_ms = ([int(ev["drop_time"] * 1000) for ev in cutouts]
                    + [int(x * 1000) for x in _stage1_times])
        _kept, _cut = [], 0
        for eff in effects:
            if is_ambient_eff(eff) or eff.layer not in (L_PULSE, L_SIG):
                _kept.append(eff)
                continue
            # NO DURATION ESCAPE. The 400ms cutoff let the actual offenders
            # through: measured on The Last Goodbye's intro -- 0.03 drum hits
            # per second -- the beat-rate pulsing came from `Candy Canes` L3
            # washes and L1 chases that were all LONGER than 400ms, so every
            # one of them survived the clear. The rhythm layers have no
            # business in a passage with no rhythm, at any length; movement
            # lives on the BASE and MOTION layers, which are untouched here.
            pass
            s_s = eff.start_time_ms / 1000.0
            if not any(a["t0"] <= s_s < a["t1"] for a in ambient_passages):
                _kept.append(eff)
                continue
            if any(abs(eff.start_time_ms - d) <= 600 for d in _drop_ms):
                _kept.append(eff)      # the drop this passage was building to
                continue
            # KEEP THE DOWNBEATS. Clearing every rhythm hit took the groove
            # out of songs that still have one in their quiet passages:
            # measured on The WhatsApp Shakespeare, whole-song beat lift went
            # 1.332x (v35) to 1.008x the moment the ambient treatment landed,
            # and only recovered to 1.09 after the classification fix. Its
            # ambient passages carry 0.20-0.86 drum hits a second -- sparse,
            # but not nothing.
            #
            # One hit per BAR survives, which is the rate Krists asked these
            # passages to move at anyway ("pulsing more like each bar or
            # couple of bars"), so the song keeps a pulse without the passage
            # turning rhythmic again.
            if downbeats and any(abs(eff.start_time_ms / 1000.0 - d) <= 0.08
                                 for d in downbeats):
                _kept.append(eff)
                continue
            _cut += 1
        # Flat ambient fill on the BASE layer comes out too: it is a constant
        # level under a passage whose whole point is that the level moves.
        # Motion layers are left alone -- smooth is not the same as empty, and
        # the movement is what keeps a long quiet passage alive.
        _kept2 = []
        for eff in _kept:
            if (id(eff) not in _ambient_ids and eff.layer == L_BASE
                    and eff.effect_name in ("Color Wash", "On", "Twinkle")
                    and any(a["t0"] <= eff.start_time_ms / 1000.0 < a["t1"]
                            for a in ambient_passages)):
                _cut += 1
                continue
            _kept2.append(eff)
        # NO BEAT-RATE PULSING IN A PASSAGE WITH NO BEAT. "Since there's no
        # drums it's odd to pulse on quarter notes or half notes, I think
        # pulsing more like each bar or couple of bars would be better...
        # The ambient sections should swell more gently and slowly."
        #
        # The holds placed by the ordinary section path carry a beat-locked
        # brightness curve, and clearing short hits does not touch them -- so
        # the passage kept pulsing on the quarter note underneath the smooth
        # treatment. Any such curve inside an ambient passage is replaced with
        # the passage's own energy contour over that effect's span, which
        # moves on the scale of bars because the music does.
        # THE AMBIENT HOLD OWNS ITS LAYER. Effects share layer slots, so an
        # earlier short span sitting on the same layer silently displaces the
        # long hold: on This Version of You the ambient pass placed Herringbone
        # from 77.10 to 114.78 and a 1.33s span already on that layer meant the
        # 37-second hold never appeared at all -- which is why the 1:17 shader
        # "changes to a different one a second later". Anything overlapping an
        # ambient hold on the same model and layer, that is not itself part of
        # the ambient treatment, is removed.
        _amb_spans = [(e.model_name, e.layer, e.start_time_ms, e.end_time_ms)
                      for e in _kept if is_ambient_eff(e)
                      and e.end_time_ms - e.start_time_ms >= 4000]
        _displaced = 0
        _kept2b = []
        for eff in _kept:
            # Long holds are never the interloper -- and cannot be identified
            # by id(), because clip_for_blackout REBUILDS effects as new
            # objects and every id-based tag is lost at that point. That is
            # how this filter deleted the very hold it was written to protect:
            # the 37-second Herringbone survived the ambient pass, was clipped
            # into a new object, and was then removed here as an intruder.
            # Anything id-based must run before the clipping, or key on
            # something the clipping preserves.
            if eff.end_time_ms - eff.start_time_ms >= 4000:
                _kept2b.append(eff)
                continue
            if not is_ambient_eff(eff) and any(
                    eff.model_name == m and eff.layer == ly
                    and eff.start_time_ms < b and eff.end_time_ms > a
                    for m, ly, a, b in _amb_spans):
                _displaced += 1
                continue
            _kept2b.append(eff)
        if _displaced:
            print(f"Ambient passages: removed {_displaced} effects sharing a "
                  f"layer with a long hold")
        _kept = _kept2b
        _kept3 = []
        for eff in _kept:
            pal = eff.palette
            if (not is_ambient_eff(eff) and pal is not None and pal.extra_parts
                    and any(x.startswith("C_VALUECURVE_Brightness=")
                            for x in pal.extra_parts)
                    and any(a["t0"] <= eff.start_time_ms / 1000.0 < a["t1"]
                            for a in ambient_passages)):
                base = pal.model_copy(update={"extra_parts": [
                    x for x in pal.extra_parts
                    if not x.startswith("C_VALUECURVE_Brightness=")]})
                eff.palette = envelope_palette(base, _rms_all, _rt_all,
                                               eff.start_time_ms,
                                               eff.end_time_ms, depth=0.6)
                # Protect it from the rebuild pass. Without this the clearing
                # swapped a beat pulse for an envelope and the rebuild swapped
                # it straight back -- which is why four separate fixes to the
                # ambient pulsing all measured as inert while each was
                # provably applied. The rebuild only skips _ambient_ids.
                _ambient_ids.add(id(eff))
            _kept3.append(eff)
        # SLOW EVERYTHING THAT MOVES, not just the holds this pass placed.
        #
        # The Last Goodbye's intro has 0.03 drum hits per second and still
        # measured 13.5% of its brightness energy at the BEAT frequency. It
        # was never the brightness curves: shader speed is tempo-scaled
        # (400 * tempo/120), so the shader MOTION itself carries the beat, and
        # chases carry it in their rotation count. Slowing only the shaders
        # this pass places left every shader the section path had already put
        # there running at song speed underneath.
        # QUIET SECTIONS SIT BELOW LOUD ONES. Measured on Harmony, the
        # ambient passages were running BRIGHTER than the choruses (70% lit
        # against 61%), which is backwards and is why "the drops and peak
        # climax moments don't seem as big -- they can't be brighter than the
        # rest." Krists's target: "maybe 15-30% darker" rather than half.
        #
        # Applied as a ceiling on the brightness curve rather than by removing
        # content, so the passage keeps its texture and movement and simply
        # sits lower. Effects with no curve get a flat one at the same level.
        # A UNIFORM FACTOR CANNOT HIT A PER-SONG TARGET. 0.50 landed Harmony
        # at -18% and This Version of You at -25% (both in the 15-30% band
        # Krists asked for) but Cruise at -37% and Meridian at -46%, which is
        # near the "half as bright" he had already rejected as too dark. How
        # much of a song's light comes from the ambient pass versus every
        # other pass differs per song, so the same multiplier lands
        # differently.
        #
        # The factor is therefore calibrated per song from a real render and
        # cached -- see calibrate_ambient_dim.py. Absent a cache it falls back
        # to 0.50, which is right for about half the library.
        _DIM = 0.50
        try:
            _DIM = float(json.loads(
                (GEN_DIR / "ambient_dim.json").read_text())["dim"])
        except (OSError, ValueError, KeyError):
            pass
        _dimmed = 0
        for eff in _kept3:
            # OVERLAP, not start-inside: a hold that begins before the passage
            # and runs through it lights the passage just as much, and those
            # were escaping the dim entirely.
            if not any(a["t0"] < eff.end_time_ms / 1000.0
                       and a["t1"] > eff.start_time_ms / 1000.0
                       for a in ambient_passages):
                continue
            # The chord accents ARE the foreground of an ambient passage --
            # the event the passage is built around. Dimming them along with
            # the background defeated the entire point: the accent was written
            # at 152% and arrived at the pixels at 64%, which is why the
            # chords still read as "a faint glow" after being given a bright
            # contrasting colour. The background around them still dims, so
            # they gain contrast from this rather than losing it.
            if eff.settings.get("X_AmbientAccent") == "1":
                continue
            pal = eff.palette
            if pal is None:
                continue
            # THE BUILD RIDES ON THE DIM, not on the envelope depth. Scaling
            # depth by song position measured as completely inert (brightness
            # by sixth identical to three decimal places), the same way every
            # other depth-based attempt has; the dim factor is the lever that
            # actually moves the render. On a building track an early passage
            # is dimmed harder than a late one, so the show grows across the
            # song instead of restating one level -- This Version of You's
            # lights peaked in the THIRD sixth while its music peaks in the
            # fifth.
            _f = _DIM
            if song_builds:
                _pos = min(1.0, eff.start_time_ms / 1000.0 / max(_dur_s, 1e-6))
                _f = _DIM * (0.62 + 0.68 * _pos)
            parts = list(pal.extra_parts or [])
            vc = next((x for x in parts if x.startswith("C_VALUECURVE_Brightness=")),
                      None)
            if vc is not None:
                m2 = re.search(r"Values=([0-9.:;]+)", vc)
                if not m2:
                    continue
                pts = []
                for seg in m2.group(1).split(";"):
                    if ":" not in seg:
                        continue
                    t_, v_ = seg.split(":")
                    pts.append(f"{float(t_):.3f}:{max(0.02, float(v_) * _f):.2f}")
                parts = [x for x in parts if x is not vc] + [
                    vc.split("Values=")[0] + "Values=" + ";".join(pts)]
            else:
                parts.append(
                    "C_VALUECURVE_Brightness=Active=TRUE|Id=ID_VALUECURVE_Brightness|"
                    f"Type=Custom|Min=0.00|Max=400.00|RV=TRUE|"
                    f"Values=0.000:{VC_NOMINAL * _f:.2f};1.000:{VC_NOMINAL * _f:.2f}")
            eff.palette = pal.model_copy(update={"extra_parts": parts})
            _dimmed += 1
        if _dimmed:
            _probe("Ambient passages: dimmed")
            print(f"Ambient passages: dimmed {_dimmed} effects to "
                  f"{_DIM * 100:.0f}% so drops can sit above them")

        _slowed = 0
        for eff in _kept:
            if not any(a["t0"] <= eff.start_time_ms / 1000.0 < a["t1"]
                       for a in ambient_passages):
                continue
            st = eff.settings
            if "E_SLIDER_Shader_Speed" in st:
                # SLOW ONCE. Without the marker this pass could reach the same
                # effect twice (240 -> 72 -> 30), landing on the floor, and a
                # shader at speed 30 does not drift, it stands still. Measured
                # on The WhatsApp Shakespeare at 1:50: Contour Drift on every
                # matrix, the tree and the tunnel at Shader_Speed=30 -- "all
                # the matrices and tunnel and tree are just blue... there might
                # be some shape or movement that is supposed to be there but
                # it's not."
                #
                # The floor is also far too low. An ambient passage wants a
                # drift, not a freeze; 90 still reads as slow next to a base of
                # 240-400 while actually moving.
                v = int(st["E_SLIDER_Shader_Speed"])
                if v > 60 and st.get("X_AmbientSlowed") != "1":
                    eff.settings = {**st, "X_AmbientSlowed": "1",
                                    "E_SLIDER_Shader_Speed": str(max(90, int(v * 0.30)))}
                    _slowed += 1
            elif "E_TEXTCTRL_Chase_Rotations" in st:
                # A chase's rotation count is its speed. One rotation over a
                # multi-bar span is a drift; four is a pulse.
                eff.settings = {**st, "E_TEXTCTRL_Chase_Rotations": "0.5"}
                _slowed += 1
        if _slowed:
            _probe("Ambient passages: slowed")
            print(f"Ambient passages: slowed {_slowed} moving effects to a drift")
        _kept = _kept3
        effects = _kept
        _probe("Ambient passages: cleared")
        print(f"Ambient passages: cleared {_cut} rhythmic hits from "
              f"{len(ambient_passages)} smooth passages")

    # --- final dropout ----------------------------------------------------
    # After the last drum hit the band leaves and so does the show: everything
    # but the linear props is cleared, and those hold a dim, steady level.
    # "Something minimal but still large scale" -- outlines spanning the house
    # read as large without being bright, which a handful of lit props would
    # not.
    if final_dropout is not None:
        # On a downbeat, not 400ms after the hit: "the outlines turn blue but
        # not on rhythm". The change of state is itself a musical moment.
        _fd_s = next((d for d in downbeats if d >= final_dropout + 0.2),
                     final_dropout + 0.4)
        _fd = int(_fd_s * 1000)
        _keepers = {t["name"] for t in cast
                    if t["cat"] in ("single_line", "icicle")
                    or any(h in t["name"].lower()
                           for h in ("outline", "vertical", "horizontal", "c9"))}
        # Clear from the HIT, not from the downbeat after it. The outline hold
        # still begins on the downbeat, but everything else stops the moment
        # the drum lands -- otherwise the gap between them fills with the
        # flurry Krists describes: "we're hitting it but then also doing
        # something right after, but it should be just a slow fade." Measured
        # on This Version of You, 145 effects started between 154.5 and 160s,
        # clustered at 154.7 through 155.5.
        _cut_from = int(final_dropout * 1000) + 150
        _kept, _cut = [], 0
        for eff in effects:
            if eff.end_time_ms <= _cut_from:
                _kept.append(eff)
                continue
            base = eff.model_name.split("/")[0]
            if eff.model_name in _keepers or base in _keepers:
                _kept.append(eff)
                continue
            if eff.start_time_ms < _cut_from:   # trim rather than drop
                # ...and fade it out. A trimmed effect otherwise just stops:
                # "the big drop out at 2:33 is on the beat but then it just
                # turns completely off a second or two later without fading,
                # so it also seems random." Trimming is a structural decision;
                # the eye should still see a decay.
                # Extend PAST the cut and fade there, rather than ending at
                # it. A fade-out lives inside an effect's own span, so cutting
                # to _cut_from and adding one made everything dim BEFORE the
                # hit -- the moment measured 66.6 down to 34.9. Letting it run
                # on and decay puts the fade where the ear expects it: the hit
                # lands at full, then the house falls away.
                # AN IMMEDIATE CUT, not a slow decay. "It's supposed to be a
                # dramatic moment where everything cuts out immediately except
                # for the outlines" -- and previously everything ran on for
                # 1.6s with a 1.5s fade, which reads as the house sagging
                # rather than stopping.
                #
                # The old long tail existed for a real reason: a fade lives
                # INSIDE an effect's span, so ending at the cut and adding a
                # fade-out dimmed the approach and cost the hit itself (66.6
                # down to 34.9 when that was last tried). The fix is to end at
                # the cut with a fade SHORT enough that the dimming happens
                # after the hit lands -- 250ms, not 1.5s.
                eff.end_time_ms = _cut_from + 250
                # Marked so nothing lengthens it again. The shader-dwell
                # minimum (4s) ran after this and stretched the whole-house
                # bed from the 2.87s this leaves back out to a full 4s,
                # relighting the entire facade 1.1s into the very dropout that
                # had just cleared it -- which is why the last drum hit read as
                # lackluster: the house never actually fell away.
                eff.settings = {**eff.settings, "T_TEXTCTRL_Fadeout": "0.25",
                                "X_TrimmedAtDropout": "1"}
                _kept.append(eff)
                continue
            _cut += 1
        effects = _kept
        _dim_tail = PALETTES["ice"].model_copy(update={"extra_parts": [
            "C_VALUECURVE_Brightness=Active=TRUE|Id=ID_VALUECURVE_Brightness|"
            "Type=Custom|Min=0.00|Max=400.00|RV=TRUE|Values=0.000:0.11;1.000:0.11"]})
        _n_tail = 0
        for t in cast:
            if t["name"] in _keepers:
                add_target(t, L_BASE, "wash", _fd, int(_dur_s * 1000), _dim_tail)
                _n_tail += 1
        # THE TAIL STILL ANSWERS THE MUSIC. "Then we don't do anything for the
        # final few chord hits at 2:40." A flat hold to the end is minimal in
        # the wrong way -- This Version of You has chords at 157.2, 159.9,
        # 162.0, 163.3, 164.4 and 167.4 after its last drum hit, and the piece
        # is still playing. The outlines swell gently on each one, so the
        # ending is quiet without being inert.
        _tail_ch = ([c for c in voices["chord"].in_window(final_dropout,
                                                          _dur_s)]
                    if "chord" in voices else [])
        # 1.5s thinning dropped half of them -- This Version of You's closing
        # chords sit 1.1-1.3s apart, so "we're only showing two of the 4".
        _thin, _last_c = [], -9.0
        for _c in _tail_ch:
            if _c - _last_c >= 0.9:
                _thin.append(_c)
                _last_c = _c
        for _ci, _c in enumerate(_thin):
            _is_final = _ci == len(_thin) - 1
            # THE FINALE OPENS OUT. "That's the finale so it needs to be calm
            # but brighter, like all the outlines and maybe all the props
            # could come in on the final chord and very slowly everything
            # fades out." Every chord before it stays on the outlines and
            # grows a little; the last one brings the whole house in and then
            # fades to the end of the song.
            # "The final 4 chords are not really visible, there's just a faint
            # blue glow on the outlines but it needs to show the chords in a
            # contrasting color." They were ice blue at 50-100% of nominal,
            # sitting on top of a blue/grey tail -- present but unreadable.
            # Now each closing chord takes the accent hue, advancing per chord,
            # and starts ABOVE nominal so it is unmistakably an event.
            _grow = 1.1 + 0.5 * (_ci / max(len(_thin) - 1, 1))
            _tail_sec = next((sm.get("palette") for sm, sec
                              in zip(section_meta, analysis.sections)
                              if sec.start_time <= _c < sec.end_time
                              and isinstance(sm, dict)), None) or PALETTES["ice"]
            _pal_t = accent_bright(contrast_palette(_tail_sec, _ci),
                                   VC_NOMINAL * _grow)
            for t in cast:
                _in = t["name"] in _keepers
                if _is_final:
                    if prominence_of(t) < 2 and not _in:
                        continue
                elif not _in:
                    continue
                _end = (int(_dur_s * 1000) if _is_final
                        else int(_c * 1000) + 2200)
                # On the last chord the whole house opens out in white, but
                # the outlines keep the accent so the CHORD itself is still
                # legible inside the swell rather than washed away by it.
                add_target(t, L_MOTION, "wash", int(_c * 1000), _end,
                           (_pal_t if _in else PALETTES["white"])
                           if _is_final else _pal_t,
                           extra={"T_TEXTCTRL_Fadein": "0.20",
                                  "T_TEXTCTRL_Fadeout":
                                  f"{max(2.5, (_end / 1000.0 - _c) * 0.85):.1f}"
                                  if _is_final else "1.80"})
        if _thin:
            print(f"Final dropout: {len(_thin)} closing chords still answered "
                  f"by the outlines")
        _probe("Final dropout: cleared")
        print(f"Final dropout: cleared {_cut} effects after "
              f"{final_dropout:.1f}s, {_n_tail} outlines held dim to the end")

    # --- stage one of a double drop: outlines ON, everything else BACK -----
    # "The first drop should have all the outlines come on or maybe just the
    # snowflakes and canes or something, and then everything else on the
    # second drop."
    #
    # Adding light to the outlines alone was not enough -- measured at
    # Harmony's 3:18, stage one read 75 against 68 for the bar before it and
    # 144 at stage two, i.e. a 10% lift nobody would notice. What makes it a
    # moment is the CONTRAST: for that beat and a half the outlines are the
    # show and everything else steps back, which also leaves stage two
    # somewhere to go.
    if _stage1_times:
        _s1_keep = {t["name"] for t in cast
                    if t["cat"] in ("single_line", "icicle")
                    or any(h in t["name"].lower()
                           for h in ("outline", "vertical", "horizontal", "c9",
                                     "snowflake", "flake", "candy", "cane"))}
        _n_hushed = 0
        for _st1 in _stage1_times:
            a_ms, b_ms = int(_st1 * 1000), int(_st1 * 1000) + 1500
            for eff in effects:
                if eff.end_time_ms <= a_ms or eff.start_time_ms >= b_ms:
                    continue
                nm = eff.model_name
                if nm in _s1_keep or nm.split("/")[0] in _s1_keep:
                    continue
                if eff.layer == L_SIG:
                    continue                     # the drop staging itself
                pal = eff.palette
                if pal is None:
                    continue
                parts = [x for x in (pal.extra_parts or [])
                         if not x.startswith("C_VALUECURVE_Brightness=")]
                parts.append(
                    "C_VALUECURVE_Brightness=Active=TRUE|Id=ID_VALUECURVE_Brightness|"
                    "Type=Custom|Min=0.00|Max=400.00|RV=TRUE|Values=0.000:0.06;1.000:0.06")
                eff.palette = pal.model_copy(update={"extra_parts": parts})
                _n_hushed += 1
        if _n_hushed:
            _probe("Double-drop stage one")
            print(f"Double-drop stage one: hushed {_n_hushed} effects so the "
                  f"outlines carry it")

    # --- flood discipline -------------------------------------------------
    # "Across songs we're using the floods too much and strobing or pulsing
    # them too much again. They should really be used sparingly, for slow
    # washes, or for lighting up the house when many other lights are dimmed
    # for a quieter part for the contrast, or sometimes in a really high
    # energy section they can do the top/bottom or left/right flashing on the
    # beat but no faster than quarter notes."
    #
    # Enforced here rather than in the flood pass, for the same reason the
    # no-strobe rule ended up here: several passes can put light on a flood
    # and no single one of them can see the total. Three rules, in order --
    # nothing faster than a quarter note, nothing at all in a quiet passage
    # unless it is a slow wash (the contrast case, which wants a wash and not
    # a flash), and a hard cap on how many flood hits a song may contain.
    _floods = [n for n in {e.model_name for e in effects} if "flood" in n.lower()]
    if _floods:
        _beat_ms = 60000.0 / max(SONG_TEMPO, 1.0)
        # "Very high energy" has to be measured PER SONG. energy_level is not
        # comparable between tracks -- Aperture's median section reads 0.94
        # against Gooey's 0.48 -- so an absolute 0.75 meant "most of the song"
        # on one and "a third of it" on the other, and Aperture lost only 4
        # flood flashes while keeping 367. The top fifth of a song's own
        # sections is what "climax" means here.
        _sec_es = sorted(sec.energy_level for sec in analysis.sections)
        _flood_climax_floor = (_sec_es[int(len(_sec_es) * 0.80)]
                               if _sec_es else 1.0)
        _amb_spans = [(a["t0"], a["t1"]) for a in ambient_passages]
        _kept, _cut_fast, _cut_quiet = [], 0, 0
        _last: dict[str, int] = {}
        for eff in sorted(effects, key=lambda e: e.start_time_ms):
            nm = eff.model_name
            if "flood" not in nm.lower():
                _kept.append(eff)
                continue
            dur = eff.end_time_ms - eff.start_time_ms
            slow_wash = dur >= 1500 or float(
                (eff.settings or {}).get("T_TEXTCTRL_Fadeout", 0) or 0) >= 1.0
            in_quiet = any(a <= eff.start_time_ms / 1000.0 < b for a, b in _amb_spans)
            # SLOW AND GENTLE EVERYWHERE EXCEPT A REAL CLIMAX. "The floods do
            # a lot of flashing and strobing, no songs should do that, the
            # floods are to be used sparingly and gently except for very high
            # energy climax moments."
            #
            # The previous rules -- no faster than a quarter note, no flashing
            # in quiet passages -- were far too permissive: Aperture still
            # carried 367 flood effects under 1.2s, Harmony 159. A flood only
            # gets to flash where the music is at its most intense; anywhere
            # else it washes or it does not appear.
            _sec_e = next((sec.energy_level for sec in analysis.sections
                           if sec.start_time <= eff.start_time_ms / 1000.0
                           < sec.end_time), 0.0)
            if not slow_wash and _sec_e < _flood_climax_floor:
                _cut_quiet += 1
                continue
            if in_quiet and not slow_wash:
                _cut_quiet += 1
                continue
            if not slow_wash and eff.start_time_ms - _last.get(nm, -10 ** 9) < _beat_ms:
                _cut_fast += 1
                continue
            if not slow_wash:
                _last[nm] = eff.start_time_ms
            _kept.append(eff)
        # ...and a hard ceiling on flood FLASHES for the whole song. On a
        # dense track a fifth of the sections is still a lot of climax --
        # Aperture kept 258 short flood effects even after the per-section
        # gate. "Sparingly" has to mean a countable number, so the flashes are
        # thinned to an even spread across the song and the rest dropped.
        _FLOOD_FLASH_CAP = 48
        _fl_short = [e for e in _kept
                     if "flood" in e.model_name.lower()
                     and e.end_time_ms - e.start_time_ms < 1200]
        if len(_fl_short) > _FLOOD_FLASH_CAP:
            _fl_short.sort(key=lambda e: e.start_time_ms)
            _step = len(_fl_short) / _FLOOD_FLASH_CAP
            _keep_ids = {id(_fl_short[int(i * _step)])
                         for i in range(_FLOOD_FLASH_CAP)}
            _before = len(_kept)
            _kept = [e for e in _kept
                     if id(e) in _keep_ids or e not in _fl_short]
            _cut_quiet += _before - len(_kept)
        effects = _kept
        if _cut_fast or _cut_quiet:
            _probe("Floods: dropped")
            print(f"Floods: dropped {_cut_fast} faster than a quarter note, "
                  f"{_cut_quiet} others (quiet passages, or over the "
                  f"{_FLOOD_FLASH_CAP}-flash ceiling)")

    # --- flash-rate cap, per prop ----------------------------------------
    # "There are a few points where we flash/pulse the matrices too fast, and
    # it looks like strobing, where it's to 8th notes or something. There's
    # also at least one point where the floods strobe like that."
    #
    # The 2-4 flashes/second rule has been stated since early on, but it lived
    # as a guideline inside individual passes, so every new pass had to
    # remember it and any two passes could stack on the same prop and break it
    # between them. Measured: short bursts landing 200ms apart on one prop,
    # i.e. 5 Hz. This enforces it once, at the end, where the whole picture is
    # visible -- the only place the rule can actually be guaranteed.
    #
    # Per PROP, so a chase travelling across many props at speed is untouched;
    # what is capped is one prop flashing at the viewer.
    _MIN_FLASH_GAP_MS = 240             # ~4 flashes/second
    # A QUARTER NOTE, not a fixed 400ms. The rule Krists set is musical --
    # "the floods would never go faster than quarter or half notes" -- and 400ms
    # is faster than a quarter note on anything under 150 BPM, so on Gooey (86
    # BPM, 697ms) the cap was letting through exactly what it existed to stop.
    _FLOOD_FLASH_GAP_MS = max(400, int(60000.0 / max(analysis.beats.tempo, 60.0)))
    # A parent-level cap was tried and reverted: it dropped 2286 flashes
    # instead of 737 and removed none of the remaining fast runs, because
    # those turned out not to be flashes at all -- see the house-bed note in
    # shader_span_test.py.
    _dropped = set()
    # BOTH the submodel and its PARENT PROP.
    #
    # "Fence Matrix 2" and "Fence Matrix 2/Horizontals" are different names but
    # one physical matrix, so capping per name counted eight starts inside
    # 300ms as eight separate props and let them all through -- measured as a
    # 10.9 Hz strobe across three matrices in Lose My Cool. The eye sees the
    # matrix, not the submodel. A parent pass was tried once before and judged
    # useless, but that was measured when a third of the effects were
    # rendering black, so the comparison was meaningless.
    for key, gap in (("sub", _MIN_FLASH_GAP_MS),
                     ("parent", _MIN_FLASH_GAP_MS)):
        by_prop: dict[str, list] = defaultdict(list)
        for eff in effects:
            if id(eff) in _dropped:
                continue
            if eff.end_time_ms - eff.start_time_ms <= 400:
                nm = (eff.model_name if key == "sub"
                      else eff.model_name.split("/")[0])
                by_prop[nm].append(eff)
        # A flash boundary is ANY effect starting on the prop, not just a short
        # one. Sphere-3 Inner measured an 8 Hz run while every individual gap
        # passed this cap: three 0.28s hits spaced 380 and 650ms, landing on
        # top of a Shockwave and a longer On that started moments earlier. The
        # eye sees the prop change, not which layer changed it.
        #
        # Starts within 60ms of each other are ONE flash, not a strobe -- a
        # drop legitimately lights many layers of a prop at the same instant,
        # and that reads as a single hit.
        _all_starts: dict[str, list[int]] = defaultdict(list)
        for eff in effects:
            if id(eff) not in _dropped:
                _all_starts[eff.model_name if key == "sub"
                            else eff.model_name.split("/")[0]].append(
                                eff.start_time_ms)
        for nm, evs in by_prop.items():
            g = _FLOOD_FLASH_GAP_MS if "flood" in nm.lower() else gap
            _clusters: list[int] = []
            for _t in sorted(_all_starts.get(nm, [])):
                if not _clusters or _t - _clusters[-1] > 60:
                    _clusters.append(_t)
            last = -10 ** 9
            for eff in sorted(evs, key=lambda e: e.start_time_ms):
                # The nearest EARLIER flash on this prop from any layer.
                _prev = max((c for c in _clusters
                             if c < eff.start_time_ms - 60), default=-10 ** 9)
                if (eff.start_time_ms - last < g
                        or eff.start_time_ms - _prev < g):
                    _dropped.add(id(eff))
                else:
                    last = eff.start_time_ms
    if _dropped:
        effects = drop_effects(effects, _dropped, "flash-rate cap")
    _probe("Flash-rate cap")
    print(f"Flash-rate cap: dropped {len(_dropped)} flashes closer than "
          f"{_MIN_FLASH_GAP_MS}ms on the same prop")

    # --- inside a drop/fade cycle, the cycle IS the content ---------------
    # "There is a single kick drum for each drop and everything between that
    # and the next single kick drum is the slow fade."
    #
    # Adding the fade envelope to more props kept making the fade WORSE, and
    # the reason shows up in a per-prop reading at t=35s: the canvases were
    # correctly down at ~40% of nominal while the rest of the show -- beat
    # anchors, submodel stepping, gutter flashes, leaping arches, ambient
    # washes -- carried on at full level underneath, so each new fading layer
    # only added light on top of a floor that never moved. Five attempts at
    # shaping the fade all landed between 1.1x and 1.8x for that reason.
    #
    # In this passage the cycle hold and the hit that starts it are the whole
    # gesture, so everything else inside the cycle window comes out. The hit
    # itself (anything landing within 0.6s of a cycle start) stays.
    if drop_cycles:
        _c0 = drop_cycles[0][0]
        _c1 = drop_cycles[-1][1]
        _starts = [c[0] for c in drop_cycles]
        _kept, _cut = [], 0
        for eff in effects:
            s_s = eff.start_time_ms / 1000.0
            if not (_c0 <= s_s < _c1) or id(eff) in _cycle_ids:
                _kept.append(eff)
                continue
            if any(0.0 <= s_s - cs <= 0.6 for cs in _starts):
                _kept.append(eff)       # the hit that opens the cycle
                continue
            _cut += 1
        effects = _kept
        print(f"Drop/fade cycles: cleared {_cut} competing effects between "
              f"{_c0:.1f}s and {_c1:.1f}s")

    # --- rebuild beat-pulse curves for the FINAL span --------------------
    # A brightness value curve is stored with its times normalised 0-1 across
    # the effect's own span, so ANY pass that shortens an effect after the
    # curve was attached squeezes the whole pulse into the time that is left.
    # Three passes do exactly that -- blackout clipping, the run-up dip, and
    # the cohort lock -- and none of them knew about the curve.
    #
    # Measured on So Easy: a Plasma hold ended up 1300ms long still carrying a
    # curve built for a much longer span, with peaks every ~104ms. That is
    # 9.6 Hz on a whole matrix, which is exactly the "flash/pulse the matrices
    # too fast, it looks like strobing" report, and it is a migraine risk
    # rather than a taste question.
    #
    # Rebuilding from the final times is better than dropping the curve: the
    # effect still pulses, on the beats it actually spans. The depth is
    # recovered from the curve's own peak, since that is where it was encoded.
    _rebuilt = 0
    for eff in effects:
        # Cycle holds carry a HIT-AND-DECAY envelope, not a beat pulse. This
        # pass would read that curve, recover a "depth" from its peak, and
        # replace it with a pulse -- silently destroying the fade. That is why
        # five successive attempts at shaping the intro fade measured between
        # 1.1x and 1.8x no matter what was changed upstream: the shape was
        # being written correctly and then overwritten here.
        # Cycle holds carry a hit-and-decay envelope and ambient holds carry
        # the section's own energy contour. Both are shapes, not pulses, and
        # this pass would read their peak as a "depth" and replace them with a
        # beat pulse -- the same way it silently destroyed the drop/fade
        # envelope until that was exempted.
        if id(eff) in _cycle_ids or is_ambient_eff(eff):
            continue
        pal = eff.palette
        if pal is None or not pal.extra_parts:
            continue
        vc = next((x for x in pal.extra_parts
                   if x.startswith("C_VALUECURVE_Brightness=")), None)
        if vc is None or "Type=Custom" in vc is False:
            continue
        m = re.search(r"Values=([0-9.:;]+)", vc)
        if not m:
            continue
        try:
            vals = [float(p.split(":")[1]) for p in m.group(1).split(";") if ":" in p]
        except (ValueError, IndexError):
            continue
        if not vals:
            continue
        depth = max(0.0, min(1.0, (max(vals) / VC_NOMINAL - 1.0) / 0.55))
        base = pal.model_copy(update={
            "extra_parts": [x for x in pal.extra_parts if x is not vc]})
        new = beat_pulse_palette(base, beats, eff.start_time_ms,
                                 eff.end_time_ms, depth)
        if new is not base or True:
            eff.palette = new
            _rebuilt += 1
    _probe("Pulse curves rebuilt")
    print(f"Pulse curves rebuilt for final spans: {_rebuilt}")

    # =====================================================================
    # Shader dwell -- the last thing before the write, so it sees the spans
    # as they will actually render rather than as any one pass intended them.
    #
    # "At :48 there's a shader and then a new one at :50 and that turns off at
    # :51 and another one turns on at :52, it's messy and not on the beat of
    # the piano chords. At 2:14 there's another shader that starts on the beat
    # and a second later turns off and a different one comes in, and same with
    # the final 2:40 moment."
    #
    # Measured on v13: 35% of shader spans were under four seconds, 29 of them
    # separated by a sub-2.5s hole, and 13 places had two different shaders
    # running on two layers of the SAME prop within a second of each other.
    # Every pass individually looked reasonable; the mess was emergent, so it
    # can only be fixed here, where the whole timeline is visible.
    _fd_cut = (int(final_dropout * 1000) + 150
               if final_dropout is not None else 0)
    _SHADER_MIN_DWELL_MS = 4000
    _SHADER_GAP_MS = 2500
    # A shader change is itself a musical event, so it belongs ON a chord
    # rather than near one -- "it's messy and not on the beat of the piano
    # chords". Anchors are the chord/bass hits the ambient analysis already
    # found plus the section boundaries; a start within 600ms of one snaps to
    # it, which is close enough to be the same intended moment and far enough
    # not to drag a start across to a different chord.
    # Section boundaries are NOT on the beat. Measured on Gooey: the detected
    # section starts sit a median of 187ms from the nearest beat, 95% of them
    # more than 40ms off, the worst 840ms off. Snapping a shader onto the raw
    # section time therefore pulled it OFF the beat -- "the shaders change
    # slightly off the main beat or section changes, seems like it's often a
    # little bit after the section change."
    #
    # The generator already snaps downbeats onto the beat grid for exactly this
    # reason; this pass was bypassing that and undoing it. Section anchors are
    # now snapped to the nearest beat first. Chord anchors are left alone --
    # those are measured onsets, so they ARE where the music is.
    _bt = np.asarray(analysis.beats.beat_times, dtype=float)

    def _on_beat(x: float) -> float:
        if _bt.size == 0:
            return x
        return float(_bt[int(np.argmin(np.abs(_bt - x)))])

    def _on_hit_beat(x: float) -> float:
        """A section boundary belongs on the hit the band lands on.

        The section detector reports where its analysis window crosses, which
        can sit BETWEEN two beats -- and `nearest beat` then rounds to
        whichever is closer, which is a coin flip. Measured on Gooey: the
        section at 15.46s sits exactly between beats at 15.16 and 15.86,
        rounds DOWN to 15.16, and the kick and bass land at 15.85. "The second
        big drum hit in the beginning around 0:16 is off -- the shaders change
        before the kick/bass hits."

        A whole beat early, and early reads far worse than late: the change
        pre-empts the thing it exists to mark. So if a drum or bass onset lies
        within about a beat after the boundary, that onset IS the boundary.
        """
        _beat = 60.0 / max(analysis.beats.tempo, 60.0)
        _lo, _hi = x - 0.12, x + 1.15 * _beat
        _cands: list[float] = []
        for _vn in ("kick", "snare", "bass"):
            _v = voices.get(_vn)
            if _v is not None:
                _cands += [float(t) for t in _v.onsets if _lo <= t <= _hi]
        # Already on a hit -> already right. Only reach forward when the
        # boundary has nothing under it; otherwise this pushes a correct time
        # onto the next hit instead.
        if any(abs(t - x) <= 0.15 for t in _cands):
            return _on_beat(x)
        return _on_beat(min(_cands)) if _cands else _on_beat(x)


    # DROPS OUTRANK EVERYTHING. A drop is the moment the whole look is
    # supposed to change, and a shader arriving even a beat late reads as a
    # mistake in a way that a slightly-off section change does not. Measured on
    # Gooey: the energy drops sit at 59.93s and 181.32s -- exactly the 1:00 and
    # 3:01 Krists named -- while the shader changed at 62.27s and 183.11s,
    # 2.3s and 1.8s late. That is far outside the 600ms window the chord and
    # section anchors use, so no amount of tightening those would have caught
    # it; drops need their own, much wider pull.
    # The SNAPPED drop times, not the raw detections. The detector reports the
    # rise; the pass above already works out where the band actually lands, and
    # anchoring on the raw time pulled the house bed onto a moment 2.7s before
    # the hit -- lighting the hush the drop exists to break.
    # BOTH the raw detection and the snapped hit. They answer different
    # questions and either can be the moment the eye picks: on Gooey the
    # detection beat-snaps to 60.00, which is the 1:00 Krists hears, while the
    # hit-snapper prefers 60.67; on This Version of You the detection is at
    # 74.37 and only the snapped 77.10 is the actual hit. Offering both lets
    # each shader change take whichever is nearer, instead of one song's
    # correct answer overriding the other's.
    # Hit-corrected, like the section anchors. A drop anchor sitting a beat
    # before the band lands is worse than no anchor at all, because the pull is
    # 3.2s wide and drags the change with it: Gooey's intro cycle was placed at
    # 15.75, correctly moved to the kick at 15.86 by the cohort lock, and then
    # yanked back to 15.16 by an anchor derived from a minor drop detected in
    # the run-up. Every anchor now names the same instant the music does.
    # Drops keep `_on_beat`: they have ALREADY been through _snap_drop, which
    # does its own "first strong hit after the detection" search. Running the
    # section-level correction over them as well double-corrects and pushes a
    # correct drop onto the NEXT hit -- Gooey's 3:01 moved from 181.37 to
    # 182.07 and split the canvases across two times.
    _drop_anchors = sorted({_on_beat(_t) for _t in SNAPPED_DROPS if _t}
                           | {_on_beat(_t) for _t in MINOR_DROPS if _t}
                           | {_on_beat(_t)
                              for _k in ("energy_drops", "predrop_cutouts")
                              for _ev in (cues.get(_k) or [])
                              for _t in ((_ev.get("drop_time")
                                          or _ev.get("time")),)
                              if _t})
    # 3.2s, not 2.5. A shader starting just before a drop lights the hush the
    # drop is supposed to break, and the hush is most of what makes the hit
    # land. Measured on This Version of You: the house bed began at 74.58s,
    # exactly 2.52s before the 77.10s hit -- twenty milliseconds outside the
    # old window -- and the hit's rise over its own run-up fell from x1.38 to
    # x1.15. Anything inside about two bars of a drop belongs ON the drop.
    _DROP_PULL_S = 3.2

    def _snap_start(ms: int) -> int:
        """Pull a start onto a drop, else onto the beat grid."""
        _x = ms / 1000.0
        _d = min(_drop_anchors, key=lambda a: abs(a - _x), default=None)
        if _d is not None and abs(_d - _x) <= _DROP_PULL_S:
            return int(_d * 1000)
        _b = _on_beat(_x)
        return int(_b * 1000) if abs(_b - _x) <= 0.25 else ms

    _anchors = sorted(
        [t for _ap in ambient_passages for t in _ap.get("sparse", [])]
        + [_on_hit_beat(sec.start_time) for sec in analysis.sections])
    _snapped = 0
    if _anchors:
        import bisect as _bisect
        for _e in effects:
            if _e.effect_name != "Shader":
                continue
            _t = _e.start_time_ms / 1000.0
            # A drop within _DROP_PULL_S wins outright.
            _dnear = min(_drop_anchors, key=lambda x: abs(x - _t), default=None)
            if _dnear is not None and abs(_dnear - _t) <= _DROP_PULL_S:
                _ms = int(_dnear * 1000)
                if _ms < _e.end_time_ms - 400:
                    _e.start_time_ms = _ms
                    _snapped += 1
                    continue
            # The hit guard runs AFTER the drop pull, not before it. A drop
            # outranks everything -- that is the whole point of the wider
            # pull -- and guarding first stopped Gooey's 3:01 change from
            # reaching the drop at all, because the placement it started
            # from happened to sit on an unrelated onset.
            # ALREADY ON A HIT? THEN LEAVE IT. Every anchor here is an
            # approximation of where the music turns, and snapping a start
            # that already sits on a drum or bass onset can only move it off
            # one. Measured on Gooey: the intro's second cycle was correctly
            # placed on the kick at 15.86, and a chord anchor at 15.35 --
            # itself rounded to the beat at 15.16 -- dragged it back a whole
            # beat, so the shaders changed before the kick they were marking.
            _on_hit_already = False
            for _vn in ("kick", "snare", "bass"):
                _v = voices.get(_vn)
                # 0.15s, not 0.06. The onset detectors and the beat grid do
                # not agree to the millisecond -- on Gooey the bass lands at
                # 15.93 while the beat marking it is 15.86, 70ms apart -- and a
                # tolerance tighter than that disagreement makes the guard
                # useless exactly where it matters.
                if _v is not None and any(abs(_o - _t) <= 0.15 for _o in _v.onsets):
                    _on_hit_already = True
                    break
            if _on_hit_already:
                continue
            _j = _bisect.bisect_left(_anchors, _t)
            _best = min((_anchors[k] for k in (_j - 1, _j)
                         if 0 <= k < len(_anchors)),
                        key=lambda x: abs(x - _t), default=None)
            if _best is not None and abs(_best - _t) <= 0.6:
                _ms = int(_on_beat(_best) * 1000)
                if _ms < _e.end_time_ms - 400:
                    _e.start_time_ms = _ms
                    _snapped += 1
                    continue
            # Not near a chord or a section, but still a hair off the grid:
            # a dozen passes compute their own times and the rounding drifts.
            # Moving a start by under an eighth of a beat can only improve
            # alignment, and a shader change landing ON the beat is most of
            # what makes it read as intentional.
            _nb = _on_beat(_t)
            if 0 < abs(_nb - _t) <= 0.12:
                _ms = int(_nb * 1000)
                if _ms < _e.end_time_ms - 400:
                    _e.start_time_ms = _ms
                    _snapped += 1

    _by_prop: dict[str, list] = {}
    for _e in effects:
        if _e.effect_name == "Shader":
            _by_prop.setdefault(_e.model_name, []).append(_e)
    _drop: set[int] = set()
    _stacked = _merged = _closed = 0
    for _mn, _lst in _by_prop.items():
        _lst.sort(key=lambda e: (e.start_time_ms, e.layer))
        # (a) two shaders on different layers of one prop, starting together:
        #     the eye reads this as the shader "changing" instantly. Keep the
        #     lower layer, which is the one carrying the section's identity.
        for _i, _a in enumerate(_lst):
            for _b in _lst[_i + 1:]:
                if _b.start_time_ms - _a.start_time_ms > 1500:
                    break
                if _b.layer != _a.layer and _b.start_time_ms < _a.end_time_ms:
                    _lose = _a if _a.layer > _b.layer else _b
                    if id(_lose) not in _drop:
                        _drop.add(id(_lose))
                        _stacked += 1
        for _lay in {e.layer for e in _lst}:
            _row = [e for e in _lst if e.layer == _lay and id(e) not in _drop]
            _row.sort(key=lambda e: e.start_time_ms)
            _i = 0
            while _i < len(_row):
                _e = _row[_i]
                _nxt = _row[_i + 1] if _i + 1 < len(_row) else None
                _span = _e.end_time_ms - _e.start_time_ms
                if (_span < _SHADER_MIN_DWELL_MS and not is_ambient_eff(_e)
                        and _e.settings.get("X_TrimmedAtDropout") != "1"):
                    # Too brief to register as anything but a blink. Prefer to
                    # LENGTHEN it into the hole ahead of it; only drop it if
                    # there is genuinely no room, so a deliberate accent on a
                    # hit survives as a longer version of itself.
                    _room = (_nxt.start_time_ms if _nxt else _e.end_time_ms + 99999)
                    # NEVER LENGTHEN ACROSS THE FINAL DROPOUT. The whole-house
                    # bed ended at 154.97 -- exactly the dropout cut -- and this
                    # rule stretched it to its 4s minimum at 157.68, relighting
                    # the entire facade straight through the last drum hit. The
                    # minimum dwell exists so a shader is not a blink; it must
                    # not override a deliberate clearing.
                    if _fd_cut and _e.start_time_ms < _fd_cut:
                        _room = min(_room, _fd_cut)
                    if _room - _e.start_time_ms >= _SHADER_MIN_DWELL_MS:
                        _e.end_time_ms = min(_room,
                                             _e.start_time_ms + _SHADER_MIN_DWELL_MS)
                        _merged += 1
                    elif _i > 0:
                        _prev = _row[_i - 1]
                        _prev.end_time_ms = max(_prev.end_time_ms, _e.end_time_ms)
                        _drop.add(id(_e))
                        _merged += 1
                        _row.pop(_i)
                        continue
                # (c) close the holes: a shader going dark for under 2.5s and
                #     being replaced reads as a glitch, not as a change.
                # ...but never across a deliberate clearing. An effect the
                # final dropout trimmed has been cut ON PURPOSE, and closing
                # the gap after it just reinstates what was cleared -- this is
                # what kept the whole-house bed alight through the last drum
                # hit after the minimum-dwell rule had already been stopped
                # from doing the same thing.
                if (_nxt is not None
                        and 0 < _nxt.start_time_ms - _e.end_time_ms <= _SHADER_GAP_MS
                        and _e.settings.get("X_TrimmedAtDropout") != "1"):
                    _e.end_time_ms = _nxt.start_time_ms
                    _closed += 1
                _i += 1
    if _drop:
        effects = drop_effects(effects, _drop, "shader dwell: stacked or too-short shader")
    _probe("shader dwell (snap+dwell)")
    print(f"Shader dwell: {_snapped} starts snapped to a chord or section, "
          f"{_stacked} stacked shaders removed, {_merged} short "
          f"spans lengthened or merged, {_closed} gaps closed "
          f"(min dwell {_SHADER_MIN_DWELL_MS / 1000:.0f}s)")

    _probe("before write")


    # NOTHING IS PLACED AFTER THE FINAL DROPOUT. That pass clears the house to
    # leave only the outlines, and any guard that adds effects afterwards
    # undoes it -- which is what kept happening: "the final dropout at 2:33-2:44
    # in TVOY is still wrong, the whole house stays really bright after all the
    # instruments drop out." Measured there: the level fell to 19.7 at 2:35 and
    # climbed back to 57 for the next five seconds, because the canvas fill and
    # the section marker both ran later and refilled what it had cleared.
    _no_add_after = (int(final_dropout * 1000) + 150
                     if final_dropout is not None else None)

    def _after_dropout(ms: int) -> bool:
        return _no_add_after is not None and ms >= _no_add_after


    def _g_fill_flat_canvases(effects):
        # A CANVAS MAY BE DARK. IT MAY NOT BE A FLAT WASH OF ONE COLOUR.
        #
        # "At 2:11 much of the house is just blue, it's a bit too much and too
        # simple, something might be missing... at 2:39-2:46 the matrices are all
        # just blue." Measured on The WhatsApp Shakespeare: every large canvas --
        # all seven matrices, the Mega Tree and the Garage Roof -- had a 44 SECOND
        # stretch carrying no effect of its own, relying entirely on the
        # whole-house spanning shader. A span gives each prop a small crop of one
        # canvas, and a crop of a slow field is nearly featureless, so the props
        # render as flat colour.
        #
        # This is not new -- the same hole is in builds from before this round --
        # but it only became visible once the brightness-curve fix made everything
        # actually render.
        #
        # The first version of this filled any canvas carrying no effect of its
        # own, which was wrong: "canvases can be left bare when in an ambient or
        # quieter passage or before a drop, but not filled with a single basic
        # color and nothing else -- the problem was that they were just blue, not
        # that they were off at some appropriate moments."
        #
        # Being dark IS the effect in a hush, and refilling those moments cost
        # This Version of You most of its shape: the first big drum hit fell from
        # x1.38 over its run-up to x1.15, and the final dropout -- which clears
        # everything but the outlines -- went from leaving 30% of the pre-dropout
        # level to leaving 83%, because this pass ran afterwards and put the
        # canvases straight back.
        #
        # So the trigger is being LIT AND FEATURELESS, not being empty: a canvas
        # covered by the whole-house span (which hands each prop a near-constant
        # crop) while having nothing of its own. A canvas covered by nothing at all
        # is dark on purpose and is left alone.
        #
        # One shader per hole, shared by every canvas in it, so they still read as
        # doing the same thing.
        # Two thresholds. A canvas bare for more than 6s is filled outright; a
        # shorter gap is filled only when most of the OTHER big canvases are
        # textured at that moment, because a matrix sitting flat while its
        # neighbours have texture is the thing Krists flagged separately -- "I
        # thought we made it so the matrices generally did the same thing".
        _BARE_MIN_S = 3.0
        _BARE_ALONE_S = 6.0
        # Defined by WHAT THE PROP IS, not by the `large` flag. That flag missed
        # the three front window matrices -- which is why they stayed flat at 2:42
        # while every other canvas showed Fracture -- and included Floods and Orbs,
        # single-pixel props that cannot hold a texture at all.
        # Synthetic groups are INCLUDED. "Front Window Matrices" is synthetic --
        # a pseudo-group that add_target broadcasts to its members -- and skipping
        # it left those three matrices flat at 2:42 while the other four showed
        # Fracture. Coverage is tracked by the names effects actually land on,
        # which for a synthetic target are the members, not the group.
        _big_targets = [t for t in cast
                        if t["name"] != "ALL Display"
                        and not _is_flood(t)
                        and (t.get("cat") == "matrix"
                             or t["name"] in CONE_SHAPED
                             # "matri" not "matrix": the three front window
                             # matrices are addressed through a GROUP called
                             # "Front Window Matrices", and the plural does not
                             # contain the singular. That one letter is why they
                             # stayed flat at 2:42 through two attempted fixes.
                             or "matri" in t["name"].lower()
                             or t["name"] == "Garage Roof")]
        def _covers(t: dict) -> list[str]:
            return list(t.get("members") or []) if t.get("synthetic") else [t["name"]]

        _big_names = {n for t in _big_targets for n in _covers(t)}
        _own_by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for _e in effects:
            if _e.model_name in _big_names:
                _own_by_name[_e.model_name].append((_e.start_time_ms, _e.end_time_ms))

        def _n_textured(ms: int) -> int:
            return sum(1 for _n, _sp in _own_by_name.items()
                       if any(a <= ms < b for a, b in _sp))

        _span_cover = sorted((e.start_time_ms, e.end_time_ms) for e in effects
                             if e.model_name == HOUSE_GROUP)

        def _lit_by_span(a: int, b: int) -> bool:
            """Is this stretch mostly covered by the whole-house span?"""
            if not _span_cover:
                return False
            hit = sum(max(0, min(b, y) - max(a, x)) for x, y in _span_cover)
            return hit >= 0.6 * max(b - a, 1)

        _filled = 0
        _hole_key: dict[int, str] = {}
        for _t in _big_targets:
            _mine = set(_covers(_t))
            _own = sorted((e.start_time_ms, e.end_time_ms) for e in effects
                          if e.model_name in _mine)
            _merged: list[list[int]] = []
            for _a, _b in _own:
                if _merged and _a <= _merged[-1][1]:
                    _merged[-1][1] = max(_merged[-1][1], _b)
                else:
                    _merged.append([_a, _b])
            _cursor = int(_merged[0][0]) if _merged else 0
            _holes = []
            for _a, _b in _merged:
                if _a - _cursor >= _BARE_MIN_S * 1000:
                    _holes.append((_cursor, _a))
                _cursor = max(_cursor, _b)
            for _h0, _h1 in _holes:
                if any(_h0 < _wb * 1000 and _wa * 1000 < _h1
                       for _wa, _wb in blackout_windows):
                    continue
                if not _lit_by_span(_h0, _h1):
                    continue            # dark on purpose -- leave it dark
                if (_h1 - _h0) < _BARE_ALONE_S * 1000 and \
                        _n_textured((_h0 + _h1) // 2) < len(_big_targets) // 2:
                    continue
                _si = next((i for i, sec in enumerate(analysis.sections)
                            if sec.start_time <= _h0 / 1000.0 < sec.end_time), 0)
                _sm = section_meta[_si] if _si < len(section_meta) else {}
                _pal = (_sm.get("palette") if isinstance(_sm, dict) else None) \
                    or PALETTES["ice"]
                _bucket = _h0 // 4000
                if _bucket not in _hole_key:
                    _hole_key[_bucket] = (rng.choice(SHADER_KEYS) if SHADER_KEYS
                                          else "wash")
                # The fill runs AFTER the shader-dwell snap, so it has to place
                # itself on the grid rather than rely on being corrected later.
                # Measured on Gooey: this pass was the thing putting the 1:00
                # shader change at 62.27s -- 2.3s past the drop -- while the drop
                # itself had been snapped correctly everywhere else.
                # NEVER EARLIER. Snapping exists so a fill lands on the beat,
                # but pulling one BACKWARDS puts it inside a moment already
                # covered -- and the moment it reaches for is a drop, which is
                # exactly where a canvas should show the hit and nothing else.
                # Measured on Gooey's 1:00 drop: the hole began at 62.27,
                # snapped back to 60.67, and laid a shader under the white drop
                # hit. The matrices read 238 there before this pass existed and
                # 86 after it.
                _s0 = _snap_start(_h0)
                if _s0 < _h0 or _s0 >= _h1 - 500:
                    _s0 = _h0
                add_target(_t, L_BASE, _hole_key[_bucket], _s0, _h1, _pal)
                _filled += 1
        if _filled:
            print(f"Bare-canvas fill: {_filled} stretches longer than "
                  f"{_BARE_MIN_S:.0f}s given the section's own texture "
                  f"(they were showing only the whole-house span)")

        # ...and a second sweep for cohesion, AFTER the fills above, so it sees the
        # canvas as it will actually render. "I thought we made it so the matrices
        # generally did the same thing": wherever most big canvases carry a
        # texture, the rest do too. The gap-based pass alone left the three front
        # window matrices flat at 2:42 while every other canvas showed Fracture,
        # because their individual gaps were short and the coverage count it used
        # was taken before its own fills landed.
        _cover: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for _e in effects:
            if _e.model_name in _big_names:
                _cover[_e.model_name].append((_e.start_time_ms, _e.end_time_ms))
        _step = 500
        _end_ms = int(_dur_s * 1000)
        _need: dict[str, list[int]] = defaultdict(list)
        for _ms in range(0, _end_ms, _step):
            _lit = {n for n, sp in _cover.items() if any(a <= _ms < b for a, b in sp)}
            if len(_lit) < 0.6 * len(_big_targets) or not _lit:
                continue
            if any(_ms < _wb * 1000 and _wa * 1000 < _ms + _step
                   for _wa, _wb in blackout_windows):
                continue
            for _t in _big_targets:
                if not (_lit & set(_covers(_t))):
                    _need[_t["name"]].append(_ms)
        _cohered = 0
        for _t in _big_targets:
            _slots = _need.get(_t["name"])
            if not _slots:
                continue
            _runs: list[list[int]] = []
            for _ms in _slots:
                if _runs and _ms - _runs[-1][1] <= _step:
                    _runs[-1][1] = _ms
                else:
                    _runs.append([_ms, _ms])
            for _a, _b in _runs:
                if _b - _a < 2000 or _after_dropout(_a):
                    continue
                if not _lit_by_span(_a, _b + _step):
                    continue
                _si = next((i for i, sec in enumerate(analysis.sections)
                            if sec.start_time <= _a / 1000.0 < sec.end_time), 0)
                _sm = section_meta[_si] if _si < len(section_meta) else {}
                _pal = (_sm.get("palette") if isinstance(_sm, dict) else None) \
                    or PALETTES["ice"]
                _bucket = _a // 4000
                if _bucket not in _hole_key:
                    _hole_key[_bucket] = (rng.choice(SHADER_KEYS) if SHADER_KEYS
                                          else "wash")
                _a2 = _snap_start(_a)
                if _a2 < _a or _a2 >= _b:
                    _a2 = _a
                add_target(_t, L_BASE, _hole_key[_bucket], _a2, _b + _step, _pal)
                _cohered += 1
        if _cohered:
            print(f"Canvas cohesion: {_cohered} stretches where one big prop was "
                  f"flat while the rest were textured")
        return effects

    def _g_bars_discipline(effects):
        """Bars land on the beat, last a bar, and fade rather than stop.

        "We're often using the diagonal bars effect very briefly, like on a
        section change it comes on for a second or two and then turns off...
        it's awkward how it turns off suddenly, and it often starts off the
        beat too. Let's make sure we're only doing that sparingly and make sure
        it's on the beat and either lasts longer or fades out."

        Measured on Gooey before this: 369 Bars, only 48% within 40ms of a
        beat, a tenth of them under 0.25s, and 359 of 369 with NO fade-out at
        all -- they simply stopped.
        """
        _bar_ms = int(4 * 60000.0 / max(analysis.beats.tempo, 60.0))
        _bt = np.asarray(analysis.beats.beat_times, dtype=float)
        _snapped = _lengthened = _faded = _dropped = 0
        _keep = []
        _last_by_prop: dict[str, int] = {}
        for _e in sorted(effects, key=lambda x: x.start_time_ms):
            # SPANNING CHASES COUNT AS BARS HERE. What Krists calls "the
            # diagonal bars" is not always the Bars effect: a SingleStrand on
            # a Per Preview buffer stretches one chase across the whole facade
            # and renders as broad diagonal stripes sweeping past. Measured at
            # This Version of You 0:58 -- a 1.44s spanning chase on All Yard
            # Props, the Arches and the C9s, with no fade-out, which stops
            # dead. "I don't like that bars for 1 second effect and it's used
            # way too often." The Bars-only filter never saw it.
            _spanning_chase = (_e.effect_name == "SingleStrand"
                               and _e.settings.get("B_CHOICE_BufferStyle")
                               == "Per Preview")
            if _e.effect_name != "Bars" and not _spanning_chase:
                _keep.append(_e)
                continue
            # On the beat.
            if _bt.size:
                _t = _e.start_time_ms / 1000.0
                _nb = float(_bt[int(np.argmin(np.abs(_bt - _t)))])
                if 0 < abs(_nb - _t) <= 0.30 and int(_nb * 1000) < _e.end_time_ms - 200:
                    _e.start_time_ms = int(_nb * 1000)
                    _snapped += 1
            # SPARINGLY: one per prop per bar. A second bars sweep arriving on
            # the same prop before the first has had a bar to read is what
            # makes them feel like flicker rather than a gesture.
            # "In general we seem to be using the diagonal bars changing
            # orientation effect way too much, it should just be one of many
            # effects we use sparingly for variety." One per prop per four
            # bars rather than per bar.
            _prev = _last_by_prop.get(_e.model_name)
            if _prev is not None and _e.start_time_ms - _prev < 4 * _bar_ms:
                _dropped += 1
                continue
            _last_by_prop[_e.model_name] = _e.start_time_ms
            # A spanning chase at a drop is MEANT to be a quick gesture, so it
            # keeps its length and gets a decay instead -- "either lasts longer
            # or fades out". Only the Bars effect is stretched.
            if _spanning_chase:
                if "T_TEXTCTRL_Fadeout" not in _e.settings:
                    _sp = (_e.end_time_ms - _e.start_time_ms) / 1000.0
                    _e.settings = {**_e.settings,
                                   "T_TEXTCTRL_Fadeout": f"{min(1.2, _sp * 0.5):.2f}"}
                    _faded += 1
                _keep.append(_e)
                continue
            # TWO bars, not one, and end on a section boundary when one is
            # near. "There are diagonal bars changing orientation to the beat
            # but only for one second and then they go away... it's jarring
            # when they start and stop so quickly. Let's stay until a
            # section/bar or fade out." One bar is under two seconds at these
            # tempos, which is what he is calling one second.
            if _e.end_time_ms - _e.start_time_ms < 2 * _bar_ms:
                _want = _e.start_time_ms + 2 * _bar_ms
                _next_sec = next((int(sec.start_time * 1000)
                                  for sec in analysis.sections
                                  if sec.start_time * 1000 > _e.start_time_ms + 800),
                                 None)
                if _next_sec is not None and \
                        abs(_next_sec - _want) < _bar_ms:
                    _want = _next_sec
                _e.end_time_ms = _want
                _lengthened += 1
            # And a decay, so it leaves rather than stops.
            if "T_TEXTCTRL_Fadeout" not in _e.settings:
                _span_s = (_e.end_time_ms - _e.start_time_ms) / 1000.0
                _e.settings = {**_e.settings,
                               "T_TEXTCTRL_Fadeout": f"{min(1.6, _span_s * 0.45):.2f}"}
                _faded += 1
            _keep.append(_e)
        if _snapped or _dropped or _lengthened or _faded:
            print(f"Bars discipline: {_snapped} snapped to the beat, "
                  f"{_lengthened} lengthened to two bars, {_faded} given a fade, "
                  f"{_dropped} dropped as too soon after another on the same prop")
        return _keep

    def _g_beat_lock_motion(effects):
        """Tie repeating motion to the beat grid, from each effect's FINAL span.

        This is the mechanism Krists's own sequences use and this generator
        did not. `E_TEXTCTRL_Bars_Cycles` is the number of times the bar
        crosses WITHIN the effect's own span -- so a bar that moves on the beat
        is simply cycles = span-in-beats, and one that moves on the half note
        is span-in-beats / 2. His Bars carry Cycles of 16.0, 2, 1 and 0.5
        against this generator's hardcoded 1, which is why ours drift across a
        span instead of stepping with the music.

        Computing it here, at the end, means it uses the span the effect
        actually ends up with -- the same reason the pulse ceiling lives here.

        The unit ALTERNATES by bar: beat, beat, half-note, beat. "We had an
        effect before where the diagonal bars moved fast and then slowed down
        on the next beat, or the speed changed back and forth on the half
        notes." That alternation is the effect; a constant rate is just a rate.
        """
        _beat_s = 60.0 / max(analysis.beats.tempo, 60.0)
        _bar_s = 4 * _beat_s

        _big_names = {t["name"] for t in cast
                      if t.get("cat") == "matrix" or t["name"] in CONE_SHAPED
                      or "matri" in t["name"].lower()
                      or t["name"] == "Garage Roof"}

        def _repeat_unit(model: str) -> float:
            """Beats per repetition, by what the prop is.

            One crossing per beat is right on a matrix, where the bar travels
            across a wide surface and the eye follows it. On a small prop --
            a spoke group, a hook, an orb -- the whole thing lights and dims
            with each crossing, so a per-beat rate is not movement, it is
            flashing. Measured: 17-second Bars effects on the Grand Illusion
            submodel groups were given 35 crossings and read as a hard strobe.
            "In general we seem to have added a lot of flashing and strobing."
            """
            return 1.0 if model.split("/")[0] in _big_names else 4.0

        # An absolute ceiling as well as a musical one. However long an effect
        # runs, past a dozen repetitions it is a texture, not a gesture.
        _MAX_REPEATS = 12
        _n = 0
        for _e in effects:
            if _e.effect_name != "Bars":
                continue
            _span = (_e.end_time_ms - _e.start_time_ms) / 1000.0
            if _span < _beat_s:
                continue
            _beats = _span / _beat_s
            _bar_i = int((_e.start_time_ms / 1000.0) / _bar_s)
            _base = _repeat_unit(_e.model_name)
            # The alternation stays, but around the prop's own unit.
            _unit = _base * (1.0, 1.0, 2.0, 1.0)[_bar_i % 4]
            _cycles = max(1.0, min(round(_beats / _unit), _MAX_REPEATS))
            _new = {**_e.settings, "E_TEXTCTRL_Bars_Cycles": f"{_cycles:.1f}"}
            # DIRECTION AND ORIENTATION CHANGE ON THE BEAT TOO. "The bars can
            # also change direction on the beat, and they can also change
            # orientation." xLights' Alternate Up/Down reverses on every
            # cycle, so once the cycle rate is the beat, the bar reverses on
            # the beat -- the two halves only work together.
            #
            # Only for bars living on ONE prop (Per Model Default). A spanning
            # sweep across the facade is a directional gesture whose whole
            # point is going one way, and alternating it would destroy that.
            if _new.get("B_CHOICE_BufferStyle") == "Per Model Default":
                _d = _new.get("E_CHOICE_Bars_Direction", "")
                _pick = random.Random(f"bars|{_e.model_name}|{_bar_i}").random()
                if _d in ("up", "down") and _pick < 0.25:
                    _new["E_CHOICE_Bars_Direction"] = (
                        "Alternate Up" if _d == "up" else "Alternate Down")
                elif _d in ("Left", "Right") and _pick < 0.15:
                    # Orientation: open from the middle instead of crossing.
                    _new["E_CHOICE_Bars_Direction"] = (
                        "expand" if _pick < 0.08 else "H-expand")
            # SPEED THAT CHANGES ON THE BEAT. Krists puts a VALUE CURVE on
            # Bars_Cycles -- one measured example ramps 30 to 141 crossings
            # across four seconds -- which is how a bar can move fast and then
            # slow down rather than running at one rate. Same encoding as the
            # brightness curves: value x Max, here Max=300.
            #
            # Stepping between a fast and a slow rate on alternate beats is
            # the "speed changed back and forth on the half notes" he
            # remembers, and it cannot be expressed with a scalar at all.
            if _span >= 2 * _beat_s and _new.get(
                    "B_CHOICE_BufferStyle") == "Per Model Default" \
                    and random.Random(f"cyc|{_e.model_name}|{_bar_i}").random() < 0.35:
                # Capped at two crossings per beat. Unbounded doubling took
                # Gooey from 6 strobe runs to 15: a bar already crossing 12
                # times in its span became 24, which is not "fast", it is
                # flicker.
                _fast = min(_cycles * 2.0, max(2.0, _beats * 2.0))
                _slow = max(1.0, _cycles * 0.5)
                _pts = []
                _nb = max(2, int(_beats))
                for _i in range(_nb + 1):
                    _t = min(1.0, _i / _nb)
                    _v = (_fast if _i % 2 == 0 else _slow) / 300.0
                    _pts.append(f"{_t:.3f}:{min(1.0, _v):.3f}")
                _new["E_VALUECURVE_Bars_Cycles"] = (
                    "Active=TRUE|Id=ID_VALUECURVE_Bars_Cycles|Type=Custom|"
                    "Min=0.00|Max=300.00|RV=TRUE|Values=" + ";".join(_pts) + "|")
            # TILT. "You can also change the orientation (like degrees tilted)
            # on the beat too." A buffer rotation turns the bars diagonal;
            # Krists uses Rotate CC 90 for this.
            if _new.get("B_CHOICE_BufferStyle") == "Per Model Default" \
                    and random.Random(f"tilt|{_e.model_name}|{_bar_i}").random() < 0.08:
                _new["B_CHOICE_BufferTransform"] = "Rotate CC 90"
            _e.settings = _new
            _n += 1
        # THE SAME ARITHMETIC FOR EVERYTHING THAT REPEATS.
        #
        # Shockwave, Ripple and Spirals all expose a repeat count, and Krists
        # sets it on nearly every one he places: Shockwave_Cycles 11,780
        # times, Ripple_Cycles 2,815. A ring that pulses once per beat is
        # cycles = span-in-beats, exactly as for Bars. Ours were left at the
        # default, so they pulsed once per effect however long it ran.
        _r2 = 0
        for _e in effects:
            _span = (_e.end_time_ms - _e.start_time_ms) / 1000.0
            if _span < 2 * _beat_s:
                continue
            _beats = _span / _beat_s
            _bar_i = int((_e.start_time_ms / 1000.0) / _bar_s)
            if _e.effect_name == "Shockwave":
                _c = max(1, min(int(round(_beats / _repeat_unit(_e.model_name))),
                                _MAX_REPEATS))
                _e.settings = {**_e.settings,
                               "E_SLIDER_Shockwave_Cycles": str(_c)}
                _r2 += 1
            elif _e.effect_name == "Ripple":
                _c = max(1, min(int(round(_beats / (2 * _repeat_unit(_e.model_name)))),
                                _MAX_REPEATS))
                _e.settings = {**_e.settings,
                               "E_TEXTCTRL_Ripple_Cycles": f"{_c}.0"}
                _r2 += 1
        if _n or _r2:
            print(f"Beat-locked motion: {_n} bars given a crossing rate from "
                  f"their own span (beat / half-note, alternating by bar); "
                  f"{_r2} shockwaves, ripples and spirals given a repeat rate "
                  f"the same way")
        return effects

    def _g_mark_section_changes(effects):
        """Every real section change gets a new look on the big canvases.

        "When sections change the look doesn't always change, like from chorus
        to verse and vice versa... ending the chorus and starting the verse had
        a distinctly different look right on the beat."

        Measured on Gooey: 10 of 56 boundaries passed with no shader change on
        any big canvas, no colour move and no level move -- nothing at all.
        This finds those and gives every canvas the same new shader on the
        boundary, so the change is a decision the whole facade makes together.

        Only boundaries that separate sections of genuinely different energy
        qualify; a song has more detected boundaries than it has real turns,
        and marking all 56 would be a strobe of its own.
        """
        _canv = [t for t in cast
                 if not t.get("synthetic") and t["name"] != "ALL Display"
                 and not _is_flood(t)
                 and (t.get("cat") == "matrix" or t["name"] in CONE_SHAPED
                      or "matri" in t["name"].lower()
                      or t["name"] == "Garage Roof")]
        if not _canv or not SHADER_KEYS:
            return effects
        _sh = [(e.start_time_ms / 1000.0, e.model_name) for e in effects
               if e.effect_name == "Shader"]
        _marked = 0
        for _i in range(1, len(analysis.sections)):
            _sec = analysis.sections[_i]
            _t = _sec.start_time
            if _t < 4 or _t > _dur_s - 4 or _after_dropout(int(_t * 1000)):
                continue
            _prev = analysis.sections[_i - 1]
            # A real turn: the energy either side actually differs.
            _e0 = getattr(_prev, "energy_level", 0.0) or 0.0
            _e1 = getattr(_sec, "energy_level", 0.0) or 0.0
            if abs(_e1 - _e0) < 0.12:
                continue
            if any(abs(_s - _t) <= 0.25 for _s, _ in _sh):
                continue                      # already marked
            if any(_t >= _wa and _t <= _wb for _wa, _wb in blackout_windows):
                continue
            _sm = section_meta[_i] if _i < len(section_meta) else {}
            _pal = (_sm.get("palette") if isinstance(_sm, dict) else None) \
                or PALETTES["ice"]
            _key = random.Random(f"secmark|{SONG_NAME}|{_i}").choice(SHADER_KEYS)
            # Ends at the NEXT boundary, never past it. A 12s marker blanketed
            # whatever came next, so marking three boundaries in Gooey left two
            # more unmarked than before -- the fix created the problem it was
            # fixing, one boundary later.
            _end = int(min(_sec.end_time, _t + 8.0) * 1000)
            for _t2 in _canv:
                add_target(_t2, L_BASE, _key, int(_t * 1000), _end, _pal)
            _marked += 1
        if _marked:
            print(f"Section marking: {_marked} boundaries that changed nothing "
                  f"given a new look across every big canvas")
        return effects

    def _g_no_new_look_during_buildup(effects):
        """The drop owns the change; the bars before it do not.

        "At 2:43 in Last Goodbye there's a big exciting and fast moving shader
        that starts even though the song is still building up to the drop
        which doesn't happen until about 2:47."

        A detected section boundary can legitimately fall inside a build-up --
        that IS where the build starts -- but giving it a new shader spends the
        moment early and leaves the drop with nothing to reveal. Measured:
        the section at 163.17 put an eight-second Spacels on every canvas 3.3s
        before the drop at 166.46.

        Any shader starting in the run-up is dropped and the previous one is
        held through to the drop, so the look changes once, on the hit.
        """
        if not SNAPPED_DROPS:
            return effects
        _lead = 5.0
        _big = {t["name"] for t in cast
                if t.get("cat") == "matrix" or t["name"] in CONE_SHAPED
                or "matri" in t["name"].lower() or t["name"] == "Garage Roof"}
        _by_prop: dict[str, list] = defaultdict(list)
        for _e in effects:
            if _e.effect_name == "Shader" and _e.model_name in _big:
                _by_prop[_e.model_name].append(_e)
        _kill: set[int] = set()
        for _row in _by_prop.values():
            _row.sort(key=lambda e: e.start_time_ms)
            for _i, _e in enumerate(_row):
                _t = _e.start_time_ms / 1000.0
                _d = next((d for d in sorted(SNAPPED_DROPS)
                           if _t < d - 0.3 and _t >= d - _lead), None)
                if _d is None:
                    continue
                # MOVED to the drop, not deleted. Deleting it left the canvas
                # with nothing of its own from the drop until the next change
                # -- a four-second hole at exactly the moment that should be
                # the biggest. Moving it makes the same shader arrive on the
                # hit instead of before it, which is what the change was for.
                _dms = int(_d * 1000)
                if _e.end_time_ms - _dms >= 1500:
                    _e.start_time_ms = _dms
                else:
                    _kill.add(id(_e))
                # Hold whatever was showing before it through to the drop.
                for _p in reversed(_row[:_i]):
                    if id(_p) not in _kill and _p.start_time_ms < _dms:
                        _p.end_time_ms = max(_p.end_time_ms, _dms)
                        break
        if _kill or True:
            effects = drop_effects(effects, _kill,
                                   "new look during a drop's run-up") \
                if _kill else effects
            print(f"Build-up hold: {len(_kill)} shader changes removed and "
                  f"the rest moved onto the drop, so the look changes on the "
                  f"hit rather than during the climb")
        return effects

    def _g_mark_every_drop(effects):
        """Every drop changes the look, on the drop.

        The section marker guarantees this for boundaries; drops had no
        equivalent, and an audit across the library found most songs with
        drops that no shader change landed on -- Cruise 2 of 5, Gooey 2 of 5,
        World Is Mine 2 of 4. "Go through the timing of the shader changes and
        section changes everywhere, make sure we're aligned."

        Runs after the build-up hold, so a change that was moved onto the drop
        already counts and is not doubled.
        """
        _canv = [t for t in cast
                 if not t.get("synthetic") and t["name"] != "ALL Display"
                 and not _is_flood(t)
                 and (t.get("cat") == "matrix" or t["name"] in CONE_SHAPED
                      or "matri" in t["name"].lower()
                      or t["name"] == "Garage Roof")]
        if not _canv or not SHADER_KEYS or not SNAPPED_DROPS:
            return effects
        _names = {t["name"] for t in _canv}
        _sh = sorted((e.start_time_ms / 1000.0, e.model_name) for e in effects
                     if e.effect_name == "Shader" and e.model_name in _names)
        _marked = 0
        for _d in sorted(set(SNAPPED_DROPS)):
            if _d < 4 or _d > _dur_s - 4 or _after_dropout(int(_d * 1000)):
                continue
            if any(abs(_t - _d) <= 0.30 for _t, _ in _sh):
                continue
            if any(_d >= _wa and _d <= _wb for _wa, _wb in blackout_windows):
                continue
            _si = next((i for i, sec in enumerate(analysis.sections)
                        if sec.start_time <= _d < sec.end_time), 0)
            _sm = section_meta[_si] if _si < len(section_meta) else {}
            _pal = (_sm.get("palette") if isinstance(_sm, dict) else None) \
                or PALETTES["ice"]
            _key = random.Random(f"dropmark|{SONG_NAME}|{_d:.2f}").choice(SHADER_KEYS)
            # To the next change on that canvas, or eight seconds.
            for _t2 in _canv:
                _nxt = min((_t for _t, _m in _sh
                            if _m == _t2["name"] and _t > _d + 0.5),
                           default=_d + 8.0)
                add_target(_t2, L_BASE, _key, int(_d * 1000),
                           int(min(_nxt, _d + 8.0) * 1000), _pal)
            _marked += 1
        if _marked:
            print(f"Drop marking: {_marked} drops that no shader change landed "
                  f"on given a new look across every big canvas")
        return effects

    def _g_floods_flat(effects):
        # FLOODS CARRY FLAT LIGHT ONLY -- enforced at the end.
        #
        # The substitution in add_target caught the effects that route through it
        # and missed 181 Shockwaves that several passes place with a direct add().
        # Rather than chase every caller, this converts whatever is left: a flood
        # is a single RGB node, so a Shockwave on it is not a ring, it is that
        # ring's brightness sampled at one point -- flicker.
        #
        # Converted rather than dropped, so the flood still marks the moment; it
        # just marks it by lighting and fading instead of by flickering.
        _flat_ok = {"On", "Off", "Color Wash"}
        _conv = 0
        for _e in effects:
            if "flood" not in _e.model_name.lower() or _e.effect_name in _flat_ok:
                continue
            _fi = _e.settings.get("T_TEXTCTRL_Fadein", "0.10")
            _fo = _e.settings.get("T_TEXTCTRL_Fadeout", "0.60")
            _e.effect_name = "Color Wash"
            _e.settings = {"B_CHOICE_BufferStyle": "Per Model Default",
                           "E_CHECKBOX_ColorWash_HFade": "1",
                           "T_TEXTCTRL_Fadein": _fi, "T_TEXTCTRL_Fadeout": _fo,
                           "X_Src": _e.settings.get("X_Src", "0")}
            # KEEP THE LEVEL THE ORIGINAL WOULD HAVE PRODUCED. A shader or
            # Shockwave on a single-pixel flood lights it for only part of its
            # span; a flat wash lights it fully for all of it. Converting without
            # compensating raised the floods -- 23 of the brightest fixtures in the
            # show -- to full for every span, which lifted the whole-house level
            # during run-ups and cost the drops their contrast. Measured on This
            # Version of You: the run-up into the 1:17 hit went from 59.7 to 99.8
            # and the hit's rise over it fell from x1.38 to x1.15.
            if _e.palette is not None and not any(
                    x.startswith("C_VALUECURVE_Brightness=")
                    for x in (_e.palette.extra_parts or [])):
                _e.palette = _e.palette.model_copy(update={
                    "extra_parts": list(_e.palette.extra_parts or []) + [
                        "C_VALUECURVE_Brightness=Active=TRUE|"
                        "Id=ID_VALUECURVE_Brightness|Type=Custom|Min=0.00|"
                        f"Max=400.00|RV=TRUE|Values=0.00:{VC_NOMINAL * 0.55:.2f};"
                        f"1.00:{VC_NOMINAL * 0.55:.2f}"]})
            _conv += 1
        if _conv:
            print(f"Floods: {_conv} spatial effects converted to a flat wash "
                  f"(a flood is one pixel -- motion on it can only be flicker)")
        return effects

    def _g_no_slab(effects):
        # NO SOLID SLAB ON A TEXTURED MATRIX.
        #
        # "There is an effect in some songs on the matrices where the bottom half
        # of the matrix is a mostly solid shape and the upper half has a shader or
        # something... I don't like that effect so let's remove it."
        # (The WhatsApp Shakespeare at 0:39.)
        #
        # Measured there: every matrix ran a shader on its body while its own
        # `Horizontals` submodel was filled by a plain `On`, which renders as a
        # hard-edged solid band across the middle with texture above and below --
        # one prop showing two unrelated things at once.
        #
        # Only a FLAT FILL on the matrix's OWN submodel is removed, and only while
        # the parent is textured. Chases along those strips still run (they read as
        # motion along the outline, not as a slab), and the chord accents on the
        # outline families are Color Wash with fades, so they are untouched -- they
        # are the thing Krists liked in This Version of You.
        _tex_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for _e in effects:
            _n = _e.model_name
            if "matrix" in _n.lower() and "tunnel" not in _n.lower() and "/" not in _n \
                    and _e.effect_name in ("Shader", "Plasma", "Spirals"):
                _tex_spans[_n].append((_e.start_time_ms, _e.end_time_ms))
        # FLAT FILLS ONLY. This was briefly broadened to every effect on a matrix
        # submodel, which was wrong on both counts: it did not fix the band (that
        # was the Paper Layers shader, see SHADER_PICKS) and it removed movement
        # that is wanted -- "the single strand chases on outlines is a desired
        # effect, it gives movement we want".
        _slab = []
        for _e in effects:
            _n = _e.model_name
            if "/" not in _n or _e.effect_name != "On":
                continue
            _parent = _n.split("/")[0]
            if "matrix" not in _parent.lower() or "tunnel" in _parent.lower():
                continue
            if any(_e.start_time_ms < _b and _a < _e.end_time_ms
                   for _a, _b in _tex_spans.get(_parent, [])):
                _slab.append(_e)
        if _slab:
            _ids = {id(e) for e in _slab}
            effects = drop_effects(effects, _ids, "solid-slab guard")
            print(f"Solid-slab guard: {len(_slab)} flat fills removed from matrix "
                  f"submodels that were sitting on a textured parent")
        return effects

    def _g_slivers(effects):
        # Sliver effects. A 30ms effect is one frame at 40fps: it cannot read as a
        # gesture, only as a glitch, and several landing together are a strobe.
        # These are not placed deliberately -- they are the residue of passes that
        # trim and re-clip spans (blackout clipping, run-up dips, section splits)
        # leaving a stub behind. Measured in Lose My Cool at 155.65s: a 0.03s Off
        # and a 0.03s Twinkle inside a 10.9 Hz run across three matrices.
        _SLIVER_MS = 120
        _slivers = [e for e in effects
                    if e.end_time_ms - e.start_time_ms < _SLIVER_MS]
        if _slivers:
            _ids = {id(e) for e in _slivers}
            effects = drop_effects(effects, _ids, "sliver (under 120ms)")
            print(f"Sliver effects: {len(_slivers)} removed "
                  f"(shorter than {_SLIVER_MS}ms -- one to four frames)")

        # These two run LAST, after the bare-canvas and cohesion fills, because
        # those passes ADD effects. Running them earlier meant a shader the fill
        # placed on a drop never had the outgoing shader cut from under it --
        # Gooey 3:01, where Circuit Trace started dead on the drop while
        # Herringbone ran on for another 1.7s -- and meant fill-created effects
        # never had their same-layer overlaps resolved at all.
        return effects

    def _g_replace_on_drop(effects):
        # A SHADER CHANGE ON A DROP IS A REPLACEMENT, NOT AN ADDITION.
        #
        # The layer-overlap pass lifts a later effect onto a free layer rather than
        # truncating the earlier one, which is right for decay tails and wrong for
        # a shader change: the new shader starts exactly on the drop while the old
        # one keeps running underneath, so the two blend and the change reads as
        # late even though it began on time. Measured on Gooey at 3:01: Circuit
        # Trace started dead on the drop at 181.37 while Herringbone ran on until
        # 183.10 -- 1.7s of both at once.
        _cut_late = 0
        _anchor_set = sorted(set(_drop_anchors)
                             | {_on_hit_beat(sec.start_time) for sec in analysis.sections})
        _prop_sh: dict[str, list] = {}
        for _e in effects:
            if _e.effect_name == "Shader":
                _prop_sh.setdefault(_e.model_name, []).append(_e)
        for _row in _prop_sh.values():
            _row.sort(key=lambda e: e.start_time_ms)
            for _i, _new in enumerate(_row):
                # NOT `_bt` -- that name holds the beat-times array this whole
                # block closes over, and reusing it here silently replaced the
                # array with a float.
                _nstart = _new.start_time_ms / 1000.0
                if not any(abs(_nstart - _a) <= 0.06 for _a in _anchor_set):
                    continue
                for _old in _row[:_i]:
                    if _old.end_time_ms > _new.start_time_ms \
                            and _old.start_time_ms < _new.start_time_ms:
                        _old.end_time_ms = _new.start_time_ms
                        _cut_late += 1
        if _cut_late:
            print(f"Shader replacement: {_cut_late} outgoing shaders cut at the "
                  f"drop or section change instead of running on underneath")

        # =====================================================================
        return effects

    def _g_layer_overlaps(effects):
        # Same-layer overlap resolution -- MUST be last, after every pass that
        # can move or lengthen an effect.
        #
        # An xLights EffectLayer is a single timeline: two effects cannot occupy
        # the same milliseconds on it. 21% of the effects in this song were doing
        # exactly that, because a dozen passes each place effects independently
        # and none of them could see the others. The chord gestures were the
        # clearest casualty -- three washes on Corner Verticals L1 at 26.43-29.02,
        # 27.23-29.82 and 28.85-31.45, each correct on its own, together
        # unrenderable, which is why those props measured DARK in the .fseq while
        # the .xsq looked right. This is almost certainly why several rounds of
        # fixes have measured inert: the change was real in the file and never
        # reached the pixels.
        #
        # Resolution moves the LATER effect up a layer, which is both what the eye
        # expects (the newer event paints over the older) and what preserves the
        # long decay tails that make a chord ring. Truncation is the fallback for
        # when there is genuinely no free layer.
        _MAX_LAYER = 12
        _lanes: dict[tuple[str, int], list] = {}
        for _e in effects:
            _lanes.setdefault((_e.model_name, _e.layer), []).append(_e)
        _moved = _cut = 0
        _busy = {k: sorted((e.start_time_ms, e.end_time_ms) for e in v)
                 for k, v in _lanes.items()}
        for (_mn, _lay), _row in sorted(_lanes.items()):
            _row.sort(key=lambda e: (e.start_time_ms, e.end_time_ms))
            _end = None
            for _e in _row:
                if _end is not None and _e.start_time_ms < _end:
                    _dest = None
                    for _l2 in range(_lay + 1, _MAX_LAYER):
                        _occ = _busy.get((_mn, _l2), [])
                        if not any(_e.start_time_ms < _b and _a < _e.end_time_ms
                                   for _a, _b in _occ):
                            _dest = _l2
                            break
                    if _dest is not None:
                        _busy.setdefault((_mn, _dest), []).append(
                            (_e.start_time_ms, _e.end_time_ms))
                        _e.layer = _dest
                        _moved += 1
                        continue
                    _e.start_time_ms = _end
                    _cut += 1
                    if _e.end_time_ms - _e.start_time_ms < 200:
                        _e.end_time_ms = _e.start_time_ms + 200
                _end = max(_end or 0, _e.end_time_ms)
        print(f"Layer overlaps: {_moved} effects lifted to a free layer, "
              f"{_cut} trimmed (an xLights layer is one timeline -- overlapping "
              f"effects on it do not render)")

        # The ceiling runs LAST, after the replacement cut and the overlap pass,
        # because those SHORTEN spans -- and a value curve's times are normalised
        # across its effect's own span, so shortening one compresses its curve.
        # Measured on Too Slow: the replacement cut took the Cable Knit house bed
        # from 11.5s to 4.0s, compressing its pulse by 2.9x, and the strobe count
        # went from 16 runs to 87 -- on props whose ONLY effect was that bed.

        # =====================================================================
        return effects

    def _g_pulse_ceiling(effects):
        # Pulse-rate ceiling, enforced on the CURVES themselves.
        #
        # A value curve's times are normalised across its effect's span, so a curve
        # built against one span and then applied to a shorter one pulses faster
        # than the music it came from -- 38 points inside 1.02s on Harmony, which
        # is a 19 Hz flicker. Several passes build curves and several others resize
        # effects, so no single one of them can prevent this; it has to be checked
        # at the end, in real seconds, against the span each curve actually ends up
        # with.
        #
        # "Across songs we're using the floods too much and strobing... no songs
        # should do that", and on the floods specifically: "it can give people
        # migraines." 240ms is the same ceiling the effect-spacing pass uses.
        _MIN_PT_S = 0.24
        # Floods get the musical limit, not the safety limit. 240ms is four
        # pulses a second, which is faster than a quarter note on every song in
        # this show -- fine for a prop with shape, far too fast for a single-pixel
        # flood that can only express brightness. "The floods would never go
        # faster than quarter or half notes."
        _QUARTER_S = 60.0 / max(analysis.beats.tempo, 60.0)
        _fixed = 0
        for _e in effects:
            _pal = _e.palette
            if _pal is None or not _pal.extra_parts:
                continue
            _vc = next((x for x in _pal.extra_parts
                        if x.startswith("C_VALUECURVE_Brightness=")), None)
            if _vc is None:
                continue
            _m = re.search(r"Values=([0-9.:;]+)", _vc)
            if not _m:
                continue
            _dur = (_e.end_time_ms - _e.start_time_ms) / 1000.0
            if _dur <= 0:
                continue
            _limit = (max(_MIN_PT_S, _QUARTER_S)
                      if "flood" in _e.model_name.lower() else _MIN_PT_S)
            try:
                _pts = [(float(a), float(b)) for a, b in
                        (x.split(":") for x in _m.group(1).split(";") if ":" in x)]
            except ValueError:
                continue
            if len(_pts) < 3:
                continue
            _pts.sort()
            _keep, _last_t = [_pts[0]], _pts[0][0]
            for _t, _v in _pts[1:-1]:
                if (_t - _last_t) * _dur >= _limit:
                    _keep.append((_t, _v))
                    _last_t = _t
            if (_pts[-1][0] - _last_t) * _dur >= _limit or len(_keep) == 1:
                _keep.append(_pts[-1])
            if len(_keep) == len(_pts):
                continue
            _new_vc = _vc[:_m.start(1)] + ";".join(
                f"{t:.3f}:{v:.2f}" for t, v in _keep) + _vc[_m.end(1):]
            _e.palette = _pal.model_copy(update={"extra_parts": [
                _new_vc if x is _vc else x for x in _pal.extra_parts]})
            _fixed += 1
        print(f"Pulse-rate ceiling: {_fixed} brightness curves thinned to at most "
              f"one point per {_MIN_PT_S * 1000:.0f}ms of real time")
        return effects

    # THE LATE GUARDS, IN ORDER. Everything above places effects; everything
    # here corrects what came out. The order is not incidental and three bugs
    # in one day came from getting it wrong, so it is written down as data
    # rather than left implicit in the order the blocks happen to sit in.
    #
    # The invariants the order encodes, in the words of the bugs that found
    # them:
    #   * anything that ADDS effects must run before anything that VALIDATES
    #     them -- the fills used to run after the overlap resolver, so nothing
    #     they placed ever had its overlaps resolved;
    #   * anything that SHORTENS a span must run before the pulse ceiling,
    #     because a value curve's times are normalised across its own span, so
    #     shortening one compresses its curve;
    #   * the overlap resolver must be last of the structural passes, since
    #     every other guard can move or lengthen an effect.
    LATE_GUARDS = [
        ("mark section changes on the big canvases", _g_mark_section_changes),
        ("fill canvases lit flat by the house span", _g_fill_flat_canvases),
        ("bars land on the beat and fade", _g_bars_discipline),
        ("no new look during a drop's run-up", _g_no_new_look_during_buildup),
        ("every drop changes the look", _g_mark_every_drop),
        ("tie repeating motion to the beat grid", _g_beat_lock_motion),
        ("floods carry flat light only", _g_floods_flat),
        ("no solid slab on a textured matrix", _g_no_slab),
        ("drop sliver effects", _g_slivers),
        ("shader change on a drop is a replacement", _g_replace_on_drop),
        ("resolve same-layer overlaps", _g_layer_overlaps),
        ("pulse-rate ceiling on the curves", _g_pulse_ceiling),
    ]
    for _gname, _gfn in LATE_GUARDS:
        _before = len(effects)
        effects = _gfn(effects)
        _probe(_gname)
        GUARD_LOG.append({"guard": _gname, "before": _before,
                          "after": len(effects)})

    _mirror = [e for e in effects if e.model_name == TUNNEL_MIRROR_SRC]
    if _mirror:
        effects.extend(e.model_copy(update={"model_name": TUNNEL_MIRROR_DST})
                       for e in _mirror)
        print(f"Tunnel preview mirror: {len(_mirror)} effects copied "
              f"{TUNNEL_MIRROR_SRC!r} -> {TUNNEL_MIRROR_DST!r} (preview-only arches)")

    # =====================================================================
    # Write
    # =====================================================================
    # Provenance summary for this build: how many effects each call site
    # produced, and how many SURVIVED to the file. A pass whose placements all
    # vanish is the single most common failure mode here and is invisible
    # otherwise -- see the run-up dip shredding ambient holds, the cohort lock
    # dragging chord hits off the beat, and the black value curves.
    try:
        from collections import Counter as _C
        _src = _C(int(e.settings.get("X_Src", 0)) for e in effects)
        _srclines = SOURCE_LINES
        _prov = {}
        for _ln, _n in _src.items():
            _prov[str(_ln)] = {"count": _n, "pass": _pass_label(_ln, _srclines)}
        (GEN_DIR / "provenance.json").write_text(json.dumps({
            "total": len(effects),
            "by_line": _prov,
            # The times the passes actually targeted, so a verification tool
            # measures against those rather than re-deriving them and
            # disagreeing. verify_alignment.py was reporting correctly-placed
            # changes as misses because it compared against the RAW detections
            # while every pass uses the snapped ones.
            "snapped_drops": sorted(set(round(float(t), 3)
                                        for t in SNAPPED_DROPS)),
            "section_starts": [round(float(sec.start_time), 3)
                               for sec in analysis.sections],
            "guards": GUARD_LOG,
            "removals": [{**_r, "from_passes": {
                _pass_label(int(_k), _srclines): _v
                for _k, _v in _r["from_lines"].items()}}
                for _r in REMOVALS],
        }, indent=1))
    except Exception as _e:                     # never block a build on this
        print(f"  (provenance summary skipped: {_e})")

    spec = SequenceSpec(
        song_title=SONG_NAME, artist="", album="",
        media_file=str(MP3_PATH),
        duration_ms=analysis.duration_ms, timing_ms=25,
        palettes=list(PALETTES.values()), effects=effects, timing_tracks=[],
    )
    write_xsq(spec, show_config, OUTPUT_PATH)

    lit_targets = len({e.model_name for e in effects})
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  {len(effects)} effect placements across {lit_targets} target elements")

    durs = sorted(e.end_time_ms - e.start_time_ms for e in effects)
    print(f"  duration ms: p10={durs[len(durs)//10]} median={durs[len(durs)//2]} "
          f"p90={durs[9*len(durs)//10]}")

    cue_log.sort(key=lambda c: c["time"])
    lines = [f"# {SONG_NAME} — cue sheet ({OUT_NAME})", ""]
    for c in cue_log:
        m, s = divmod(c["time"], 60)
        lines.append(f"- **{int(m)}:{s:05.2f}** `{c['kind']}` — {c['desc']}")
    CUESHEET_PATH.write_text("\n".join(lines))
    print(f"  cue sheet: {CUESHEET_PATH} ({len(cue_log)} entries)")


if __name__ == "__main__":
    main()
