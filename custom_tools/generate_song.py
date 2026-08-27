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
OUT_NAME = f"{SONG_NAME} (v8).xsq"
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

def _chase(direction: str, fade: str = "Head and Tail", span: bool = True,
           chases: str = "1") -> dict:
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
        "E_TEXTCTRL_Chase_Rotations": "1.0",
    }
    if span:
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


def _bars(direction: str, cycles: str = "1.0", buffer: dict | None = None) -> dict:
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
        "E_CHECKBOX_Bars_Highlight": "0", "E_CHOICE_Bars_Direction": direction,
        "E_SLIDER_Bars_BarCount": "1", "E_TEXTCTRL_Bars_Cycles": cycles,
    }


EFFECTS: dict[str, tuple[str, dict]] = {
    # --- feature motion (span the group as one canvas) ---
    "chase_lr":       ("SingleStrand", _chase("Left-Right")),
    "chase_rl":       ("SingleStrand", _chase("Right-Left")),
    "chase_mid_out":  ("SingleStrand", _chase("From Middle")),
    "chase_mid_in":   ("SingleStrand", _chase("To Middle")),
    "chase_bounce":   ("SingleStrand", _chase("Bounce from Left")),
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
    # Travel ALONG each strand (physically up/down on a vertical strip).
    # `Bounce from Left` does the out-and-back in a single effect, which is
    # what v5 used and what reads as a strip pulsing up and down.
    "strand_bounce":  ("SingleStrand",
                       {**_chase("Bounce from Left", span=False), **PMSL}),
    "strand_up":      ("SingleStrand",
                       {**_chase("Left-Right", span=False), **PMSL}),
    "strand_down":    ("SingleStrand",
                       {**_chase("Right-Left", span=False), **PMSL}),
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
SHADER_PICKS = [
    "Wisps", "Metablob Starz", "InnerDimensionalMatrix", "Spirograph Spectrum",
    "Hex Vortex", "Mobius Spiral", "Spacels",
    "SearchingForSomething", "Hex 3D Spiral",
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
    "Paper Layers",
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
]


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
    # Smooth over ~0.6s so individual notes do not become flicker.
    w = max(3, int(0.6 / max(float(rt[1] - rt[0]), 1e-6)))
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
        v = VC_NOMINAL * (1.0 - 0.55 * depth + 1.05 * depth * float(e[i]))
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
                    "frost_growth", "paper_layers")
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


def line_motion_for(target: dict) -> tuple[str, str]:
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
        return ("strand_up", "strand_down")
    wide = target.get("spread_x", 0) >= 40 or "c9" in name or "gutter" in name
    return ("chase_dots_lr", "chase_dots_rl") if wide else \
           ("chase_pair_lr", "chase_pair_rl")


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
        return rng.choice(["chase_bounce", "chase_mid_out", "chase_lr", "chase_rl"])
    if cat == "arch":
        # Deliberate, readable sweeps -- NOT fast multi-rotation blur.
        return rng.choice(["chase_lr", "chase_rl", "chase_mid_out", "chase_mid_in"])
    if cat == "matrix":
        # No Curtain, no Ripple here -- both have been occasional red-flash
        # sources on matrix-category targets across multiple rounds even
        # with the buffer-style fix. Curtain stays reserved for genuine
        # curtain-shaped containers (LARGE_CANVAS); real shaders substitute
        # for the texture Ripple/Curtain would have added.
        if escalation >= 2:
            return rng.choice(["chase_mid_out", "plasma_organic", *SHADER_KEYS])
        return rng.choice(["chase_lr", "chase_rl", "wash", "plasma_organic"])
    if cat == "tree":
        return rng.choice(["chase_mid_out", "meteors_down", "spirals"])
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
        ])
    if n >= 4:
        return rng.choice(["sweep_right", "sweep_left"] if wide else ["sweep_up", "sweep_down"])
    return rng.choice(["chase_lr", "chase_rl", "shockwave"])


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
    show_config = load_show_config(SHOW_PATH)
    layout = json.loads((SHOW_PATH / ".generation" / "layout.json").read_text())
    positions = {m["name"]: m for m in layout["models"]}
    analysis = SongAnalysis.model_validate_json((GEN_DIR / "analysis.json").read_text())
    cues = json.loads((GEN_DIR / "cues.json").read_text())

    cast = build_cast(show_config, positions)
    covered = sum(t["n"] for t in cast)
    print(f"Cast: {len(cast)} targets covering {covered} models "
          f"({sum(1 for t in cast if t['kind']=='group')} groups, "
          f"{sum(1 for t in cast if t['kind']=='model')} models)")

    effects: list[EffectPlacement] = []
    cue_log: list[dict] = []

    def add(target_name, layer, key, start_ms, end_ms, palette, extra=None):
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
        if key in FLAT_KEYS and target_name in FLASH_SPLIT:
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
        if key in FLAT_KEYS and target_name in NO_FLAT_PROPS:
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
    FLASH_SPLIT: dict[str, list[str]] = {}
    for _gname, _mem in _xgroups.items():
        if _gname in NO_FLAT_PROPS or not _mem:
            continue
        if any(_m in NO_FLAT_PROPS for _m in _mem):
            _safe = [_m for _m in _mem if _m not in NO_FLAT_PROPS]
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
            extra = {k: v for k, v in (extra or {}).items()
                     if k != "B_CHOICE_BufferStyle"}
            extra = {**extra, **PERMODEL}
        # Per-model rendering wins over any spanning buffer style: for a group
        # of discrete scattered objects (stars, mini trees, floods) spanning
        # is exactly the thing that turns motion into strobing. Applied last
        # so it overrides the SPAN injection above.
        if target.get("per_model") and target["kind"] == "group" \
                and not target.get("synthetic"):
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
    # (canvas, cycle-start) pairs already given their one span -- see hold_spans.
    _cycle_claimed: set[tuple] = set()

    def hold_spans(target, layer, key, section, palette, seat, energy_level):
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
                            t_ms = int(t * 1000)
                            add(m, L_MOTION, "on_hit", t_ms, t_ms + hold, palette)
                            continue
                        going_forward = (bi // n_m) % 2 == 0
                        m = (ordered if going_forward else reversed_order)[bi % n_m]
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
                    else:
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
    ambient_passages: list[dict] = []
    try:
        from ambient_sections import analyse as _amb_analyse
        _amb = _amb_analyse(SONG_NAME)
        ambient_passages = [p for p in _amb["passages"]
                            if p["ambient"] and p["dur"] >= 8.0]
    except Exception as _e:
        print(f"  (ambient-passage analysis unavailable: {_e})")

    n_amb = 0
    _ambient_ids: set[int] = set()
    if ambient_passages:
        _rms = np.asarray(analysis.spectrum.rms_energy)
        _rt = np.asarray(analysis.spectrum.rms_times)
        amb_rng = random.Random(f"{SONG_NAME}|ambient")
        big = [t for t in cast if t.get("large")]
        movers = [t for t in cast if prominence_of(t) >= 2 and not t.get("large")]
        for pi, ap in enumerate(ambient_passages):
            t0, t1 = ap["t0"], ap["t1"]
            si_at = next((i for i, sec in enumerate(analysis.sections)
                          if sec.start_time <= t0 < sec.end_time), 0)
            _sm = section_meta[si_at] if si_at < len(section_meta) else {}
            pal = (_sm.get("palette") if isinstance(_sm, dict) else None) or PALETTES["ice"]
            # Intensity follows the passage's measured loudness: a loud
            # drumless synth passage should be big, a dark intro should not.
            loud = float(ap.get("loud", 0.5))
            # (a) the canvases hold ONE developing pattern for the whole
            #     passage, breathing with the section's own energy curve.
            key = amb_rng.choice([k for k in SHADER_KEYS
                                  if is_palette_driven(k) and not is_centred(k)]
                                 or SHADER_KEYS)
            # EVERY recognisable prop carries the contour, not just the big
            # canvases. Ten targets out of several hundred moved the measured
            # correlation between display brightness and music energy by 0.01
            # -- the shaped layer was sitting on an unshaped floor, exactly as
            # the drop/fade envelope did before the same fix.
            for seat, tgt in enumerate(big + movers):
                _k = key if tgt.get("large") else "wash"
                p_ = palette_variant(pal, seat) if is_palette_driven(_k) else pal
                p_ = envelope_palette(p_, _rms, _rt, int(t0 * 1000), int(t1 * 1000),
                                      depth=0.55 + 0.45 * loud)
                _n0 = len(effects)
                add_target(tgt, L_BASE, _k, int(t0 * 1000), int(t1 * 1000), p_)
                for _e2 in effects[_n0:]:
                    _ambient_ids.add(id(_e2))
                n_amb += 1
            # (b) a bloom on each sparse event, rotating around the house so
            #     the passage travels instead of repeating in one place.
            hold = int(min(2400, max(900, (t1 - t0) / max(len(ap["sparse"]), 1) * 900)))
            for ei, et in enumerate(ap["sparse"]):
                if not movers:
                    break
                tgt = movers[(ei * 3 + pi) % len(movers)]
                _n0 = len(effects)
                add_target(tgt, L_MOTION, "wash", int(et * 1000),
                           int(et * 1000) + hold, pal,
                           extra={"T_TEXTCTRL_Fadein": "0.30",
                                  "T_TEXTCTRL_Fadeout": f"{hold / 1000.0 * 0.75:.2f}"})
                for _e2 in effects[_n0:]:
                    _ambient_ids.add(id(_e2))
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
                    for _e2 in effects[_n0:]:
                        _ambient_ids.add(id(_e2))
                    n_amb += 1
        print(f"Ambient passages: {len(ambient_passages)} treated as shape, not "
              f"beat ({n_amb} placements; "
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
                "Type=Flat|Min=0.00|Max=400.00|RV=TRUE|Values=0.00:0.09;1.00:0.09"]})
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
                return "on_pop"
            # C9 gutters and other free linear props: a burst travelling out
            # from the middle marks the beat AND shows the prop's length.
            return "chase_mid_out"
        if target["cat"] == "arch":
            return "on_hit"
        return "on_pop"

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
                        if any(w in _low for w in ("spoke", "arm", "petal",
                                                   "whirl", "spinner", "flower")):
                            _shape = ["chase_mid_out", "pinwheel", "on_pop"]
                        elif any(w in _low for w in ("ring", "circle")):
                            _shape = ["shockwave", "ripple_explode", "on_pop"]
                        else:
                            _shape = ["chase_lr", "chase_rl", "on_pop"]
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
                # answer each other rather than firing as one slab.
                if (hi + gi) % 2:
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
                    max(25, int(int(slowed.get("E_SLIDER_Shader_Speed", 400)) * 0.12)))
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
                    mot = line_motion_for(tgt)
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
        span = max(end_ms - start_ms, n * 25)
        hold = max(160, int(span / n * 2.2))
        for k, t in enumerate(seq):
            s = start_ms + int(span * k / (n - 1))
            add(target_name(t, k), L_SIG, key, s, s + hold, palette)

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
                return climax_drop_rng.choice(SAFE_LARGE)
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
    cutouts = (_drops + _rest)[:6]
    cutouts.sort(key=lambda ev: ev["cutout_start"])
    blackout_windows = []
    ci_drop = 0
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
            collapse_key = "shockwave" if t.get("large") else "ripple_implode"
            add_target(t, L_SIG, collapse_key, c0, dark_start, PALETTES["white"])
            if full_blackout:
                add_target(t, L_SIG, "off", dark_start, drop, PALETTES["white"])
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
        grid = sorted(cohort_marks)

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
            if id(eff) not in _cycle_ids and eff.layer in (L_BASE, L_MOTION):
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
    if melody_notes:
        mel_candidates = [
            t for t in cast
            if t["n"] >= 4 and not t.get("symmetric")
            and t["name"] not in prop_voice and t["cat"] != "arch"
            and "/" not in t["name"]
            and not any(h in t["name"].lower() for h in OUTLINE_NAME_HINTS)
        ]
        if mel_candidates:
            mel_target = max(mel_candidates,
                             key=lambda t: (t["name"] in MUST_HAVE_GROUPS, t["n"]))
            members = sorted(mel_target["members"],
                             key=lambda m: positions.get(m, {}).get("x", 0))
            n_deg = melody_notes[0].get("n_degrees", 1) or 1
            for note in melody_notes:
                # Map the note's pitch rank onto a member index. More notes
                # than props is normal -- several adjacent semitones share a
                # prop, which still preserves "melody goes up = light moves
                # right" as the readable gesture.
                idx = int(note["degree"] * len(members) / max(1, n_deg))
                idx = max(0, min(len(members) - 1, idx))
                s_ms = int(note["start"] * 1000)
                e_ms = max(s_ms + 160, int(note["end"] * 1000))
                add(members[idx], L_SIG, "on_swell", s_ms, e_ms,
                    PALETTES["gold"])
                n_melody += 1
            log(melody_notes[0]["start"], "melody",
                f"{len(melody_notes)} notes over {n_deg} pitches -> "
                f"{mel_target['name']} ({len(members)} members) by pitch height")
    print(f"Melody notes mapped: {n_melody}"
          + (f" -> {mel_target['name']}" if n_melody else ""))

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
        _drop_ms = [int(ev["drop_time"] * 1000) for ev in cutouts]
        _kept, _cut = [], 0
        for eff in effects:
            if id(eff) in _ambient_ids or eff.layer not in (L_PULSE, L_SIG):
                _kept.append(eff)
                continue
            if eff.end_time_ms - eff.start_time_ms > 400:
                _kept.append(eff)
                continue
            s_s = eff.start_time_ms / 1000.0
            if not any(a["t0"] <= s_s < a["t1"] for a in ambient_passages):
                _kept.append(eff)
                continue
            if any(abs(eff.start_time_ms - d) <= 600 for d in _drop_ms):
                _kept.append(eff)      # the drop this passage was building to
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
        _kept = _kept2
        effects = _kept
        print(f"Ambient passages: cleared {_cut} rhythmic hits from "
              f"{len(ambient_passages)} smooth passages")

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
    _FLOOD_FLASH_GAP_MS = 400           # floods get more room: big, bright, close
    # A parent-level cap was tried and reverted: it dropped 2286 flashes
    # instead of 737 and removed none of the remaining fast runs, because
    # those turned out not to be flashes at all -- see the house-bed note in
    # shader_span_test.py.
    _dropped = set()
    for key, gap in (("sub", _MIN_FLASH_GAP_MS),):
        by_prop: dict[str, list] = defaultdict(list)
        for eff in effects:
            if id(eff) in _dropped:
                continue
            if eff.end_time_ms - eff.start_time_ms <= 400:
                nm = (eff.model_name if key == "sub"
                      else eff.model_name.split("/")[0])
                by_prop[nm].append(eff)
        for nm, evs in by_prop.items():
            g = _FLOOD_FLASH_GAP_MS if "flood" in nm.lower() else gap
            last = -10 ** 9
            for eff in sorted(evs, key=lambda e: e.start_time_ms):
                if eff.start_time_ms - last < g:
                    _dropped.add(id(eff))
                else:
                    last = eff.start_time_ms
    if _dropped:
        effects = [e for e in effects if id(e) not in _dropped]
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
        if id(eff) in _cycle_ids or id(eff) in _ambient_ids:
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
    print(f"Pulse curves rebuilt for final spans: {_rebuilt}")

    _mirror = [e for e in effects if e.model_name == TUNNEL_MIRROR_SRC]
    if _mirror:
        effects.extend(e.model_copy(update={"model_name": TUNNEL_MIRROR_DST})
                       for e in _mirror)
        print(f"Tunnel preview mirror: {len(_mirror)} effects copied "
              f"{TUNNEL_MIRROR_SRC!r} -> {TUNNEL_MIRROR_DST!r} (preview-only arches)")

    # =====================================================================
    # Write
    # =====================================================================
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
