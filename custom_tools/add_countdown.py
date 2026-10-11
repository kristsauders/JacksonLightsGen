"""A countdown clock on the matrices, hitting 0:00 on a given moment.

    uv run --with pillow python custom_tools/add_countdown.py <merged.xsq> --zero 95.89 [--from 0.89]

Krists (rev3 round 4): "The synth intro will be telling people the show is
about to start and I want to put a timer on the matrices, a countdown. Make
sure it's large and clear and uses the resolution appropriately. I want the
timer to hit zero when the organ starts for be with you and then the timer can
immediately disappear."

Each matrix gets one image per second drawn at ITS OWN pixel grid (strands x
nodes per strand, from the layout) and squeezed by its pixel aspect (ScaleX /
ScaleY -- the fence matrices' pixels are 1.2-1.4x wider than tall), so the
digits fill ~85% of its height and look undistorted on the house. The images
play through the Pictures effect (Scale To Fit = exactly one image pixel per
light) on a NEW top layer, opaque, so for the countdown the matrices are
clocks. M:SS counts down with the tick on whole seconds before `zero`; the
last ten seconds turn gold; 0:00 lands exactly on `zero` and fades out in
0.45 s.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
FONT = "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"
MATRICES = ["Garage Door Matrix", "Upper Window Matrix", "Sun Room Window Matrix",
            "Fence Matrix 1", "Fence Matrix 2"]
WHITE, GOLD = (255, 255, 255), (255, 194, 26)


def geometry(name: str) -> tuple[int, int, float]:
    x = (SHOW / "xlights_rgbeffects.xml").read_text(errors="ignore")
    at = dict(re.findall(r'(\w+)="([^"]*)"', re.search(r'<model ([^>]*name="' + re.escape(name)
                                                       + r'"[^>]*)>', x).group(1)))
    cm = json.loads((SHOW / ".generation" / "channel_map.json").read_text())["models"]
    per = int(at["NodesPerString"]) // int(at["StrandsPerString"])
    strands = (cm[name]["count"] // 3) // per
    aspect = float(at.get("ScaleX", 1)) / float(at.get("ScaleY", 1))   # pixel width / height
    return strands, per, aspect


def draw(label: str, w: int, h: int, aspect: float, colour, out: Path) -> None:
    """Text drawn at high resolution in PHYSICAL proportions, then resampled to
    the w x h light grid (which squeezes x by the pixel aspect)."""
    k = 12
    pw, ph = int(round(w * aspect * k)), h * k
    img = Image.new("RGB", (pw, ph), (0, 0, 0))
    d = ImageDraw.Draw(img)
    size = int(ph * 0.95)
    while size > 8:
        f = ImageFont.truetype(FONT, size)
        x0, y0, x1, y1 = d.textbbox((0, 0), label, font=f)
        if (y1 - y0) <= ph * 0.86 and (x1 - x0) <= pw * 0.94:
            break
        size -= 2
    x0, y0, x1, y1 = d.textbbox((0, 0), label, font=f)
    d.text(((pw - (x1 - x0)) / 2 - x0, (ph - (y1 - y0)) / 2 - y0), label, font=f, fill=colour)
    img.resize((w, h), Image.LANCZOS).save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xsq"); ap.add_argument("--zero", type=float, required=True)
    ap.add_argument("--from", dest="start", type=float, default=None)
    a = ap.parse_args()
    zero = a.zero
    start = a.start if a.start is not None else zero - math.floor(zero)
    n0 = int(round(zero - start))                       # seconds shown at `start`
    tree = ET.parse(a.xsq); r = tree.getroot()
    db = r.find("EffectDB"); edb = [(c.text or "") for c in db]
    eix = {s: i for i, s in enumerate(edb)}

    def ge(s: str) -> int:
        if s not in eix:
            eix[s] = len(edb); edb.append(s); ET.SubElement(db, "Effect").text = s
        return eix[s]

    ee = r.find("ElementEffects"); els = {e.get("name"): e for e in ee}
    total = 0
    for m in MATRICES:
        w, h, asp = geometry(m)
        d = SHOW / "Countdown" / f"{w}x{h}_{asp:.2f}"
        d.mkdir(parents=True, exist_ok=True)
        el = els.get(m)
        if el is None:
            el = ET.SubElement(ee, "Element", {"type": "model", "name": m}); els[m] = el
        layer = ET.Element("EffectLayer")
        el.insert(0, layer)                              # layer 0 = on top
        for n in range(n0, -1, -1):
            t0 = zero - n if n < n0 else start
            t1 = zero - (n - 1) if n > 0 else zero + 0.45
            label = f"{n // 60}:{n % 60:02d}"
            col = GOLD if n <= 10 else WHITE
            png = d / f"{'g' if col == GOLD else 'w'}_{n:03d}.png"
            if not png.exists():
                draw(label, w, h, asp, col, png)
            st = ("E_CHECKBOX_Pictures_PixelOffsets=0,E_CHECKBOX_Pictures_Shimmer=0,"
                  "E_CHECKBOX_Pictures_TransparentBlack=0,E_CHECKBOX_Pictures_WrapX=0,"
                  "E_CHOICE_Pictures_Direction=none,E_CHOICE_Scaling=Scale To Fit,"
                  f"E_FILEPICKER_Pictures_Filename={png},E_SLIDER_PicturesXC=0,E_SLIDER_PicturesYC=0,"
                  "E_SLIDER_Pictures_EndScale=100,E_SLIDER_Pictures_StartScale=100,"
                  "E_TEXTCTRL_Pictures_FrameRateAdj=1.0,E_TEXTCTRL_Pictures_Speed=1.0"
                  + (",T_TEXTCTRL_Fadeout=0.40" if n == 0 else "") + ",X_Countdown=1")
            ET.SubElement(layer, "Effect", {"ref": str(ge(st)), "name": "Pictures", "palette": "0",
                                            "startTime": str(int(round(t0 * 1000 / 25)) * 25),
                                            "endTime": str(int(round(t1 * 1000 / 25)) * 25)})
            total += 1
        print(f"  {m}: {w}x{h} lights, pixel aspect {asp:.2f}, {n0 + 1} frames {n0 // 60}:{n0 % 60:02d} -> 0:00")
    tree.write(a.xsq, encoding="UTF-8", xml_declaration=True)
    print(f"countdown: {total} frames, 0:00 at {zero:.2f}s")


if __name__ == "__main__":
    main()
