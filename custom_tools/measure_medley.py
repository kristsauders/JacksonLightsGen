"""Measure a rendered medley against its brief, from the .fseq.

    python custom_tools/measure_medley.py "<mix>" "<mix> (vN)"

Reads the rendered pixels (never the .xsq) through fseq_stream, so a multi-GB
render is fine, and checks the things the design promised:

  * per song: how much of the display is lit, how bright the lit pixels are,
    and how much the light moves -- so fifteen songs can be compared side by
    side;
  * every brightness cap (the warm-up at 10-20%, the organ at 30-50%): the
    brightest lit pixels inside the window;
  * the reveal: each of the hits, the gaps between them, the break and the
    drop -- lit fraction, level and the two colours;
  * the outro: frame-by-frame, how many props flash (rise and fall back within
    half a second) after no_flash_from, against the song before it, and the
    fade to black.

Brightness here is a pixel's brightest channel / 255, so a fully lit prop at a
20% brightness curve reads ~0.20.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fseq_stream import FseqStream  # noqa: E402

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")
ROOT = Path(__file__).resolve().parent
LIT = 0.04


def main() -> None:
    mix, ver = sys.argv[1], sys.argv[2]
    G = SHOW / ".generation" / mix
    plan = json.loads((G / "segments.json").read_text())
    slug = re.sub(r"[^a-z0-9]+", "_", mix.lower()).strip("_")
    spec = importlib.util.spec_from_file_location("d", ROOT / "medleys" / f"{slug}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    D = mod.DESIGN
    cmap = json.loads((SHOW / ".generation" / "channel_map.json").read_text())["models"]
    # Switched-off props (Active="0") render nothing; counting them would make
    # a fully lit house read as half dark.
    _x = (SHOW / "xlights_rgbeffects.xml").read_text(errors="ignore")
    dead = set(re.findall(r'name="([^"]+)"[^>]*Active="0"', _x))
    models = {n: r for n, r in cmap.items()
              if not n.startswith("Tunnel Arch") and n not in dead}
    fs = FseqStream(SHOW / f"{ver}.fseq")
    print(f"{ver}: {fs.frames} frames x {fs.step_ms} ms = {fs.duration_s:.1f}s, "
          f"{len(models)} props measured")

    idx = np.concatenate([np.arange(r["first"] - 1, r["last"]) for r in models.values()])
    idx = idx[: len(idx) // 3 * 3]
    # per-prop pixel slices for the flash test
    prop_sl = {n: (r["first"] - 1, r["last"]) for n, r in models.items()}

    def levels(t: float) -> np.ndarray:
        px = fs.at(t)[idx].reshape(-1, 3)
        return px.max(axis=1) / 255.0

    def stat(t: float) -> tuple[float, float, float]:
        lv = levels(t)
        lit = lv > LIT
        return (float(lit.mean()), float(lv[lit].mean()) if lit.any() else 0.0,
                float(np.percentile(lv, 99.5)))

    # ---- per song overview, 4 samples a second ---------------------------
    print(f"\n{'song':34s} {'lit%':>5s} {'level':>6s} {'p99.5':>6s} {'move':>6s}")
    for p in plan:
        ts = np.arange(p["start"] + 0.125, p["end"], 0.25)
        rows, prev, mv = [], None, []
        for t in ts:
            lv = levels(t)
            lit = lv > LIT
            rows.append((lit.mean(), lv[lit].mean() if lit.any() else 0, np.percentile(lv, 99.5)))
            if prev is not None:
                mv.append(np.abs(lv - prev).mean())
            prev = lv
        r = np.array(rows)
        print(f"{p['name']:34s} {100*r[:,0].mean():5.1f} {r[:,1].mean():6.2f} "
              f"{r[:,2].max():6.2f} {np.mean(mv):6.3f}")

    # ---- the arc inside each song -----------------------------------------
    # A chorus or drop must not read dimmer than the verses and breaks around
    # it. The brightness curve does not decide this -- the shader's own
    # content does: on the first full render nine songs had it inverted.
    print("\nsection arc (display mean; * = staged drop; ! = chorus/drop dimmer "
          "than the song's quiet sections):")
    for p in plan:
        d = D.get(p["name"], {})
        secs = d.get("sections", [])
        looks = d.get("section_look", {})
        rows = []
        for i, (t0, lab) in enumerate(secs):
            t1 = secs[i + 1][0] if i + 1 < len(secs) else p["end"]
            if t1 - t0 < 2:
                continue
            v = float(np.mean([levels(t).mean() for t in np.arange(t0 + 0.3, t1, 0.4)]))
            rows.append((t0, lab, v, looks.get(t0, {}).get("shader", "")[7:]))
        quiet = [v for _, lab, v, _ in rows if lab in ("verse", "break", "intro", "outro")]
        ref = float(np.median(quiet)) if quiet else 0.0
        cells = []
        for t0, lab, v, sh in rows:
            loud = lab == "chorus" or t0 in d.get("drops", [])
            flag = "!" if loud and v < ref else ""
            cells.append(f"{lab[:5]}{'*' if t0 in d.get('drops', []) else ''} {v:.2f}{flag} "
                         f"({sh[:12]})")
        print(f"  {p['name'][:22]:22s} " + " | ".join(cells))

    # ---- brightness caps --------------------------------------------------
    print("\nbrightness caps (brightest 0.5% of pixels, over 2 samples/s):")
    for name, d in D.items():
        for a, b, lo, hi in d.get("caps", []):
            p = next(x for x in plan if x["name"] == name)
            a, b = max(a, p["start"]), min(b, p["end"])
            if name.startswith("02"):
                b = min(b, d["reveal"]["hits"][0] - 0.05)
            vals = np.array([stat(t) for t in np.arange(a + 0.25, b, 0.5)])
            print(f"  {name:28s} {a:7.2f}-{b:7.2f} want {lo:.2f}-{hi:.2f}: "
                  f"p99.5 median {np.median(vals[:,2]):.2f} max {vals[:,2].max():.2f}; "
                  f"lit {100*vals[:,0].mean():.0f}%, lit level {vals[:,1].mean():.2f}")

    # ---- the reveal -------------------------------------------------------
    for name, d in D.items():
        rv = d.get("reveal")
        if not rv:
            continue
        print(f"\nreveal ({name}):")
        hits, drop = rv["hits"], rv["drop"]
        marks = []
        for i, h in enumerate(hits):
            marks.append((f"hit {i+1}", h + 0.05))
            if i + 1 < len(hits):
                marks.append(("  gap", (h + hits[i + 1]) / 2 + 0.06))
        marks += [("break", (hits[-1] + drop) / 2 + 0.1), ("DROP", drop + 0.05),
                  ("drop +0.5", drop + 0.5), ("drop +2", drop + 2.0)]
        for lab, t in marks:
            fr = fs.at(t)
            px = fr[idx].reshape(-1, 3)[:, [1, 2, 0]].astype(float)    # B,R,G -> R,G,B
            lv = px.max(axis=1) / 255.0
            lit = lv > LIT
            gold = ((px[:, 0] > 120) & (px[:, 1] > 90) & (px[:, 2] < 80)).mean()
            blue = ((px[:, 2] > 120) & (px[:, 0] < 80)).mean()
            print(f"  {lab:10s} {t:8.3f}  lit {100*lit.mean():5.1f}%  level "
                  f"{(lv[lit].mean() if lit.any() else 0):.2f}  gold {100*gold:4.1f}%  "
                  f"blue {100*blue:4.1f}%")

    # ---- the outro --------------------------------------------------------
    for name, d in D.items():
        of = d.get("outro")
        if not of:
            continue
        p = next(x for x in plan if x["name"] == name)
        nf, (f0, f1) = of["no_flash_from"], of["fade"]

        def flashes(a: float, b: float) -> tuple[int, float]:
            """Props that rise by 0.25+ and fall back within 0.5 s."""
            step = fs.step_ms / 1000.0
            n_fr = int((b - a) / step)
            series = {n: [] for n in prop_sl}
            for k in range(n_fr):
                fr = fs.at(a + k * step)
                for n, (s0, s1) in prop_sl.items():
                    px = fr[s0:s1]
                    px = px[: len(px) // 3 * 3].reshape(-1, 3)
                    series[n].append(px.max(axis=1).mean() / 255.0 if len(px) else 0)
            w = int(0.5 / step)
            cnt = 0
            for n, s in series.items():
                s = np.asarray(s)
                k = 1
                while k < len(s) - w:
                    if s[k] - s[k - 1] > 0.25 and s[k:k + w].min() < s[k] - 0.25:
                        cnt += 1; k += w
                    else:
                        k += 1
            return cnt, (b - a)

        print(f"\noutro ({name}): no flashing from {nf}, fade {f0}-{f1}")
        before = flashes(p["start"] + 50, p["start"] + 70)
        after = flashes(nf + 0.5, min(f1, nf + 40))
        print(f"  prop flashes per minute: song body {60*before[0]/before[1]:.1f}, "
              f"after no_flash_from {60*after[0]/after[1]:.1f}")
        for t in list(np.arange(f0 - 5, f1 + 6, 2.5)) + [p["end"] - 1]:
            s = stat(t)
            print(f"   {t:8.2f}  lit {100*s[0]:5.1f}%  lit level {s[1]:.2f}  p99.5 {s[2]:.2f}")


if __name__ == "__main__":
    main()
