"""A stage lighting designer's read of a rendered sequence, in numbers.

    python design_review.py "<song>" [version]

Not "does it verify" but "is it a good show": contrast, arc, tension before a
drop, colour story, balance across the house, movement, and repetition.
Pixels stored B,R,G -- decoded before any hue.
"""
import sys, json, re, colorsys
from pathlib import Path
from collections import Counter
sys.path.insert(0, "/Users/kristsauders/xlights-mcp-server/custom_tools")
import numpy as np
import channel_map
from fseq_reader import read_fseq

SHOW = Path("/Volumes/XLIGHTS/XLights 2026")


def main(song, ver=None):
    vs = sorted(int(m.group(1)) for p in SHOW.glob(f"{song} (v*).xsq")
                if (m := re.search(r"\(v(\d+)\)\.xsq$", p.name)))
    v = int(ver) if ver else next(x for x in reversed(vs)
                                  if (SHOW / f"{song} (v{x}).fseq").exists())
    G = SHOW / ".generation" / song
    a = json.loads((G / "analysis.json").read_text())
    prov = json.loads((G / "provenance.json").read_text()) if (G / "provenance.json").exists() else {}
    beats = np.array(a["beats"]["beat_times"])
    secs = a["sections"]
    import importlib.util as _il
    _sp = _il.spec_from_file_location("gs", "/Users/kristsauders/xlights-mcp-server/custom_tools/generate_song.py")
    drops = []
    try:
        _txt = Path("/Users/kristsauders/xlights-mcp-server/custom_tools/generate_song.py").read_text()
        _m = re.search(r'SONG_DROPS[^{]*\{(.*?)\n\}', _txt, re.S)
        if _m and song in _m.group(1):
            _seg = _m.group(1).split(song)[1]
            drops = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", _seg.split("]")[0])]
    except Exception:
        pass
    drops = drops or (prov.get("snapped_drops") or [])
    f = read_fseq(SHOW / f"{song} (v{v}).fseq")
    dur = f.duration_s
    cm = channel_map.build()["models"]

    step = 0.20
    ts = np.arange(1.0, dur - 0.3, step)
    frames = [np.frombuffer(f.frame_at(t), dtype=np.uint8) for t in ts]
    mean = np.array([fr.mean() for fr in frames])
    lit = np.array([(fr > 20).mean() for fr in frames])

    # music energy for comparison
    rms = np.array(a["spectrum"]["rms_energy"])
    rt = np.array(a["spectrum"]["rms_times"])
    music = np.interp(ts, rt, rms / (rms.max() + 1e-9))

    print(f"\n================ {song} (v{v})   {dur:.0f}s")
    print("\n--- CONTRAST -------------------------------------------------")
    print(f"  house brightness: mean {mean.mean():5.1f}  p10 {np.percentile(mean,10):5.1f}  "
          f"p90 {np.percentile(mean,90):5.1f}  peak {mean.max():5.1f}")
    print(f"  dynamic range (p90/p10): {np.percentile(mean,90)/max(np.percentile(mean,10),1e-9):5.2f}x")
    dark = mean < 0.10 * mean.max()
    print(f"  time near black (<10% of peak): {100*dark.mean():4.1f}%  "
          f"longest stretch {step*max([len(list(g)) for k,g in __import__('itertools').groupby(dark) if k] or [0]):.1f}s")
    corr = float(np.corrcoef(mean, music)[0, 1])
    print(f"  brightness vs music energy correlation: {corr:+.2f}  "
          f"({'follows the song' if corr>0.4 else 'loosely tied' if corr>0.15 else 'NOT following the song'})")

    print("\n--- ARC (per section) ----------------------------------------")
    print(f"  {'section':>8s} {'label':8s} {'music':>6s} {'light':>6s} {'lit%':>5s} {'top2 hue':>8s} {'movement':>9s}")
    prev_pal = None
    for s in secs:
        m = (ts >= s["start_time"]) & (ts < s["end_time"])
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        hues = Counter()
        for i in idx[::max(1, len(idx)//12)]:
            _fr = frames[i]; _fr = _fr[:len(_fr)//3*3]
            px = _fr.reshape(-1, 3)[:, [1, 2, 0]].astype(float)
            q = px[px.max(axis=1) > 25]
            if len(q):
                for c in q[::max(1, len(q)//400)]:
                    mx, mn = c.max(), c.min()
                    if mx > 0 and (mx-mn)/mx > 0.25:
                        hues[int(colorsys.rgb_to_hsv(*(c/255))[0]*360)//30] += 1
        mv = np.mean([np.abs(frames[i].astype(np.int16)-frames[i-1].astype(np.int16)).mean()
                      for i in idx[1:]]) if len(idx) > 1 else 0
        print(f"  {s['start_time']:8.1f} {str(s.get('label'))[:8]:8s} "
              f"{music[m].mean():6.2f} {mean[m].mean():6.1f} {100*lit[m].mean():4.0f}% "
              f"{(100*sum(v for _,v in hues.most_common(2))/max(sum(hues.values()),1)):4.0f}% {mv:9.1f}")

    print("\n--- TENSION BEFORE EACH DROP ---------------------------------")
    if drops:
        print(f"  {'drop':>7s} {'2s before':>10s} {'at drop':>8s} {'ratio':>6s}  reads as")
        for d in drops:
            pre = mean[(ts >= d-2) & (ts < d-0.15)]
            at = mean[(ts >= d) & (ts < d+1.0)]
            if not pre.size or not at.size:
                continue
            r = at.max()/max(pre.mean(), 1e-9)
            print(f"  {d:7.1f} {pre.mean():10.1f} {at.max():8.1f} {r:6.2f}x  "
                  f"{'a real hit' if r>2 else 'a lift' if r>1.4 else 'BARELY REGISTERS'}")

    print("\n--- BALANCE ACROSS THE HOUSE ---------------------------------")
    groups = {"Mega Tree": ["Mega Tree"], "Matrices": [n for n in cm if "matrix" in n.lower()],
              "Roof/Gutters": [n for n in cm if n.startswith(("Garage Roof", "Single Line"))],
              "Arches/Tunnel": [n for n in cm if "arch" in n.lower() or "tunnel" in n.lower()],
              "Spheres": [n for n in cm if "Hanging Sphere" in n],
              "Orbs": [n for n in cm if n.lower().startswith("orb")],
              "Verticals": [n for n in cm if n.startswith("Vertical Strip")],
              "Flakes/Spinners": [n for n in cm if any(h in n for h in ("Flake","Snowflake","Spinner","Rosa","Illusion"))],
              "Canes/Trees": [n for n in cm if "Cane" in n or "Mini Tree" in n],
              "Floods": [n for n in cm if n.startswith("Flood")]}
    print(f"  {'group':16s} {'props':>5s} {'lit% of song':>13s} {'mean':>6s}")
    for name, props in groups.items():
        props = [p for p in props if p in cm]
        if not props:
            continue
        vals = []
        for i in range(0, len(ts), 3):
            fr = frames[i]
            vals.append(np.mean([fr[max(0,cm[p]["first"]-1):cm[p]["last"]].mean() for p in props[:14]]))
        vals = np.array(vals)
        print(f"  {name:16s} {len(props):5d} {100*np.mean(vals>12):12.0f}% {vals.mean():6.1f}")

    print("\n--- WHO RESTS (a group dark for most of a section) ------------")
    for s2 in secs:
        m = (ts >= s2["start_time"]) & (ts < s2["end_time"])
        if not m.any():
            continue
        idx = np.flatnonzero(m)[::3]
        resting = []
        for name, props in groups.items():
            props = [q for q in props if q in cm]
            if not props:
                continue
            vals = [np.mean([frames[i][max(0,cm[q]["first"]-1):cm[q]["last"]].mean()
                             for q in props[:14]]) for i in idx]
            if np.mean(np.array(vals) > 12) < 0.25:
                resting.append(name)
        print(f"  {s2['start_time']:7.1f} {str(s2.get('label'))[:8]:8s} resting: "
              + (", ".join(resting) if resting else "NOTHING -- every group is lit"))

    txt = (SHOW / f"{song} (v{v}).xsq").read_text()
    print("\n--- COVERAGE HOLES (the dark section vs the bright one) -------")
    # Brightness and effect COUNTS both look normal when a big canvas has lost
    # its bed: measured on Aperture v173, the darkest section (16% lit) held
    # 1979 active effects against 2031 in the brightest, with near-identical
    # effect-name histograms. What separated them was per-element COVERAGE --
    # the fraction of the window with any effect on that element at all. The
    # Tunnel sat at 3.7% there and 100% here, which named the culprit in one
    # line. See "a rest must END when it ends" in docs/new-song-runbook.md.
    _rows = []
    _cur = None
    _dbraw = txt.split("<EffectDB>")[1].split("</EffectDB>")[0]
    for _m in re.finditer(r'<Element [^>]*name="([^"]+)"|<Effect ([^>]*)/>', txt):
        if _m.group(1):
            _cur = _m.group(1)
            continue
        _at = _m.group(2)
        _rows.append((_cur,
                      int(re.search(r'startTime="(\d+)"', _at).group(1)),
                      int(re.search(r'endTime="(\d+)"', _at).group(1))))

    def _cover(el, a, b):
        _sp = sorted((max(a, x[1]), min(b, x[2]))
                     for x in _rows if x[0] == el and x[1] < b and x[2] > a)
        _tot, _end = 0, a
        for _s, _e in _sp:
            _tot += max(0, _e - max(_s, _end))
            _end = max(_end, _e)
        return 100.0 * _tot / max(1, b - a)

    _persec = []
    for s in secs:
        m = (ts >= s["start_time"]) & (ts < s["end_time"])
        if m.sum() >= 4 and s["end_time"] < dur - 1.0:
            _persec.append((100 * lit[m].mean(), s))
    if len(_persec) >= 2:
        _persec.sort()
        _dk, _br = _persec[0][1], _persec[-1][1]
        _cands = sorted({x[0] for x in _rows},
                        key=lambda e: -(cm.get(e, {}).get("count", 0)))[:10]
        print(f"  darkest {_dk['start_time']:.1f}s ({_persec[0][0]:.0f}% lit)  vs  "
              f"brightest {_br['start_time']:.1f}s ({_persec[-1][0]:.0f}% lit)")
        print(f"  {'element':30s} {'dark':>7s} {'bright':>7s}")
        for _el in _cands:
            _a = _cover(_el, int(_dk["start_time"] * 1000), int(_dk["end_time"] * 1000))
            _b = _cover(_el, int(_br["start_time"] * 1000), int(_br["end_time"] * 1000))
            _warn = "  <<< LOST ITS BED" if _b - _a > 40 else ""
            print(f"  {_el[:30]:30s} {_a:6.1f}% {_b:6.1f}%{_warn}")

    print("\n--- REPETITION / MEMORABILITY --------------------------------")
    looks = prov.get("sphere_looks", [])
    if looks:
        c = Counter(l["look"] for l in looks)
        print(f"  sphere looks: {len(looks)}, {len(c)} distinct, most common "
              f"{c.most_common(1)[0][0]} x{c.most_common(1)[0][1]}")
    names = Counter(m.group(1) for m in re.finditer(r'<Effect [^>]*name="([^"]*)"[^>]*startTime=', txt))
    print(f"  effect vocabulary: {len(names)} types; top: "
          + ", ".join(f"{k} {v}" for k, v in names.most_common(6)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
