"""Measure what an individual PROP actually does in a rendered sequence.

    python custom_tools/prop_probe.py "Gooey (v18)" "Corner Verticals" 20 40

Until now verification worked at two levels: the .xsq (what we asked for) and
whole-display colour stats from the .fseq (what came out, in aggregate).
Neither can answer "is this specific prop moving, and in which direction?" --
which is exactly the question that took four rounds to resolve for the
vertical props, because the effect key said `sweep_up` while the buffer style
silently made it travel sideways.

The key measurement here is the **lit-node centroid over time**. Within a
model, channels run in node order along the strand, so the mean index of the
lit nodes is a position along the prop. Watch it across frames:

    rising    -> travel toward the far end of the strand
    falling   -> travel back
    oscillating -> the prop is swinging (a sway)
    flat      -> the prop is lit but NOT moving

For a vertically-mounted strip, strand order is physical height, so this reads
out vertical movement directly -- with no need for any 3D geometry, which is
the part of a visual preview that can't be made accurate.

Requires the measured channel map (custom_tools/channel_map.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import channel_map  # noqa: E402
from fseq_reader import read_fseq  # noqa: E402

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")


def probe(fseq_path: Path, model: str, t0: float, t1: float,
          stride: int = 2) -> dict:
    cmap = channel_map.build()
    info = cmap["models"].get(model)
    resolved = model
    if info is None:
        # A GROUP has no channels of its own -- probe a representative member.
        # For "is this prop moving along its own axis", one member is exactly
        # the right unit: the question is about travel WITHIN a strand, and
        # every member of an outline/vertical group runs the same effect.
        from xlights_mcp.xlights.show import load_show_config
        sc = load_show_config(SHOW_PATH)
        members: list[str] = []
        for g in sc.model_groups:
            if g.name == model:
                members = [m.split("/")[0] for m in g.members]
                break
        # Prefer a member that actually LIGHTS in this window. Picking blindly
        # lands on a disabled member (the generator correctly skips those, so
        # they render nothing) and the probe reports "no light" for a group
        # that is in fact working -- a confusing false negative.
        f_peek = read_fseq(fseq_path, frame_stride=1)
        pi0 = max(0, int(t0 * 1000 / f_peek.step_time_ms))
        pi1 = min(len(f_peek) - 1, int(t1 * 1000 / f_peek.step_time_ms))
        best = None
        for m in members:
            mi = cmap["models"].get(m)
            if mi is None:
                continue
            if best is None:
                best = (m, mi)                       # fallback: first mapped
            energy = 0
            for i in range(pi0, pi1 + 1, max(1, (pi1 - pi0) // 12 or 1)):
                seg = np.frombuffer(f_peek.frame(i), dtype=np.uint8)[
                    mi["first"]:mi["last"] + 1]
                energy += int((seg > 40).sum())
            if energy > 0:
                best = (m, mi)
                break
        if best:
            resolved, info = best
        if info is None:
            raise SystemExit(
                f"{model!r} has no measured channel range and no mapped "
                f"member (checked {len(members)} members)")
        print(f"[{model!r} is a group -- probing member {resolved!r}]")
    lo, hi = info["first"], info["last"] + 1
    model = resolved

    f = read_fseq(fseq_path, frame_stride=1)
    i0 = max(0, int(t0 * 1000 / f.step_time_ms))
    i1 = min(len(f) - 1, int(t1 * 1000 / f.step_time_ms))

    samples = []
    for i in range(i0, i1 + 1, stride):
        raw = np.frombuffer(f.frame(i), dtype=np.uint8)[lo:hi].astype(int)
        n = len(raw) // 3
        if n == 0:
            continue
        px = raw[:n * 3].reshape(-1, 3)
        bright = px.sum(axis=1).astype(float)
        lit = bright > 40
        t = i * f.step_time_ms / 1000.0
        if not lit.any():
            samples.append({"t": t, "lit": 0.0, "centroid": None,
                            "peak_frac": 0.0, "rgb": (0, 0, 0)})
            continue

        # Centroid measured against the frame's OWN peak, not an absolute
        # floor. A dim full-coverage wash underneath (the whole-house shader
        # bed) puts every node above any fixed threshold, so an absolute
        # measure reports "100% lit, centroid pinned at 50%" and hides a
        # perfectly good chase riding on top of it. What matters visually is
        # the BRIGHT part -- that's what the eye tracks -- so the centroid is
        # taken over nodes above 60% of this frame's maximum, and `peak_frac`
        # reports how much of the prop that is. Caught this while about to
        # "fix" a bug that the measurement itself was inventing.
        hi_thresh = max(40.0, bright.max() * 0.6)
        hot = bright >= hi_thresh
        idx = np.nonzero(hot)[0]
        samples.append({
            "t": t,
            "lit": float(lit.mean()),
            "peak_frac": float(hot.mean()),
            "centroid": float(idx.mean() / max(1, n - 1)) if len(idx) else None,
            "rgb": tuple(int(v) for v in px[lit].mean(axis=0).round()),
        })

    cents = [s["centroid"] for s in samples if s["centroid"] is not None]
    verdict = "no light"
    if len(cents) >= 4:
        arr = np.array(cents)
        spread = float(arr.max() - arr.min())
        # count direction reversals -- a sway oscillates, a one-way chase
        # doesn't, and a static fill barely moves at all
        d = np.diff(arr)
        d = d[np.abs(d) > 0.01]
        flips = int(np.count_nonzero(np.diff(np.sign(d)) != 0)) if len(d) > 1 else 0
        if spread < 0.06:
            verdict = f"STATIC (centroid moves only {spread * 100:.0f}% of the prop)"
        elif flips >= 2:
            verdict = (f"OSCILLATING / sway -- {flips} direction changes, "
                       f"travel {spread * 100:.0f}% of the prop")
        else:
            direction = "toward far end" if arr[-1] > arr[0] else "toward near end"
            verdict = (f"TRAVELLING {direction} -- "
                       f"travel {spread * 100:.0f}% of the prop")
    return {"model": model, "channels": (lo, hi), "nodes": (hi - lo) // 3,
            "samples": samples, "verdict": verdict}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    seq = sys.argv[1]
    model = sys.argv[2]
    t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    t1 = float(sys.argv[4]) if len(sys.argv) > 4 else t0 + 12.0
    fseq = SHOW_PATH / f"{seq}.fseq"
    r = probe(fseq, model, t0, t1)
    print(f"{r['model']}  ({r['nodes']} nodes, ch {r['channels'][0]}-{r['channels'][1]})")
    print(f"  {t0:.1f}-{t1:.1f}s   VERDICT: {r['verdict']}\n")
    print(f"  {'time':>7} {'lit%':>6} {'hot%':>6} {'bright pos':>11}  mean RGB")
    for s in r["samples"][:40]:
        pos = "-" if s["centroid"] is None else f"{s['centroid'] * 100:5.1f}%"
        bar = ""
        if s["centroid"] is not None:
            bar = " " * int(s["centroid"] * 24) + "#"
        print(f"  {s['t']:7.2f} {s['lit'] * 100:6.1f} {s.get('peak_frac', 0) * 100:6.1f} "
              f"{pos:>11}  {str(s['rgb']):>16}  {bar}")


if __name__ == "__main__":
    main()
