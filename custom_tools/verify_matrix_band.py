"""Find matrices showing a solid horizontal BAND against a textured body.

The artifact Krists reported twice -- "the bottom half of the matrix is a
mostly solid shape and the upper half has a shader" -- is a contiguous run of
rows that is both much brighter than the rest of the matrix AND much flatter
along its length. Half-mean comparisons miss it, because a band in the middle
leaves the two halves equal; that is why an earlier scan reported the shaders
clean when one of them was drawing a seam.

    python custom_tools/verify_matrix_band.py "Song (v12)" ["Song (v13)"...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import channel_map  # noqa: E402
from fseq_reader import read_fseq  # noqa: E402

SHOW = Path(json.loads((Path(__file__).resolve().parents[1] / "config.json")
                       .read_text())["show_folders"]["christmas2026"])
def _geometry() -> dict[str, tuple[int, int]]:
    """Strands x nodes-per-strand, READ FROM THE LAYOUT.

    This was a hardcoded table and it went stale the first time Krists changed
    his hardware: Fence Matrix 1 went from 80x32 to 60x24 when he swapped in
    the real controllers, and every measurement this tool made would have been
    reading the wrong pixels while still printing confident numbers.
    """
    import xml.etree.ElementTree as ET
    out: dict[str, tuple[int, int]] = {}
    try:
        root = ET.parse(next(SHOW.glob("xlights_rgbeffects.xml"))).getroot()
    except (StopIteration, OSError, ET.ParseError):
        return out
    for m in root.iter("model"):
        name = m.get("name", "")
        if "Matrix" not in name or "Tunnel" in name:
            continue
        try:
            strings = int(m.get("NumStrings") or 1)
            per_string = int(m.get("NodesPerString") or 0)
            strands_per = int(m.get("StrandsPerString") or 1)
        except ValueError:
            continue
        strands = strings * strands_per
        per_strand = per_string // max(strands_per, 1)
        if strands >= 4 and per_strand >= 4:
            out[name] = (strands, per_strand)
    return out


GEO = _geometry()
ROWS = 16


def band_score(px: np.ndarray) -> tuple[float, int, int]:
    """Strongest (score, first_row, last_row) solid band in a matrix frame."""
    n = px.shape[1]
    rows = px.reshape(px.shape[0], ROWS, n // ROWS).mean(2).mean(0)
    flat = px.reshape(px.shape[0], ROWS, n // ROWS).mean(2)
    # Flatness ALONG each row: a band is uniform across the prop's width.
    var = np.array([flat[:, r].std() / max(flat[:, r].mean(), 1e-6)
                    for r in range(ROWS)])
    # The top and bottom rows are the matrix's own outline submodels, and a
    # chase along those is WANTED -- "the single strand chases on outlines is a
    # desired effect, it gives movement we want". Only look for a band in the
    # body of the matrix.
    best = (0.0, 0, 0)
    for a in range(1, ROWS - 1):
        for b in range(a, min(a + ROWS // 2, ROWS)):
            inside = rows[a:b + 1]
            outside = np.concatenate([rows[:a], rows[b + 1:]])
            if b >= ROWS - 1 or outside.size < 4 or inside.mean() < 20:
                continue
            # Bright relative to the rest, and flatter than the rest.
            lift = (inside.mean() - outside.mean()) / max(inside.mean(), 1e-6)
            calm = float(var[a:b + 1].mean())
            rough = float(np.concatenate([var[:a], var[b + 1:]]).mean())
            if lift > 0.30 and calm < rough * 0.7:
                score = lift * (1.0 - calm / max(rough, 1e-6))
                if score > best[0]:
                    best = (score, a, b)
    return best


def main() -> None:
    cm = channel_map.build()["models"]
    for name in sys.argv[1:]:
        f = read_fseq(SHOW / f"{name}.fseq")
        hits = []
        for i in range(0, f.frame_count, 20):
            fr = np.frombuffer(f.frame(i), dtype=np.uint8).astype(float)
            for mn, (S, N) in GEO.items():
                if mn not in cm:
                    continue
                a = cm[mn]["first"] - 1
                px = fr[a:a + S * N * 3].reshape(S, N, 3).mean(2)
                if px.mean() < 15:
                    continue
                sc, r0, r1 = band_score(px)
                if sc > 0.18:
                    hits.append((sc, i * 0.025, mn, r0, r1))
        hits.sort(reverse=True)
        print(f"{name}: {len(hits)} frames with a solid band on a matrix")
        for sc, t, mn, r0, r1 in hits[:6]:
            print(f"    {t:7.2f}s  {mn:26s} rows {r0}-{r1} of {ROWS}  score {sc:.2f}")


if __name__ == "__main__":
    main()
