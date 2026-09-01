"""Does the look actually CHANGE when a section changes?

    python custom_tools/verify_sections.py "Cruise (v11)"

"When sections change the look doesn't always change, like from chorus to
verse and vice versa." A section boundary should read as a decision: the
shaders, the palette and the level should not all continue straight through it.

  shader   did the shader on the big canvases change at the boundary?
  colour   how far the house colour mix moves across it (0 = identical)
  level    ratio of the 2s after to the 2s before
  verdict  a boundary where none of the three moves is not being marked
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from fseq_reader import read_fseq  # noqa: E402

SHOW = Path(json.loads((Path(__file__).resolve().parents[1] / "config.json")
                       .read_text())["show_folders"]["christmas2026"])
BIG = ("Fence Matrix 1", "Fence Matrix 2", "Garage Door Matrix",
       "Sun Room Window Matrix", "Mega Tree", "Garage Roof", "ALL Display")


def main() -> None:
    for name in sys.argv[1:]:
        song = name.rsplit(" (", 1)[0]
        analysis = json.loads(
            (SHOW / ".generation" / song / "analysis.json").read_text())
        secs = [s["start_time"] for s in analysis["sections"]]
        f = read_fseq(SHOW / f"{name}.fseq")
        root = ET.parse(SHOW / f"{name}.xsq").getroot()
        db = [x.text or "" for x in root.find("EffectDB")]

        shaders: list[tuple[float, float, str]] = []
        for el in root.find("ElementEffects"):
            if el.get("name") not in BIG:
                continue
            for layer in el.findall("EffectLayer"):
                for ef in layer.findall("Effect"):
                    if ef.get("name") != "Shader":
                        continue
                    m = re.search(r"Shaders/([^.]+)\.fs", db[int(ef.get("ref"))])
                    shaders.append((int(ef.get("startTime")) / 1000.0,
                                    int(ef.get("endTime")) / 1000.0,
                                    m.group(1) if m else "?"))

        def look(t: float) -> dict:
            """What each big canvas is showing -- per prop, not pooled.

            Pooling them meant one canvas changing counted as "the look
            changed", and the look is what the FACADE does. A boundary where
            one matrix swaps shader while five carry straight on is exactly
            the "doesn't always change" case."""
            out = {}
            for el in root.find("ElementEffects"):
                nm = el.get("name")
                if nm not in BIG:
                    continue
                cur = [n for a, b, n in shaders if a <= t < b]
                out[nm] = tuple(sorted(cur))
            return out

        def mix(a: float, b: float) -> np.ndarray:
            fr = np.zeros(3)
            n = 0
            for t in np.arange(a, b, 0.1):
                x = np.frombuffer(f.frame(min(int(t / 0.025), f.frame_count - 1)),
                                  dtype=np.uint8).astype(float)
                x = x[:len(x) // 3 * 3].reshape(-1, 3).mean(0)
                fr += x
                n += 1
            fr /= max(n, 1)
            s = fr.sum()
            return fr / s if s > 1 else fr

        def lvl(a: float, b: float) -> float:
            return float(np.mean([
                np.frombuffer(f.frame(min(int(t / 0.025), f.frame_count - 1)),
                              dtype=np.uint8).mean()
                for t in np.arange(a, b, 0.1)]))

        print(f"\n=== {name} ===")
        print(f"{'section':>9}  {'shader':>7} {'colour':>7} {'level':>6}   verdict")
        unmarked = 0
        for t in secs:
            if t < 4 or t > (f.frame_count * 0.025) - 4:
                continue
            before, after = look(t - 0.5), look(t + 0.5)
            keys = set(before) | set(after)
            diff = sum(1 for k in keys if before.get(k) != after.get(k))
            changed = diff >= max(1, len(keys) // 2)
            dc = float(np.abs(mix(t - 2, t) - mix(t, t + 2)).sum())
            dl = lvl(t, t + 2) / max(lvl(t - 2, t), 1e-6)
            marked = changed or dc > 0.05 or dl > 1.25 or dl < 0.80
            if not marked:
                unmarked += 1
            print(f"{t:9.2f}  {'yes' if changed else 'no':>7} {dc:7.3f} "
                  f"{dl:6.2f}   {'ok' if marked else 'NOT MARKED'}")
        print(f"  {unmarked} of {len([t for t in secs if 4 < t])} boundaries "
              f"pass with nothing changing")


if __name__ == "__main__":
    main()
