"""Discover each model's channel range by MEASUREMENT, not by inference.

Everything downstream (per-prop probes, the visual preview) needs to know
which channels in a rendered `.fseq` belong to which model. The obvious route
is to compute it: parse `xlights_networks.xml`, sum each controller's
`MaxChannels`, resolve every model's `StartChannel="!Controller:offset"`.
That was tried and **does not work** -- summing the networks gives 128,361
channels while the rendered fseq reports 153,230, a 24,869 discrepancy. xLights'
real allocation involves details (inactive outputs, auto-sizing, gaps) that
aren't worth reverse-engineering, and any error would silently corrupt every
measurement built on top.

So instead: ask xLights. This builds a probe sequence that BINARY-ENCODES each
model's index -- in time slot `f`, every model whose index has bit `f` set is
lit white -- renders it with the real xLights engine, and reads the resulting
fseq back. Each channel's on/off pattern across the slots spells out the index
of the model that owns it. 12 slots covers 4095 models; this show has ~250.

The result is exact by construction, includes whatever quirks the real
allocator has, and self-validates: a channel whose decoded index is out of
range, or a model that never lights, is visible in the output rather than
silently wrong. This is the same "render it and measure" approach that found
the palette bugs, applied to geometry instead of colour.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fseq_reader import read_fseq  # noqa: E402
from xlights_mcp.xlights.palettes import ColorPalette  # noqa: E402
from xlights_mcp.xlights.show import load_show_config  # noqa: E402
from xlights_mcp.xlights.xsq_writer import (EffectPlacement,  # noqa: E402
                                             SequenceSpec, write_xsq)

SHOW_PATH = Path("/Volumes/XLIGHTS/XLights 2026")
XLIGHTS = Path("/Applications/xLights.app/Contents/MacOS/xLights")
PROBE_NAME = "_channel_probe"
SLOT_MS = 1000
N_SLOTS = 12
WHITE = ColorPalette(colors=["#FFFFFF"], active_colors=[1])


def build_probe(models: list[str], out_path: Path, media: Path) -> None:
    effects: list[EffectPlacement] = []
    for idx, name in enumerate(models, start=1):     # 1-based: 0 means "no model"
        for bit in range(N_SLOTS):
            if idx & (1 << bit):
                s = bit * SLOT_MS + 100
                effects.append(EffectPlacement(
                    model_name=name, layer=0, effect_name="On",
                    start_time_ms=s, end_time_ms=s + SLOT_MS - 200,
                    settings={"T_TEXTCTRL_Fadein": "0", "T_TEXTCTRL_Fadeout": "0",
                              "E_TEXTCTRL_Eff_On_End": "100",
                              "E_TEXTCTRL_Eff_On_Start": "100"},
                    palette=WHITE))
    spec = SequenceSpec(
        song_title=PROBE_NAME, artist="", album="", media_file=str(media),
        duration_ms=N_SLOTS * SLOT_MS + 2000, timing_ms=25,
        palettes=[WHITE], effects=effects, timing_tracks=[])
    write_xsq(spec, load_show_config(SHOW_PATH), out_path)


def decode(fseq_path: Path, models: list[str]) -> dict:
    f = read_fseq(fseq_path, frame_stride=1)
    step = f.step_time_ms
    codes = np.zeros(f.frame_size, dtype=np.int32)
    for bit in range(N_SLOTS):
        # sample the middle of the slot, well inside the effect
        t_ms = bit * SLOT_MS + SLOT_MS // 2
        idx = min(int(t_ms / step), len(f) - 1)
        frame = np.frombuffer(f.frame(idx), dtype=np.uint8).astype(np.int32)
        codes |= ((frame > 40).astype(np.int32) << bit)

    ranges: dict[str, dict] = {}
    for i, name in enumerate(models, start=1):
        hits = np.nonzero(codes == i)[0]
        if len(hits) == 0:
            continue
        ranges[name] = {"first": int(hits[0]), "last": int(hits[-1]),
                        "count": int(len(hits))}
    unmapped = int(np.count_nonzero((codes != 0) &
                                    (codes > len(models))))
    return {"models": ranges, "frame_size": f.frame_size,
            "sparse_ranges": f.sparse_ranges, "unmapped_channels": unmapped,
            "lit_channels": int(np.count_nonzero(codes))}


def build(force: bool = False) -> dict:
    cache = SHOW_PATH / ".generation" / "channel_map.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    sc = load_show_config(SHOW_PATH)
    models = sorted(m.name for m in sc.models)
    probe_xsq = SHOW_PATH / f"{PROBE_NAME}.xsq"
    media = next(SHOW_PATH.glob("*.mp3"))
    print(f"probing {len(models)} models over {N_SLOTS} slots ...")
    build_probe(models, probe_xsq, media)
    subprocess.run([str(XLIGHTS), "-s", str(SHOW_PATH), "-r", str(probe_xsq)],
                   capture_output=True, text=True)
    fseq = probe_xsq.with_suffix(".fseq")
    if not fseq.exists():
        raise SystemExit("probe render produced no fseq")
    result = decode(fseq, models)
    result["model_order"] = models
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    # The probe files are scratch -- don't leave them in the show folder.
    for p in (probe_xsq, fseq):
        try:
            p.unlink()
        except OSError:
            pass
    return result


if __name__ == "__main__":
    res = build(force="--force" in sys.argv)
    m = res["models"]
    print(f"mapped {len(m)} models; {res['lit_channels']} channels lit; "
          f"{res['unmapped_channels']} undecodable")
    for name in list(m)[:10]:
        r = m[name]
        print(f"   {name:34s} ch {r['first']:6d}-{r['last']:6d} ({r['count']})")
