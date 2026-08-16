"""Read an xLights .fseq (v2) render output back as actual pixel values.

Why this exists: for this whole project the only way to know what a generated
sequence *actually looks like* was to ask Krists to open xLights and watch it.
That made bugs like the "red flash" a multi-round guessing game. xLights can
render headlessly from the CLI:

    /Applications/xLights.app/Contents/MacOS/xLights \
        -s "<show dir>" -r "<sequence>.xsq"

which writes a `.fseq` next to it containing the real, post-render channel
values for every frame. Parsing that turns "does this look wrong?" into a
measurement. Format reference: fseq v2 as written by xLights (FSEQFile.cpp).

Header layout (little-endian):
    0-3   'PSEQ'
    4-5   uint16 offset to channel data
    6     uint8  minor version
    7     uint8  major version (2)
    8-9   uint16 offset to variable headers (== end of block/sparse index)
    10-13 uint32 channel count per *full* frame
    14-17 uint32 frame count
    18    uint8  step time (ms)
    19    uint8  flags
    20    uint8  low nibble = compression type (0 none, 1 zstd, 2 zlib),
                 high nibble = numCompressionBlocks >> 8
    21    uint8  numCompressionBlocks & 0xFF
    22    uint8  numSparseRanges
    23    uint8  flags2
    24-31 uint64 unique id
    then  numCompressionBlocks * 8  : uint32 firstFrame, uint32 blockByteLength
    then  numSparseRanges * 6       : uint24 startChannel(0-based), uint24 count

When sparse ranges are present the per-frame payload contains ONLY those
ranges, concatenated in order -- not the full channel count. That is the
usual case for a real show (xLights only saves channels that controllers
actually use), so `frame_size` below is the sum of the sparse counts.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FseqFile:
    path: Path
    channel_count: int          # channels in a notional full frame
    frame_count: int
    step_time_ms: int
    compression: int            # 0 none, 1 zstd, 2 zlib
    sparse_ranges: list[tuple[int, int]] = field(default_factory=list)
    frame_size: int = 0         # bytes actually stored per frame
    _frames: list[bytes] = field(default_factory=list, repr=False)

    @property
    def duration_s(self) -> float:
        return self.frame_count * self.step_time_ms / 1000.0

    def frame_at(self, t_seconds: float) -> bytes:
        idx = int(round(t_seconds * 1000.0 / self.step_time_ms))
        idx = max(0, min(idx, len(self._frames) - 1))
        return self._frames[idx]

    def frame(self, idx: int) -> bytes:
        return self._frames[idx]

    def __len__(self) -> int:
        return len(self._frames)

    def sparse_offset_of_channel(self, channel_1based: int) -> int | None:
        """Map an absolute (1-based, as xLights shows them) channel number to
        its byte offset inside a stored frame, or None if this channel wasn't
        saved (not covered by any sparse range)."""
        target = channel_1based - 1
        off = 0
        for start, count in self.sparse_ranges:
            if start <= target < start + count:
                return off + (target - start)
            off += count
        return None


def read_fseq(path: str | Path, max_frames: int | None = None,
              frame_stride: int = 1) -> FseqFile:
    """Parse an fseq. `frame_stride` decimates frames (stride 40 at 25ms/frame
    gives one sample per second) -- decompression of a full 3-minute show is
    ~50MB of payload, and for colour/anomaly analysis a coarse sample is
    plenty. Note frames are stored in compressed *blocks*, so a whole block
    still has to be decompressed to reach any frame inside it; the stride
    saves memory, not decompression time."""
    path = Path(path)
    raw = path.read_bytes()

    if raw[0:4] != b"PSEQ":
        raise ValueError(f"{path.name}: not an fseq file (magic={raw[0:4]!r})")
    major = raw[7]
    if major != 2:
        raise ValueError(f"{path.name}: only fseq v2 supported (got v{major})")

    ch_data_offset = struct.unpack_from("<H", raw, 4)[0]
    channel_count = struct.unpack_from("<I", raw, 10)[0]
    frame_count = struct.unpack_from("<I", raw, 14)[0]
    step_time = raw[18]
    comp_byte = raw[20]
    compression = comp_byte & 0x0F
    num_blocks = ((comp_byte >> 4) << 8) | raw[21]
    num_sparse = raw[22]

    pos = 32
    blocks: list[tuple[int, int]] = []
    for _ in range(num_blocks):
        first_frame, length = struct.unpack_from("<II", raw, pos)
        pos += 8
        blocks.append((first_frame, length))

    sparse: list[tuple[int, int]] = []
    for _ in range(num_sparse):
        start = int.from_bytes(raw[pos:pos + 3], "little")
        count = int.from_bytes(raw[pos + 3:pos + 6], "little")
        pos += 6
        sparse.append((start, count))

    frame_size = sum(c for _, c in sparse) if sparse else channel_count

    if compression == 1:
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        def decompress(b): return dctx.decompress(b, max_output_size=1 << 28)
    elif compression == 2:
        import zlib
        def decompress(b): return zlib.decompress(b)
    elif compression == 0:
        def decompress(b): return b
    else:
        raise ValueError(f"unknown compression type {compression}")

    frames: list[bytes] = []
    off = ch_data_offset
    stop = False
    for _first_frame, length in blocks:
        if length == 0 or stop:
            continue
        payload = decompress(raw[off:off + length])
        off += length
        for i in range(0, len(payload), frame_size):
            chunk = payload[i:i + frame_size]
            if len(chunk) < frame_size:
                break
            frames.append(chunk)
        if max_frames is not None and len(frames) >= max_frames * frame_stride:
            stop = True

    if frame_stride > 1:
        frames = frames[::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]

    return FseqFile(path=path, channel_count=channel_count,
                    frame_count=frame_count, step_time_ms=step_time,
                    compression=compression, sparse_ranges=sparse,
                    frame_size=frame_size, _frames=frames)


def frame_color_stats(frame: bytes, lit_threshold: int = 12) -> dict:
    """Summarise one frame's RGB triples: how much is lit, and how much of
    what's lit is (near-)pure red. A prop stuck in an error/garbage state
    shows up as a high `red_frac` with essentially no green or blue."""
    n = len(frame) // 3
    lit = 0
    pure_red = 0
    sum_r = sum_g = sum_b = 0
    for i in range(0, n * 3, 3):
        r, g, b = frame[i], frame[i + 1], frame[i + 2]
        if r > lit_threshold or g > lit_threshold or b > lit_threshold:
            lit += 1
            sum_r += r
            sum_g += g
            sum_b += b
            if r > 100 and g < 32 and b < 32:
                pure_red += 1
    return {
        "pixels": n,
        "lit": lit,
        "lit_frac": lit / n if n else 0.0,
        "red_frac": pure_red / lit if lit else 0.0,
        "mean_rgb": (sum_r / lit, sum_g / lit, sum_b / lit) if lit else (0, 0, 0),
    }


if __name__ == "__main__":
    import sys
    f = read_fseq(sys.argv[1], frame_stride=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    print(f"{f.path.name}: {f.frame_count} frames @ {f.step_time_ms}ms "
          f"({f.duration_s:.1f}s), {f.channel_count} channels, "
          f"{len(f.sparse_ranges)} sparse ranges, {f.frame_size} bytes/frame, "
          f"sampled {len(f)} frames")
    print(f"{'time':>8} {'lit%':>6} {'red% of lit':>12}  mean RGB")
    for i in range(len(f)):
        t = i * stride * f.step_time_ms / 1000.0
        s = frame_color_stats(f.frame(i))
        r, g, b = s["mean_rgb"]
        print(f"{t:8.1f} {s['lit_frac'] * 100:6.1f} {s['red_frac'] * 100:12.1f}"
              f"  ({r:5.0f},{g:5.0f},{b:5.0f})")
