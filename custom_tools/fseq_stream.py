"""Random access into a LARGE .fseq without reading it into memory.

fseq_reader.read_fseq() reads the whole file and is fine for a 3-minute song
(~800 MB). The 29-minute Jackson Lights mix renders to several GB, so this
memory-maps the file and decompresses only the block that holds the frame
asked for (with a one-block cache, so walking forward is cheap).

Frames come back as uint8 arrays at RAW channel offsets -- slice a model as
frame[first-1:last] -- and pixels are stored B,R,G (see fseq_reader notes).
"""
from __future__ import annotations

import mmap
import struct
from bisect import bisect_right
from pathlib import Path

import numpy as np


class FseqStream:
    def __init__(self, path: str | Path):
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        raw = self.m
        if raw[0:4] != b"PSEQ" or raw[7] != 2:
            raise ValueError(f"{path}: not an fseq v2 file")
        self.data_offset = struct.unpack_from("<H", raw, 4)[0]
        self.channels = struct.unpack_from("<I", raw, 10)[0]
        self.frames = struct.unpack_from("<I", raw, 14)[0]
        self.step_ms = raw[18]
        comp = raw[20]
        self.compression = comp & 0x0F
        nblocks = ((comp >> 4) << 8) | raw[21]
        nsparse = raw[22]
        pos = 32
        self.blocks: list[tuple[int, int, int]] = []     # first_frame, offset, length
        off = self.data_offset
        for _ in range(nblocks):
            first, length = struct.unpack_from("<II", raw, pos)
            pos += 8
            if length:
                self.blocks.append((first, off, length))
            off += length
        sparse = []
        for _ in range(nsparse):
            s = int.from_bytes(raw[pos:pos + 3], "little")
            c = int.from_bytes(raw[pos + 3:pos + 6], "little")
            pos += 6
            sparse.append((s, c))
        self.frame_size = sum(c for _, c in sparse) if sparse else self.channels
        self._firsts = [b[0] for b in self.blocks]
        self._cache: tuple[int, bytes] | None = None
        if self.compression == 1:
            import zstandard
            self._dctx = zstandard.ZstdDecompressor()

    @property
    def duration_s(self) -> float:
        return self.frames * self.step_ms / 1000.0

    def frame(self, idx: int) -> np.ndarray:
        idx = max(0, min(self.frames - 1, idx))
        if self.compression == 0:
            o = self.data_offset + idx * self.frame_size
            return np.frombuffer(self.m[o:o + self.frame_size], dtype=np.uint8)
        bi = bisect_right(self._firsts, idx) - 1
        if self._cache is None or self._cache[0] != bi:
            first, off, length = self.blocks[bi]
            blob = self.m[off:off + length]
            if self.compression == 1:
                data = self._dctx.decompress(blob, max_output_size=1 << 30)
            else:
                import zlib
                data = zlib.decompress(blob)
            self._cache = (bi, data)
        first = self.blocks[bi][0]
        data = self._cache[1]
        o = (idx - first) * self.frame_size
        return np.frombuffer(data[o:o + self.frame_size], dtype=np.uint8)

    def at(self, t_s: float) -> np.ndarray:
        return self.frame(int(round(t_s * 1000.0 / self.step_ms)))
