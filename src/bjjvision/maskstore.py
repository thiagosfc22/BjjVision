"""Persist per-frame masks so rendering can happen off the rented GPU.

Rendering inside the tracking loop freezes the visual into whatever the pipeline
decided at run time. Every change of overlay style or crop would mean renting the
box again. Writing the masks out buys unlimited local re-rendering.

Streamed, not accumulated. The first version held every packed mask in a dict and
called savez_compressed at the end: fine for the 300-frame windows used while
debugging, 5 GB of resident memory for a full 22,000-frame match.

Records are compressed individually rather than the file as a whole. Writing raw
fixed-width records solved the memory problem and created a bandwidth one -- the
same masks that compress 10x inside an npz became 5 GB per match to pull back off
the rented box. Per-record deflate keeps the write streaming and the read seekable
while restoring most of that, at the cost of carrying an offset table.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np


class MaskWriter:
    def __init__(self, path: Path, shape: tuple[int, int]):
        self.path = Path(path).with_suffix(".bin")
        self.idx_path = self.path.with_suffix(".idx.json")
        self.shape = shape
        self.n_px = shape[0] * shape[1]
        self.stride = (self.n_px + 7) // 8          # bytes per packed mask
        self._frames: list[int] = []
        self._offsets: list[int] = []          # byte offset of each frame's record pair
        self._sizes: list[list[int]] = []      # compressed length per mask
        self._pos = 0
        self._fh = self.path.open("wb")

    def add(self, frame_idx: int, masks: dict[str, np.ndarray]) -> None:
        self._offsets.append(self._pos)
        sizes = []
        for fid in ("A", "B"):
            m = masks.get(fid)
            if m is None:
                m = np.zeros(self.shape, dtype=bool)
            packed = np.packbits(np.ascontiguousarray(m, dtype=bool)).tobytes()
            if len(packed) != self.stride:         # refuse to write a ragged record
                raise ValueError(f"mask {fid} at frame {frame_idx} packed to "
                                 f"{len(packed)} bytes, expected {self.stride}")
            buf = zlib.compress(packed, 6)
            self._fh.write(buf)
            sizes.append(len(buf))
            self._pos += len(buf)
        self._sizes.append(sizes)
        self._frames.append(frame_idx)

    def close(self) -> str:
        self._fh.close()
        self.idx_path.write_text(json.dumps({
            "shape": list(self.shape), "stride": self.stride,
            "order": ["A", "B"], "frames": self._frames,
            "offsets": self._offsets, "sizes": self._sizes,
        }))
        return str(self.path)


class MaskReader:
    def __init__(self, path: Path):
        path = Path(path)
        self.path = path.with_suffix(".bin")
        meta = json.loads(path.with_suffix(".idx.json").read_text())
        self.shape = tuple(meta["shape"])
        self.stride = meta["stride"]
        self.order = meta["order"]
        self.frames = meta["frames"]
        self._offsets = meta["offsets"]
        self._sizes = meta["sizes"]
        self._pos = {f: i for i, f in enumerate(self.frames)}
        self._fh = self.path.open("rb")
        self.n_px = self.shape[0] * self.shape[1]

    def get(self, frame_idx: int) -> dict[str, np.ndarray]:
        i = self._pos.get(frame_idx)
        if i is None:
            return {}
        out = {}
        off = self._offsets[i]
        for k, fid in enumerate(self.order):
            self._fh.seek(off)
            raw = np.frombuffer(zlib.decompress(self._fh.read(self._sizes[i][k])),
                                dtype=np.uint8)
            out[fid] = np.unpackbits(raw)[:self.n_px].astype(bool).reshape(self.shape)
            off += self._sizes[i][k]
        return out

    def close(self) -> None:
        self._fh.close()

    def __len__(self) -> int:
        return len(self.frames)

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
