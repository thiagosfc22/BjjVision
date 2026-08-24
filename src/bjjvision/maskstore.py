"""Persist per-frame masks so rendering can happen off the rented GPU.

Rendering inside the tracking loop freezes the visual into whatever the pipeline
decided at run time. Every change of overlay style, crop or typography would mean
renting the box again. Writing the masks out costs a few tens of megabytes and
buys unlimited local re-rendering.

Bit-packed: a 1280x720 boolean mask is 921,600 bytes raw and 115,200 packed,
before zip compression knocks it down further.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class MaskWriter:
    def __init__(self, path: Path, shape: tuple[int, int]):
        self.path = Path(path)
        self.shape = shape
        self._data: dict[str, np.ndarray] = {}
        self._frames: list[int] = []

    def add(self, frame_idx: int, masks: dict[str, np.ndarray]) -> None:
        for fid in ("A", "B"):
            m = masks.get(fid)
            if m is None:
                m = np.zeros(self.shape, dtype=bool)
            self._data[f"{fid}_{frame_idx:08d}"] = np.packbits(m.astype(bool))
        self._frames.append(frame_idx)

    def close(self) -> str:
        np.savez_compressed(self.path, frames=np.array(self._frames, dtype=np.int32),
                            shape=np.array(self.shape, dtype=np.int32), **self._data)
        return str(self.path.with_suffix(".npz"))


class MaskReader:
    def __init__(self, path: Path):
        self.z = np.load(Path(path))
        self.shape = tuple(int(v) for v in self.z["shape"])
        self.frames = [int(f) for f in self.z["frames"]]

    def get(self, frame_idx: int) -> dict[str, np.ndarray]:
        out = {}
        n = self.shape[0] * self.shape[1]
        for fid in ("A", "B"):
            key = f"{fid}_{frame_idx:08d}"
            if key in self.z:
                out[fid] = np.unpackbits(self.z[key])[:n].astype(bool).reshape(self.shape)
        return out

    def __len__(self) -> int:
        return len(self.frames)
