"""Turn the SAM2 teacher's output into a dataset a student can train on.

`masks.bin` is the asset. What it is not, yet, is a dataset: the masks live in
six chunk directories indexed by frame number, the frames live in a 320 MB
video, and roughly one frame in thirteen is poisoned -- the teacher put a mask
on the crowd, on a bystander walking past the scoreboard, or on nothing at all.

Three things are decided here and each was measured first.

**The frames come from `data/interim/<slug>_norm.mp4`.** Verified rather than
assumed: that file has exactly 23,306 frames, the count the pipeline reports,
and the mean BGR inside mask A at frame 2500 is (79, 40, 28) against (164, 157,
134) inside B -- the blue gi and the white gi, in the order the index claims.

**Resolution is not the bottleneck, so stop paying for it.** Downsampling a
ground-truth mask to 320x180 and back costs 2.7% IoU; to 512x288, 1.8%. A
student that reached either ceiling would already be finished. Training at 720p
buys accuracy the student will not use for 16x the memory.

**The quality filter keeps occlusion.** The obvious filter -- drop frames where
a mask is small -- throws away exactly the frames the whole project is about,
because an athlete flattened under side control genuinely occupies 5,000 px.
Measured: a min-area floor at 1% of frame rejected 16% of the deepest-occlusion
band. The filter used instead is colour purity, tracker confidence, cross-mask
IoU and a non-empty check, which keeps 83-96% of every occlusion band evenly
(92.4% overall) while still zeroing shot 7, where A's purity is 0.106 because
its mask sat on a spectator.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import cv2
import numpy as np

from .maskstore import MaskReader

# Kept as a named default so the filter that produced a dataset is recoverable
# from the manifest rather than from memory.
QUALITY = {"min_purity": 0.70, "max_cross_iou": 0.10, "min_confidence": 0.70}


def _load_features(run_dir: Path) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    cols = ["frame", "shot_id", "mask_iou", "A_mask_area", "B_mask_area",
            "A_purity", "B_purity", "track_confidence", "occl_a_by_b", "occl_b_by_a"]
    parts = sorted(glob.glob(str(run_dir / "chunk_*" / "features.parquet"))) or \
            sorted(glob.glob(str(run_dir / "features.parquet")))
    acc: dict[str, list] = {c: [] for c in cols}
    for p in parts:
        t = pq.read_table(p, columns=cols)
        for c in cols:
            acc[c].append(t[c].to_numpy(zero_copy_only=False))
    D = {c: np.concatenate(v) for c, v in acc.items()}
    order = np.argsort(D["frame"])
    return {c: v[order] for c, v in D.items()}


def quality_mask(D: dict[str, np.ndarray], q: dict = QUALITY) -> np.ndarray:
    """Frames whose teacher output is trustworthy enough to imitate."""
    purity = np.minimum(D["A_purity"], D["B_purity"])
    empty = (D["A_mask_area"] <= 1e-6) | (D["B_mask_area"] <= 1e-6)
    return ((purity >= q["min_purity"])
            & (D["mask_iou"] <= q["max_cross_iou"])
            & (D["track_confidence"] >= q["min_confidence"])
            & ~empty)


def build(run_dir: str | Path, video: str | Path, out: str | Path,
          size: tuple[int, int] = (320, 180), stride: int = 1,
          drop_shots: tuple[int, ...] = (), q: dict = QUALITY) -> dict:
    """Write `<out>/img.u8`, `<out>/lab.u8` memmaps plus a manifest.

    Labels are a single uint8 plane -- 0 background, 1 fighter A, 2 fighter B --
    rather than two binary planes. The teacher's masks overlap on 1.9% of frames
    (median 0 px, max 8 px on the validated window), so a per-pixel argmax loses
    almost nothing and makes the student's output directly comparable to a
    softmax. Overlap is resolved toward A, arbitrarily and consistently.
    """
    run_dir, out = Path(run_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    W, H = size

    D = _load_features(run_dir)
    keep = quality_mask(D, q)
    for s in drop_shots:
        keep &= D["shot_id"] != s
    frames = D["frame"][keep][::stride]
    shots = D["shot_id"][keep][::stride]
    occl = np.maximum(D["occl_a_by_b"], D["occl_b_by_a"])[keep][::stride]
    wanted = {int(f): i for i, f in enumerate(frames)}
    n = len(frames)

    readers = []
    for d in sorted(glob.glob(str(run_dir / "chunk_*"))) or [str(run_dir)]:
        R = MaskReader(Path(d) / "masks.bin")
        readers.append((set(R.frames), R))

    img = np.lib.format.open_memmap(out / "img.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, H, W, 3))
    lab = np.lib.format.open_memmap(out / "lab.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, H, W))

    cap = cv2.VideoCapture(str(video))
    i, written = -1, 0
    while written < n:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        j = wanted.get(i)
        if j is None:
            continue
        masks = {}
        for fs, R in readers:
            if i in fs:
                masks = R.get(i)
                break
        plane = np.zeros(frame.shape[:2], np.uint8)
        b = masks.get("B")
        if b is not None:
            plane[b] = 2
        a = masks.get("A")
        if a is not None:
            plane[a] = 1                      # A wins the overlap, consistently
        img[j] = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        # INTER_NEAREST on the label plane: averaging class ids would invent a
        # class 1 boundary between background and B.
        lab[j] = cv2.resize(plane, (W, H), interpolation=cv2.INTER_NEAREST)
        written += 1
    cap.release()
    for _, R in readers:
        R.close()
    img.flush(); lab.flush()

    manifest = {
        "video": str(video), "run_dir": str(run_dir), "size": [W, H],
        "stride": stride, "n": int(written), "quality": q,
        "drop_shots": list(drop_shots),
        "frames": frames.astype(int).tolist(),
        "shots": shots.astype(int).tolist(),
        "occlusion": np.round(occl, 4).tolist(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class StudentSet:
    """Read side of the memmaps. Indexing returns (HWC uint8, HW uint8)."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.meta = json.loads((root / "manifest.json").read_text())
        self.img = np.load(root / "img.npy", mmap_mode="r")
        self.lab = np.load(root / "lab.npy", mmap_mode="r")
        self.shots = np.array(self.meta["shots"])
        self.frames = np.array(self.meta["frames"])
        self.occlusion = np.array(self.meta["occlusion"])

    def __len__(self) -> int:
        return len(self.img)

    def __getitem__(self, i):
        return self.img[i], self.lab[i]
