"""Render the student's masks next to the teacher's, on held-out shots.

A number on its own has been wrong here before. This draws the student's
prediction (top row of each pair) against the teacher's label (bottom) so the
error mode is visible: a missing limb and a swapped athlete both cost IoU, and
only one of them matters.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.student import UNetStudent, normalise   # noqa: E402
from bjjvision.studentdata import StudentSet           # noqa: E402

COL = {1: (0, 0, 255), 2: (0, 255, 0)}


def paint(img: np.ndarray, plane: np.ndarray) -> np.ndarray:
    out = img.copy()
    for cid, col in COL.items():
        m = plane == cid
        if not m.any():
            continue
        out[m] = (0.45 * np.array(col) + 0.55 * out[m]).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, col, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/out/student_ckpt_v1/student.pt")
    ap.add_argument("--data", default="data/out/student_gx_320")
    ap.add_argument("--shots", default="6,10,12,2")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pick", default="spread", choices=["spread", "worst"])
    ap.add_argument("--out", default="data/out/student_qualitative.jpg")
    a = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = UNetStudent(ck["width"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = StudentSet(a.data)
    shots = {int(s) for s in a.shots.split(",")}
    idx = np.where(np.isin(ds.shots, list(shots)))[0]

    if a.pick == "worst":
        from bjjvision.student import metrics
        sc = []
        with torch.no_grad():
            for s in range(0, len(idx), 16):
                sel = idx[s:s + 16]
                x = normalise(np.asarray(ds.img[sel]), dev)
                y = torch.from_numpy(np.asarray(ds.lab[sel])).long().to(dev)
                sc.append(metrics(model(x), y)["assigned"])
        sc = np.concatenate(sc)
        pick = idx[np.argsort(sc)[:a.n]]
    else:
        pick = idx[np.linspace(0, len(idx) - 1, a.n).astype(int)]

    tiles = []
    with torch.no_grad():
        for i in pick:
            img = np.asarray(ds.img[i])
            pred = model(normalise(img[None], dev)).argmax(1)[0].cpu().numpy().astype(np.uint8)
            gt = np.asarray(ds.lab[i])
            row = np.hstack([paint(img, pred), paint(img, gt)])
            cv2.putText(row, "student f%d shot%d" % (ds.frames[i], ds.shots[i]),
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(row, "teacher", (img.shape[1] + 6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(row)
    grid = np.vstack(tiles)
    cv2.imwrite(a.out, grid)
    print(a.out, grid.shape)


if __name__ == "__main__":
    main()
