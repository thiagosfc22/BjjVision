"""Does the per-frame student need a temporal model? Measure, do not assume.

The argument for training on windows instead of single frames is that identity
and mask should be temporally coherent. That is only worth its cost if the
per-frame student is actually incoherent, so this compares consecutive-frame
stability of the student against the same statistic on the teacher's own masks
over the same runs. The teacher is a video model with memory attention, so its
number is the floor that motion alone imposes; anything the student loses
beyond it is flicker the student invented.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.student import UNetStudent, normalise   # noqa: E402
from bjjvision.studentdata import StudentSet           # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/out/student_ckpt_v1/student.pt")
    ap.add_argument("--data", default="data/out/student_gx_320")
    ap.add_argument("--shots", default="6,10,12,2")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--len", type=int, default=60)
    a = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = UNetStudent(ck["width"]).to(dev); model.load_state_dict(ck["model"]); model.eval()

    ds = StudentSet(a.data)
    shots = {int(s) for s in a.shots.split(",")}
    idx = np.where(np.isin(ds.shots, list(shots)))[0]
    # contiguous stretches in ORIGINAL frame numbering, so "consecutive" is real
    fr = ds.frames[idx]
    brk = np.where(np.diff(fr) != 1)[0]
    segs = [s for s in np.split(idx, brk + 1) if len(s) >= a.len]
    rng = np.random.default_rng(0)
    segs = [segs[i] for i in rng.choice(len(segs), min(a.runs, len(segs)), replace=False)]

    st_stab, te_stab, id_switch = [], [], []
    with torch.no_grad():
        for seg in segs:
            seg = seg[:a.len]
            preds = []
            for s in range(0, len(seg), 16):
                x = normalise(np.asarray(ds.img[seg[s:s + 16]]), dev)
                preds.append(model(x).argmax(1).cpu().numpy().astype(np.uint8))
            P = np.concatenate(preds)
            G = np.asarray(ds.lab[seg])
            for t in range(1, len(seg)):
                st_stab.append(0.5 * (iou(P[t] == 1, P[t - 1] == 1) + iou(P[t] == 2, P[t - 1] == 2)))
                te_stab.append(0.5 * (iou(G[t] == 1, G[t - 1] == 1) + iou(G[t] == 2, G[t - 1] == 2)))
                # an identity switch shows up as the cross term beating the direct one
                direct = 0.5 * (iou(P[t] == 1, P[t - 1] == 1) + iou(P[t] == 2, P[t - 1] == 2))
                cross = 0.5 * (iou(P[t] == 1, P[t - 1] == 2) + iou(P[t] == 2, P[t - 1] == 1))
                id_switch.append(cross > direct)

    st, te, sw = np.array(st_stab), np.array(te_stab), np.array(id_switch)
    print("consecutive-frame mask IoU over %d runs of %d frames (%d pairs)" % (len(segs), a.len, len(st)))
    print("  teacher (SAM2, has memory) : mean %.4f  p05 %.4f" % (te.mean(), np.percentile(te, 5)))
    print("  student (per-frame, blind) : mean %.4f  p05 %.4f" % (st.mean(), np.percentile(st, 5)))
    print("  gap the student invents    : %.4f" % (te.mean() - st.mean()))
    print("  frames where the student flips A/B vs its own previous frame: %.2f%%" % (100 * sw.mean()))


if __name__ == "__main__":
    main()
