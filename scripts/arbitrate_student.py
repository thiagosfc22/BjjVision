"""When student and teacher disagree, which one is right? Ask the colours.

The student's lowest-IoU held-out frames turned out, on inspection, to be
frames where the teacher had painted the white-gi athlete as A. IoU cannot see
that -- it grades the student against the label. So this arbitrates with a
signal neither model was trained on: A is the blue gi and B is the white one,
measured as (blue - red) in BGR inside each mask. Whichever mask set is more
colour-consistent is the one describing reality.

Not ground truth, and it inherits the same colour assumption as the teacher's
own purity audit. It is enough to tell an argument about identity from an
argument about a missing foot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.student import UNetStudent, metrics, normalise   # noqa: E402
from bjjvision.studentdata import StudentSet                    # noqa: E402


def contrast(img: np.ndarray, plane: np.ndarray) -> float | None:
    """(blueness of A) - (blueness of B). Positive means the labels agree with
    the gi colours; near zero or negative means the two are being confused."""
    a, b = plane == 1, plane == 2
    if a.sum() < 50 or b.sum() < 50:
        return None
    ba = img[a][:, 0].mean() - img[a][:, 2].mean()
    bb = img[b][:, 0].mean() - img[b][:, 2].mean()
    return float(ba - bb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/out/student_ckpt_v1/student.pt")
    ap.add_argument("--data", default="data/out/student_gx_320")
    ap.add_argument("--shots", default="6,10,12,2")
    ap.add_argument("--n", type=int, default=200, help="size of the disagreement tail")
    a = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = UNetStudent(ck["width"]).to(dev); model.load_state_dict(ck["model"]); model.eval()

    ds = StudentSet(a.data)
    idx = np.where(np.isin(ds.shots, [int(s) for s in a.shots.split(",")]))[0]
    scores, preds = [], []
    with torch.no_grad():
        for s in range(0, len(idx), 16):
            sel = idx[s:s + 16]
            x = normalise(np.asarray(ds.img[sel]), dev)
            y = torch.from_numpy(np.asarray(ds.lab[sel])).long().to(dev)
            lg = model(x)
            scores.append(metrics(lg, y)["assigned"])
            preds.append(lg.argmax(1).cpu().numpy().astype(np.uint8))
    scores = np.concatenate(scores); preds = np.concatenate(preds)

    for label, take in (("worst disagreement", np.argsort(scores)[:a.n]),
                        ("random control", np.random.default_rng(0).choice(len(idx), a.n, replace=False))):
        st, te = [], []
        for k in take:
            img = np.asarray(ds.img[idx[k]]).astype(np.float32)
            cs = contrast(img, preds[k]); ct = contrast(img, np.asarray(ds.lab[idx[k]]))
            if cs is None or ct is None:
                continue
            st.append(cs); te.append(ct)
        st, te = np.array(st), np.array(te)
        print(f"\n{label} (n={len(st)}, mean IoU {scores[take].mean():.3f})")
        print(f"  gi-colour contrast, student : {st.mean():+7.2f}   (negative = identity confused)")
        print(f"  gi-colour contrast, teacher : {te.mean():+7.2f}")
        print(f"  student more colour-consistent than teacher on {100*(st>te).mean():.0f}% of frames")
        print(f"  teacher contrast <= 0 (label itself is wrong) on {100*(te<=0).mean():.0f}%"
              f"   | student <= 0 on {100*(st<=0).mean():.0f}%")


if __name__ == "__main__":
    main()
