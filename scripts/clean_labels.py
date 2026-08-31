"""Find the frames where the teacher is wrong, using students that never saw them.

The student's worst frames turned out, on inspection, to be frames where the
LABEL is broken -- the teacher's mask collapsed to a fragment while the student
segmented the whole athlete. IoU cannot tell that apart from a student error,
because IoU grades the student against the label.

So: split the training shots into folds, train one student per fold on the other
folds, and score each frame with the student that never saw it. Where that
student disagrees sharply with the teacher, a colour arbiter breaks the tie --
fighter A wears the blue gi and B the white one, so whichever mask set has the
larger (blue - red) gap between its A and its B is the one describing reality.
The arbiter is not ground truth and shares the teacher's colour assumption, but
it is independent of the disagreement itself, which is what a tie-break needs.

Flagged frames are DROPPED, never relabelled with the student's own prediction.
Replacing labels with model output is self-distillation: it would launder the
student's mistakes into the training set and there would be no signal left to
catch them with.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.student import UNetStudent, metrics, normalise   # noqa: E402
from bjjvision.studentdata import StudentSet                    # noqa: E402

# Balanced by frame count, not by shot count: shot 3 alone is a third of the data.
FOLDS = [[3], [0, 5, 8], [13, 11, 4], [9, 14]]


def contrast(img: np.ndarray, plane: np.ndarray) -> float | None:
    a, b = plane == 1, plane == 2
    if a.sum() < 50 or b.sum() < 50:
        return None
    return float((img[a][:, 0].mean() - img[a][:, 2].mean())
                 - (img[b][:, 0].mean() - img[b][:, 2].mean()))


def train_one(ds, tr_idx, dev, steps, batch, lr, width, seed):
    model = UNetStudent(width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    counts = np.bincount(np.asarray(ds.lab[tr_idx[::37]]).ravel(), minlength=3).astype(np.float64)
    cw = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))) ** 0.5,
                      dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(steps):
        sel = np.sort(rng.choice(tr_idx, batch, replace=False))
        img, lab = np.asarray(ds.img[sel]), np.asarray(ds.lab[sel])
        if rng.random() < 0.5:
            img, lab = img[:, :, ::-1], lab[:, :, ::-1]
        x = normalise(img, dev)
        y = torch.from_numpy(np.ascontiguousarray(lab)).long().to(dev)
        loss = F.cross_entropy(model(x), y, weight=cw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 200 == 0:
            print(f"    step {step+1}/{steps} loss {loss.item():.4f}", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/out/student_gx_320")
    ap.add_argument("--test-shots", default="6,10,12,2")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--iou-max", type=float, default=0.50,
                    help="only frames below this disagree enough to be worth arbitrating")
    ap.add_argument("--out", default="data/out/student_gx_320/label_audit.json")
    a = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = StudentSet(a.data)
    test_shots = {int(s) for s in a.test_shots.split(",") if s}
    pool = np.where(~np.isin(ds.shots, list(test_shots)))[0]
    print(f"device {dev} | auditando {len(pool)} frames de treino em {len(FOLDS)} folds")

    iou = np.full(len(ds), np.nan)
    c_stu = np.full(len(ds), np.nan)
    c_tea = np.full(len(ds), np.nan)
    t0 = time.time()

    for fi, fold in enumerate(FOLDS):
        held = np.where(np.isin(ds.shots, fold))[0]
        tr = np.setdiff1d(pool, held)
        print(f"\n  fold {fi}: shots {fold} fora ({len(held)} frames), treino {len(tr)}", flush=True)
        model = train_one(ds, tr, dev, a.steps, a.batch, a.lr, a.width, seed=fi)
        model.eval()
        with torch.no_grad():
            for s in range(0, len(held), 32):
                sel = held[s:s + 32]
                img = np.asarray(ds.img[sel])
                lg = model(normalise(img, dev))
                iou[sel] = metrics(lg, torch.from_numpy(np.asarray(ds.lab[sel])).long().to(dev))["assigned"]
                pred = lg.argmax(1).cpu().numpy().astype(np.uint8)
                lab = np.asarray(ds.lab[sel])
                for j, k in enumerate(sel):
                    f32 = img[j].astype(np.float32)
                    cs, ct = contrast(f32, pred[j]), contrast(f32, lab[j])
                    if cs is not None:
                        c_stu[k] = cs
                    if ct is not None:
                        c_tea[k] = ct
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()

    ok = ~np.isnan(iou)
    disagree = ok & (iou < a.iou_max)
    arb = ~np.isnan(c_stu) & ~np.isnan(c_tea)
    # The student wins the tie-break, so the teacher's frame is the broken one.
    bad = disagree & arb & (c_stu > c_tea)
    unclear = disagree & ~(arb & (c_stu > c_tea))

    print(f"\n=== auditoria em {time.time()-t0:.0f}s ===")
    print(f"frames auditados            {ok.sum()}")
    print(f"IoU do fold, media          {np.nanmean(iou[ok]):.4f}")
    print(f"discordancia forte (<{a.iou_max})  {disagree.sum()}  ({100*disagree.sum()/ok.sum():.1f}%)")
    print(f"  arbitro escolhe o ALUNO   {bad.sum()}  -> rotulo quebrado, descartar")
    print(f"  arbitro escolhe o PROFESSOR ou empata {unclear.sum()}  -> manter")
    print(f"rotulo com contraste de gi invertido (c_tea<=0): "
          f"{(ok & arb & (c_tea <= 0)).sum()} ({100*(ok & arb & (c_tea <= 0)).sum()/ok.sum():.1f}%)")
    print("\npor shot:")
    for s in sorted(set(ds.shots[pool].tolist())):
        m = (ds.shots == s) & ok
        print(f"  shot {s:2d}  n={m.sum():5d}  IoU {np.nanmean(iou[m]):.3f}  "
              f"descartar {(m & bad).sum():5d} ({100*(m & bad).sum()/m.sum():5.1f}%)")

    Path(a.out).write_text(json.dumps({
        "frames": ds.frames.tolist(),
        "shots": ds.shots.tolist(),
        "fold_iou": np.where(ok, iou, -1).round(4).tolist(),
        "drop": bad.tolist(),
        "params": {"steps": a.steps, "iou_max": a.iou_max, "folds": FOLDS},
    }))
    print(f"\nescrito {a.out}")


if __name__ == "__main__":
    main()
