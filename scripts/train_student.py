"""Train the student on the teacher's masks and report a number per held-out shot.

The split is by SHOT, and that is the whole point of the script.

A random split over 18,836 frames would be a lie in two independent ways.
Consecutive frames are near-identical -- a perceptual hash puts frames 0.03 s
apart at 93 differing bits out of 1024, saturating only after about a second --
so a random split puts each test frame's neighbour in train. And the deeper
problem survives even a temporal split: within one shot the camera, lens, mat,
lighting and both gis are fixed, so a model can score well by memorising this
particular blue jacket. Shots are the coarsest boundary the single available
match offers. Even that is optimistic; the honest test is a different match,
which is why `--test-run` exists.
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
from bjjvision.studentdata import ConcatStudentSet, StudentSet  # noqa: E402


def load_data(spec: str):
    """One root, or comma-separated roots pooled through ConcatStudentSet."""
    roots = [r for r in spec.split(",") if r]
    return StudentSet(roots[0]) if len(roots) == 1 else ConcatStudentSet(roots)


def git_sha() -> str:
    """A checkpoint that does not know which code produced it is not reproducible."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:                              # noqa: BLE001
        return "unknown"


def device_of(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_gray(img: np.ndarray, on: bool) -> np.ndarray:
    if not on:
        return img
    g = (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299)
    return np.repeat(g[..., None].astype(np.uint8), 3, axis=-1)


def evaluate(model, ds, idx, dev, bs=16, gray=False) -> dict:
    model.eval()
    acc = {k: [] for k in ("assigned", "best", "flipped", "iou_A", "iou_B", "iou_union")}
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            sel = idx[s:s + bs]
            x = normalise(_maybe_gray(np.asarray(ds.img[sel]), gray), dev)
            y = torch.from_numpy(np.asarray(ds.lab[sel])).long().to(dev)
            m = metrics(model(x), y)
            for k in acc:
                acc[k].append(m[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/out/student_gx_320")
    ap.add_argument("--test-shots", default="6,10,12,2")
    ap.add_argument("--drop-audit", default=None,
                    help="label_audit.json from clean_labels.py; drops the frames whose "
                         "label lost the arbitration against a student that never saw them")
    ap.add_argument("--test-run", default=None,
                    help="second dataset root, evaluated as a true out-of-match test")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--flip", action="store_true",
                    help="horizontal flip; safe here because A/B is gi colour, not side")
    ap.add_argument("--gray", action="store_true",
                    help="ablation: strip colour. The gap to the colour run is the "
                         "share of identity the student is getting from the gi tone, "
                         "which is exactly what will not transfer to white-vs-white.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="data/out/student_ckpt")
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()

    dev = device_of(args.device)
    ds = load_data(args.data)
    test_shots = {int(s) for s in args.test_shots.split(",") if s != ""}
    is_test = np.isin(ds.shots, list(test_shots))
    tr_idx = np.where(~is_test)[0]
    te_idx = np.where(is_test)[0]
    test_ds = load_data(args.test_run) if args.test_run else None
    if not len(te_idx) and test_ds is None:
        raise SystemExit("sem avaliacao: passe --test-shots e/ou --test-run")
    if args.drop_audit:
        audit = json.loads(Path(args.drop_audit).read_text())
        drop = np.array(audit["drop"], dtype=bool)
        if len(drop) < len(ds):
            # The audit was run on the FIRST source of a concat pool (the
            # teacher set); pseudo-label sources carry their own gates and are
            # not covered by it, so they pass through un-dropped.
            print(f"auditoria cobre so os primeiros {len(drop)} frames do pool "
                  f"de {len(ds)}; fontes pseudo seguem com seus proprios gates")
            drop = np.concatenate([drop, np.zeros(len(ds) - len(drop), bool)])
        elif len(drop) != len(ds):
            raise ValueError(f"audit cobre {len(drop)} frames, dataset tem {len(ds)}")
        kept = tr_idx[~drop[tr_idx]]
        print(f"auditoria: {len(tr_idx)-len(kept)} frames de treino descartados "
              f"({100*(len(tr_idx)-len(kept))/len(tr_idx):.1f}%), {len(kept)} restam")
        tr_idx = kept
    print(f"device {dev} | train {len(tr_idx)} frames over shots "
          f"{sorted(set(ds.shots[tr_idx].tolist()))} | test {len(te_idx)} over {sorted(test_shots)}")

    model = UNetStudent(args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.15)
    # Background is ~89% of pixels. Left unweighted the model spends its first
    # thousand steps learning to predict background everywhere.
    counts = np.bincount(np.asarray(ds.lab[tr_idx[::37]]).ravel(), minlength=3).astype(np.float64)
    w = (counts.sum() / (3 * np.maximum(counts, 1))) ** 0.5
    print("class pixel share %s -> loss weights %s" % (np.round(counts / counts.sum(), 4), np.round(w, 3)))
    cw = torch.tensor(w, dtype=torch.float32, device=dev)

    rng = np.random.default_rng(0)
    hist, t0 = [], time.time()
    model.train()
    for step in range(1, args.steps + 1):
        sel = np.sort(rng.choice(tr_idx, args.batch, replace=False))
        img = np.asarray(ds.img[sel])
        lab = np.asarray(ds.lab[sel])
        if args.flip and rng.random() < 0.5:
            img, lab = img[:, :, ::-1], lab[:, :, ::-1]
        x = normalise(_maybe_gray(img, args.gray), dev)
        y = torch.from_numpy(np.ascontiguousarray(lab)).long().to(dev)
        loss = F.cross_entropy(model(x), y, weight=cw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"  step {step:5d}/{args.steps}  loss {loss.item():.4f}  "
                  f"{step / (time.time() - t0):.2f} steps/s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            h = {"step": step}
            if len(te_idx):
                m = evaluate(model, ds, te_idx, dev, args.batch, args.gray)
                h.update(assigned=float(m["assigned"].mean()),
                         best=float(m["best"].mean()),
                         flipped=float(m["flipped"].mean()),
                         union=float(m["iou_union"].mean()))
                print(f"  [eval {step}] assigned {h['assigned']:.4f}  "
                      f"best {h['best']:.4f}  union {h['union']:.4f}  "
                      f"flipped {h['flipped']:.3f}", flush=True)
            if test_ds is not None:
                mr = evaluate(model, test_ds, np.arange(len(test_ds)), dev,
                              args.batch, args.gray)
                h["run_assigned"] = float(mr["assigned"].mean())
                h["run_best"] = float(mr["best"].mean())
                print(f"  [eval {step}] OUT-OF-MATCH assigned {h['run_assigned']:.4f}  "
                      f"best {h['run_best']:.4f}", flush=True)
            hist.append(h)
            model.train()

    payload = {"history": hist, "test_shots": sorted(test_shots),
               "data": args.data}
    if len(te_idx):
        m = evaluate(model, ds, te_idx, dev, args.batch, args.gray)
        print("\n=== held-out shots (same match) ===")
        print("shot     n   IoU_assigned  IoU_best  IoU_union  flipped%  IoU_A   IoU_B")
        for s in sorted(test_shots):
            k = ds.shots[te_idx] == s
            if not k.any():
                continue
            print("%4d %5d       %.4f     %.4f     %.4f     %5.1f   %.4f  %.4f" % (
                s, k.sum(), m["assigned"][k].mean(), m["best"][k].mean(),
                m["iou_union"][k].mean(), 100 * m["flipped"][k].mean(),
                m["iou_A"][k].mean(), m["iou_B"][k].mean()))
        print("%4s %5d       %.4f     %.4f     %.4f     %5.1f   %.4f  %.4f" % (
            "ALL", len(te_idx), m["assigned"].mean(), m["best"].mean(),
            m["iou_union"].mean(), 100 * m["flipped"].mean(),
            m["iou_A"].mean(), m["iou_B"].mean()))
        occ = ds.occlusion[te_idx]
        print("\nby occlusion of the more-hidden athlete:")
        for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
            k = (occ >= lo) & (occ < hi)
            if k.sum() < 10:
                continue
            print("  %3.0f-%3.0f%%  n=%5d  assigned %.4f  best %.4f  flipped %4.1f%%" % (
                lo * 100, hi * 100, k.sum(), m["assigned"][k].mean(),
                m["best"][k].mean(), 100 * m["flipped"][k].mean()))
        payload["final"] = {k: float(v.mean()) for k, v in m.items()}
        payload["per_shot"] = {int(s): {k: float(v[ds.shots[te_idx] == s].mean())
                                        for k, v in m.items()}
                               for s in sorted(test_shots)}

    if test_ds is not None:
        mr = evaluate(model, test_ds, np.arange(len(test_ds)), dev,
                      args.batch, args.gray)
        print("\n=== OUT-OF-MATCH test (%s) ===" % args.test_run)
        print("  n=%d  assigned %.4f  best %.4f  union %.4f  flipped %.1f%%" % (
            len(test_ds), mr["assigned"].mean(), mr["best"].mean(),
            mr["iou_union"].mean(), 100 * mr["flipped"].mean()))
        payload["test_run"] = args.test_run
        payload["final_test_run"] = {k: float(v.mean()) for k, v in mr.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    payload["git_sha"] = sha
    torch.save({"model": model.state_dict(), "width": args.width, "args": vars(args),
                "git_sha": sha},
               out / "student.pt")
    (out / "history.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out/'student.pt'}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
