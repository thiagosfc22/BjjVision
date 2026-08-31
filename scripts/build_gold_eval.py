"""Build the GOLD evaluation set: ~200 eye-verified frames from the held-out fights.

    python scripts/build_gold_eval.py cards      # sample frames, render judgment cards
    python scripts/build_gold_eval.py judge      # VLM rules each card (resumable)
    python scripts/build_gold_eval.py assemble   # gold dataset from the verdicts
    python scripts/build_gold_eval.py eval       # score a checkpoint against gold

Why this exists: v4's held-out numbers (0.18 / 0.30) turned out to measure
agreement with a CONTAMINATED reference -- the teacher fused athletes and
masked bystanders on exactly the domains reserved for evaluation. Until a
trustworthy ruler exists, no number on the hard domains means anything.

Design decisions:
- Every frame carries TWO candidate labellings, teacher and student v4,
  because the side-by-side renders showed them wrong in DIFFERENT ways; the
  VLM judge picks whichever is actually correct, or neither. Provenance is
  recorded per gold frame so any evaluation can split by label source --
  student-sourced labels mildly favour student-lineage models, and hiding
  that would be worse than having it.
- Sampling covers the teacher's HOLES too (stretches where no chunk
  calibrated): those are the hard parts, and an eval set that skips them
  grades on a curve.
- The judge's default answer is 'none'. A 120-frame honest gold set beats a
  200-frame contaminated one; contamination is the disease this treats.
- paulista23 is white-vs-white: per-frame A/B identity is not predictable
  from a single frame, so the manifest marks it identity_unreliable and the
  eval reports permutation-invariant IoU (best) as the headline there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bjjvision.studentdata import StudentSet          # noqa: E402
from bjjvision.triage import paint                    # noqa: E402

OUT = ROOT / "data" / "out" / "gold_eval"
SIZE = (320, 180)

# (slug, n from teacher-labelled frames, n from teacher holes, note for judge)
PLAN = [
    ("paulista23-master2", 90, 30,
     "regional gym, BOTH athletes in WHITE gis, crowded venue with white "
     "tables and other fights on neighbouring mats"),
    ("wardzinski-ferreira", 60, 20,
     "dark modern arena with a bright magenta LED wall; one athlete in a "
     "BLUE gi (easily lost in the dark), the other in WHITE"),
]
IDENTITY_UNRELIABLE = {"paulista23-master2"}


def chunk_holes(slug: str, n_frames: int) -> list[tuple[int, int]]:
    covered = []
    for c in sorted((ROOT / "data" / "out" / slug).glob("chunk_*")):
        idx = json.loads((c / "masks.idx.json").read_text())
        if idx["frames"]:
            a, b = c.name.split("_")[1:3]
            covered.append((int(a), int(b)))
    holes, pos = [], 0
    for a, b in sorted(covered):
        if a > pos:
            holes.append((pos, a))
        pos = max(pos, b)
    if pos < n_frames:
        holes.append((pos, n_frames))
    return holes


def load_v4():
    import torch
    from bjjvision.student import UNetStudent
    ck = torch.load(ROOT / "data/out/student_ckpt_v4/student.pt",
                    map_location="cpu", weights_only=False)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = UNetStudent(ck["width"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, dev


def phase_cards() -> None:
    import torch
    from bjjvision.student import normalise
    model, dev = load_v4()
    OUT.mkdir(parents=True, exist_ok=True)
    samples = []
    for slug, n_teach, n_hole, note in PLAN:
        ds = StudentSet(ROOT / f"data/out/teacher_{slug}_320")
        n_frames = json.loads((ROOT / f"data/interim/{slug}.json").read_text())["n_frames"]
        pick_t = ds.frames[np.linspace(0, len(ds) - 1, n_teach).astype(int)]
        holes = chunk_holes(slug, n_frames)
        pool = np.concatenate([np.arange(a, b) for a, b in holes]) if holes else np.array([])
        pick_h = (pool[np.linspace(0, len(pool) - 1, min(n_hole, len(pool))).astype(int)]
                  if len(pool) else np.array([], int))
        teach_lab = {int(f): i for i, f in enumerate(ds.frames)}

        cap = cv2.VideoCapture(str(ROOT / f"data/interim/{slug}_norm.mp4"))
        for f in sorted(set(pick_t.tolist()) | set(int(x) for x in pick_h)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, fr = cap.read()
            if not ok:
                continue
            small = cv2.resize(fr, SIZE, interpolation=cv2.INTER_AREA)
            with torch.no_grad():
                pred = model(normalise(small[None], dev)).argmax(1)[0].cpu().numpy().astype(np.uint8)
            ti = teach_lab.get(int(f))
            tplane = np.asarray(ds.lab[ti]) if ti is not None else None

            up = lambda im: cv2.resize(im, (640, 360), interpolation=cv2.INTER_NEAREST)  # noqa: E731
            panels = [up(small), up(paint(small, tplane)) if tplane is not None
                      else up(small.copy()), up(paint(small, pred))]
            if tplane is None:
                cv2.putText(panels[1], "SEM PROFESSOR", (180, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            for p, name in zip(panels, ("ORIGINAL", "PROFESSOR", "STUDENT")):
                cv2.putText(p, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
            card = np.hstack(panels)
            key = f"{slug}_{f:06d}"
            cv2.imwrite(str(OUT / f"card_{key}.jpg"), card)
            np.savez_compressed(OUT / f"cand_{key}.npz", teacher=tplane
                                if tplane is not None else np.zeros((1,), np.uint8),
                                student=pred, has_teacher=tplane is not None)
            samples.append({"slug": slug, "frame": int(f),
                            "has_teacher": tplane is not None, "note": note})
        cap.release()
        print(f"{slug}: {sum(1 for s in samples if s['slug'] == slug)} cartoes", flush=True)
    (OUT / "samples.json").write_text(json.dumps(samples, indent=1))
    print(f"total {len(samples)} cartoes -> {OUT}")


def phase_judge() -> None:
    from bjjvision.llm_supervisor import MODEL, LlmSupervisor
    sup = LlmSupervisor({"llm": {"enabled": True, "model": MODEL,
                                 "max_calls_per_minute": 12}})
    if not sup.enabled:
        raise SystemExit("supervisor indisponivel -- ANTHROPIC_API_KEY?")
    samples = json.loads((OUT / "samples.json").read_text())
    vpath = OUT / "verdicts.json"
    verdicts = json.loads(vpath.read_text()) if vpath.exists() else {}
    todo = [s for s in samples if f"{s['slug']}_{s['frame']:06d}" not in verdicts]
    print(f"{len(todo)} cartoes a julgar ({len(verdicts)} ja julgados)", flush=True)
    for k, s in enumerate(todo):
        key = f"{s['slug']}_{s['frame']:06d}"
        card = cv2.imread(str(OUT / f"card_{key}.jpg"))
        note = s["note"]
        if not s["has_teacher"]:
            note += ("; the TEACHER panel is EMPTY because no teacher label "
                     "exists here -- judge teacher as quality='wrong' and "
                     "consider only the student")
        v = sup.gold(card, {"slug": s["slug"], "frame": s["frame"], "note": note})
        if v is None:
            print(f"  {key}: chamada falhou, seguindo", flush=True)
            continue
        verdicts[key] = v
        vpath.write_text(json.dumps(verdicts, indent=1))
        if (k + 1) % 20 == 0:
            ch = [x["chosen"] for x in verdicts.values()]
            print(f"  {k+1}/{len(todo)}  teacher {ch.count('teacher')} | "
                  f"student {ch.count('student')} | none {ch.count('none')} | "
                  f"custo ~${sup.stats.est_cost_usd:.2f}", flush=True)
    ch = [x["chosen"] for x in verdicts.values()]
    print(f"julgamento completo: teacher {ch.count('teacher')} | "
          f"student {ch.count('student')} | none {ch.count('none')} | "
          f"custo ~${sup.stats.est_cost_usd:.2f}")


def phase_assemble() -> None:
    samples = json.loads((OUT / "samples.json").read_text())
    verdicts = json.loads((OUT / "verdicts.json").read_text())
    keep = []
    for s in samples:
        key = f"{s['slug']}_{s['frame']:06d}"
        v = verdicts.get(key)
        if v and v["chosen"] != "none":
            keep.append((s, v, key))
    n = len(keep)
    W, H = SIZE
    img = np.lib.format.open_memmap(OUT / "img.npy", mode="w+", dtype=np.uint8,
                                    shape=(n, H, W, 3))
    lab = np.lib.format.open_memmap(OUT / "lab.npy", mode="w+", dtype=np.uint8,
                                    shape=(n, H, W))
    caps: dict[str, cv2.VideoCapture] = {}
    rows = []
    for j, (s, v, key) in enumerate(keep):
        slug = s["slug"]
        if slug not in caps:
            caps[slug] = cv2.VideoCapture(str(ROOT / f"data/interim/{slug}_norm.mp4"))
        caps[slug].set(cv2.CAP_PROP_POS_FRAMES, s["frame"])
        ok, fr = caps[slug].read()
        img[j] = cv2.resize(fr, SIZE, interpolation=cv2.INTER_AREA)
        z = np.load(OUT / f"cand_{key}.npz")
        lab[j] = z["teacher"] if v["chosen"] == "teacher" else z["student"]
        rows.append({"slug": slug, "frame": s["frame"], "source": v["chosen"],
                     "confidence": v["confidence"],
                     "identity_unreliable": slug in IDENTITY_UNRELIABLE})
    for c in caps.values():
        c.release()
    img.flush(); lab.flush()
    manifest = {
        "n": n, "size": [W, H], "source": "gold-eval-vlm-judged",
        "judge": "claude-opus-5 via LlmSupervisor.gold",
        "rows": rows,
        "frames": [r["frame"] for r in rows],
        "shots": [0] * n, "occlusion": [0.0] * n,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    by = {}
    for r in rows:
        by.setdefault((r["slug"], r["source"]), 0)
        by[(r["slug"], r["source"])] += 1
    print(f"ouro montado: {n} frames de {len(samples)} julgados")
    for (slug, src), c in sorted(by.items()):
        print(f"  {slug:22s} fonte {src:8s} {c}")


def phase_eval(ckpt: str) -> None:
    import torch
    from bjjvision.student import UNetStudent, metrics, normalise
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNetStudent(ck["width"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    man = json.loads((OUT / "manifest.json").read_text())
    img = np.load(OUT / "img.npy", mmap_mode="r")
    lab = np.load(OUT / "lab.npy", mmap_mode="r")
    rows = man["rows"]
    acc = {k: [] for k in ("assigned", "best", "iou_union")}
    with torch.no_grad():
        for s in range(0, len(rows), 16):
            x = normalise(np.asarray(img[s:s + 16]), dev)
            y = torch.from_numpy(np.asarray(lab[s:s + 16])).long().to(dev)
            m = metrics(model(x), y)
            for k in acc:
                acc[k].append(m[k])
    M = {k: np.concatenate(v) for k, v in acc.items()}
    print(f"=== GOLD eval: {ckpt} (git {ck.get('git_sha', '?')[:8]}) ===")
    for slug in sorted({r["slug"] for r in rows}):
        k = np.array([r["slug"] == slug for r in rows])
        unrel = rows[int(np.argmax(k))]["identity_unreliable"]
        head = "best" if unrel else "assigned"
        print(f"  {slug:22s} n={k.sum():3d}  {head} {M[head][k].mean():.4f}  "
              f"union {M['iou_union'][k].mean():.4f}"
              + ("  [identidade por-frame nao confiavel: headline=best]" if unrel else ""))
    for src in ("teacher", "student"):
        k = np.array([r["source"] == src for r in rows])
        if k.any():
            print(f"  fonte {src:8s}          n={k.sum():3d}  best {M['best'][k].mean():.4f}  "
                  f"union {M['iou_union'][k].mean():.4f}")
    print(f"  GERAL                  n={len(rows)}  best {M['best'].mean():.4f}  "
          f"union {M['iou_union'].mean():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["cards", "judge", "assemble", "eval"])
    ap.add_argument("--ckpt", default="data/out/student_ckpt_v4/student.pt")
    a = ap.parse_args()
    {"cards": phase_cards, "judge": phase_judge, "assemble": phase_assemble,
     "eval": lambda: phase_eval(a.ckpt)}[a.phase]()


if __name__ == "__main__":
    main()
