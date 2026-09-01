"""Is 'the camera is too wide' the failure, or is the wide footage just hard?

    python scripts/scale_scope.py ruler    # mask-free scale of every judged frame
    python scripts/scale_scope.py cross    # scale x verdict
    python scripts/scale_scope.py zoom     # render the crop-zoom stack, to LOOK at
    python scripts/scale_scope.py quant    # crop-zoom measured, no labels needed

Why not the obvious proxy: the handoff proposed bbox-of-the-union / frame
height, straight off the cand_*.npz planes. That proxy is circular -- a mask
that lands on a banner is huge, so it measures the failure, not the camera.
Measured: judge-'wide' median 0.547 vs judge-'medium' 0.558, no separation at
all, p=0.71 against the verdicts.

So the ruler here is mask-free: YOLO person boxes, median height of the two
tallest / frame height. It answers 'how big is a person in this frame' without
consulting either candidate. It agrees with the judge's eye (wide 0.338 vs
medium 0.466, p=1.3e-13), which is what a ruler has to do before it is used.

Correlation is not the claim, though. `zoom` and `quant` are the causal test:
same model, same frame, ONLY the scale changes -- crop 2.5x around the pair,
re-run v4, compare. The crop is centred on the pair of person boxes with the
largest mutual overlap, because grapplers are the entangled pair and
bystanders are not.
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
from bjjvision.triage import paint                    # noqa: E402

OUT = ROOT / "data" / "out" / "gold_eval"
SIZE = (320, 180)
ZOOM = 2.5
WIDE = 0.40          # person height below which the shot counts as wide


def judged() -> list[tuple[str, dict]]:
    verdicts = json.loads((OUT / "verdicts.json").read_text())
    samples = {f"{s['slug']}_{s['frame']:06d}": s
               for s in json.loads((OUT / "samples.json").read_text())}
    # a couple of verdicts predate the current card generation; they have no frame
    return [(k, samples[k]) for k in verdicts if k in samples]


def frames(keys):
    caps: dict[str, cv2.VideoCapture] = {}
    for key, s in keys:
        slug = s["slug"]
        if slug not in caps:
            caps[slug] = cv2.VideoCapture(str(ROOT / f"data/interim/{slug}_norm.mp4"))
        caps[slug].set(cv2.CAP_PROP_POS_FRAMES, s["frame"])
        ok, fr = caps[slug].read()
        if ok:
            yield key, s, fr
    for c in caps.values():
        c.release()


def best_pair(boxes) -> tuple[np.ndarray, float]:
    """The two person boxes with the largest mutual IoU -- the entangled pair."""
    b = np.asarray(boxes, float)
    best, bi = 0.0, (0, min(1, len(b) - 1))
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            xa = max(0.0, min(b[i][2], b[j][2]) - max(b[i][0], b[j][0]))
            ya = max(0.0, min(b[i][3], b[j][3]) - max(b[i][1], b[j][1]))
            it = xa * ya
            un = ((b[i][2] - b[i][0]) * (b[i][3] - b[i][1])
                  + (b[j][2] - b[j][0]) * (b[j][3] - b[j][1]) - it)
            v = it / un if un > 0 else 0.0
            if v > best:
                best, bi = v, (i, j)
    return b[list(bi)], best


def crop_box(pair, W, H) -> tuple[int, int, int, int]:
    cx = (pair[:, 0].min() + pair[:, 2].max()) / 2
    cy = (pair[:, 1].min() + pair[:, 3].max()) / 2
    hw, hh = W / (2 * ZOOM), H / (2 * ZOOM)
    cx = float(np.clip(cx, hw, W - hw))
    cy = float(np.clip(cy, hh, H - hh))
    return int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)


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


def predictor():
    import torch
    from bjjvision.student import normalise
    model, dev = load_v4()

    def infer(bgr):
        small = cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            p = model(normalise(small[None], dev)).argmax(1)[0]
        return small, p.cpu().numpy().astype(np.uint8)
    return infer


# ---------------------------------------------------------------- ruler

def phase_ruler() -> None:
    from ultralytics import YOLO
    yolo = YOLO(str(ROOT / "yolo11x-pose.pt"))
    scale, boxes = {}, {}
    keys = judged()
    for n, (key, s, fr) in enumerate(frames(keys)):
        H, W = fr.shape[:2]
        b = yolo.predict(fr, verbose=False, conf=0.35, device="mps")[0].boxes.xyxy.cpu().numpy()
        hs = sorted(((y1 - y0) / H for _, y0, _, y1 in b), reverse=True)
        scale[key] = {"n_people": len(hs),
                      "top2": float(np.median(hs[:2])) if hs else 0.0}
        boxes[key] = {"wh": [W, H], "boxes": b.tolist()}
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(keys)}", flush=True)
    pair = {}
    for key, d in boxes.items():
        if len(d["boxes"]) >= 2:
            _, iou = best_pair(d["boxes"])
            pair[key] = float(iou)
    (OUT / "scale_yolo.json").write_text(json.dumps(scale, indent=1))
    (OUT / "boxes_yolo.json").write_text(json.dumps(boxes))
    (OUT / "pairiou.json").write_text(json.dumps(pair, indent=1))
    print(f"regua pronta: {len(scale)} quadros")


# ---------------------------------------------------------------- cross

def phase_cross() -> None:
    from scipy import stats
    V = json.loads((OUT / "verdicts.json").read_text())
    SC = json.loads((OUT / "scale_yolo.json").read_text())
    rows = [dict(key=k, scale=s["top2"], judge=V[k].get("scale"),
                 t=V[k]["teacher"]["both_athletes_covered"],
                 s=V[k]["student"]["both_athletes_covered"],
                 any=V[k]["teacher"]["both_athletes_covered"]
                 or V[k]["student"]["both_athletes_covered"])
            for k, s in SC.items()]

    print("1) a regua objetiva confere com o olho do juiz?")
    for j in ("wide", "medium"):
        g = [r["scale"] for r in rows if r["judge"] == j]
        print(f"   juiz '{j}': n={len(g):3d}  altura de pessoa mediana {np.median(g):.3f}")
    w = [r["scale"] for r in rows if r["judge"] == "wide"]
    m = [r["scale"] for r in rows if r["judge"] == "medium"]
    print(f"   Mann-Whitney (medium > wide): p={stats.mannwhitneyu(m, w, alternative='greater').pvalue:.1e}")

    print("\n2) achar os dois corpos x escala")
    print(f"   {'altura de pessoa':18s} {'n':>4s} {'prof':>6s} {'stud':>6s} {'algum':>6s}")
    for a, b in [(0, .30), (.30, .40), (.40, .60), (.60, 2)]:
        g = [r for r in rows if a <= r["scale"] < b]
        if g:
            print(f"   [{a:.2f},{b:.2f})       {len(g):4d} {sum(r['t'] for r in g)/len(g):5.0%} "
                  f"{sum(r['s'] for r in g)/len(g):5.0%} {sum(r['any'] for r in g)/len(g):5.0%}")
    f = [r["scale"] for r in rows if r["any"]]
    l = [r["scale"] for r in rows if not r["any"]]
    print(f"\n   achou (n={len(f)}) mediana {np.median(f):.3f} | perdeu (n={len(l)}) "
          f"mediana {np.median(l):.3f} | p={stats.mannwhitneyu(f, l, alternative='greater').pvalue:.1e}")


# ---------------------------------------------------------------- zoom

def phase_zoom(n_rows: int = 8) -> None:
    infer = predictor()
    B = json.loads((OUT / "boxes_yolo.json").read_text())
    SC = json.loads((OUT / "scale_yolo.json").read_text())
    IOU = json.loads((OUT / "pairiou.json").read_text())
    samples = {k: s for k, s in judged()}
    # wide frames that actually contain an entangled pair: the fair test
    pick = sorted([k for k in IOU if SC[k]["top2"] < WIDE and IOU[k] >= 0.40],
                  key=lambda k: -IOU[k])[:n_rows]
    stack = []
    for key, s, fr in frames([(k, samples[k]) for k in pick]):
        H, W = fr.shape[:2]
        pair, iou = best_pair(B[key]["boxes"])
        X0, Y0, X1, Y1 = crop_box(pair, W, H)
        _, full = infer(fr)
        small_c, cropped = infer(fr[Y0:Y1, X0:X1])
        fx0, fy0 = int(X0 * SIZE[0] / W), int(Y0 * SIZE[1] / H)
        fx1, fy1 = int(X1 * SIZE[0] / W), int(Y1 * SIZE[1] / H)
        fs = cv2.resize(fr, SIZE, interpolation=cv2.INTER_AREA)
        up = lambda im: cv2.resize(im, (426, 240), interpolation=cv2.INTER_NEAREST)  # noqa: E731
        panels = [up(cv2.resize(fr[Y0:Y1, X0:X1], SIZE, interpolation=cv2.INTER_AREA)),
                  up(paint(fs[fy0:fy1, fx0:fx1], full[fy0:fy1, fx0:fx1])),
                  up(paint(small_c, cropped))]
        for im, t in zip(panels, ("RECORTE no par", f"v4 HOJE (h={SC[key]['top2']:.2f})",
                                  f"v4 NO ZOOM {ZOOM}x (h~{SC[key]['top2']*ZOOM:.2f})")):
            cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        row = np.hstack(panels)
        cv2.putText(row, f"{key} iou_par={iou:.2f}", (6, 232),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        stack.append(row)
    p = OUT / "cropzoom_par.jpg"
    cv2.imwrite(str(p), np.vstack(stack), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"-> {p}  ({len(stack)} linhas)  OLHE antes de acreditar nos numeros")


# ---------------------------------------------------------------- quant

def rasterise(boxes, sx, sy, off=(0, 0)) -> np.ndarray:
    m = np.zeros((SIZE[1], SIZE[0]), bool)
    for x0, y0, x1, y1 in boxes:
        a = max(0, int(round((x0 - off[0]) * sx)))
        c = min(SIZE[0], int(round((x1 - off[0]) * sx)))
        b = max(0, int(round((y0 - off[1]) * sy)))
        d = min(SIZE[1], int(round((y1 - off[1]) * sy)))
        if c > a and d > b:
            m[b:d, a:c] = True
    return m


def _score(pred, pairm, peoplem):
    fg = pred > 0
    rec = (fg & pairm).sum() / max(pairm.sum(), 1)
    pur = (fg & peoplem).sum() / fg.sum() if fg.sum() else np.nan
    return float(rec), float(pur)


def phase_quant() -> None:
    infer = predictor()
    B = json.loads((OUT / "boxes_yolo.json").read_text())
    SC = json.loads((OUT / "scale_yolo.json").read_text())
    IOU = json.loads((OUT / "pairiou.json").read_text())
    samples = {k: s for k, s in judged()}
    rows = []
    for key, s, fr in frames([(k, v) for k, v in samples.items() if k in IOU]):
        H, W = fr.shape[:2]
        allb = np.asarray(B[key]["boxes"], float)
        pair, _ = best_pair(allb)
        X0, Y0, X1, Y1 = crop_box(pair, W, H)
        inside = bool(((pair[:, 0] >= X0) & (pair[:, 2] <= X1)
                       & (pair[:, 1] >= Y0) & (pair[:, 3] <= Y1)).all())
        _, full = infer(fr)
        _, cropped = infer(fr[Y0:Y1, X0:X1])
        rf = _score(full, rasterise(pair, SIZE[0] / W, SIZE[1] / H),
                    rasterise(allb, SIZE[0] / W, SIZE[1] / H))
        sx, sy = SIZE[0] / (X1 - X0), SIZE[1] / (Y1 - Y0)
        rc = _score(cropped, rasterise(pair, sx, sy, (X0, Y0)),
                    rasterise(allb, sx, sy, (X0, Y0)))
        rows.append(dict(key=key, h=SC[key]["top2"], iou=IOU[key], pair_in_crop=inside,
                         rec_hoje=rf[0], rec_zoom=rc[0], pur_hoje=rf[1], pur_zoom=rc[1]))
    (OUT / "zoom_quant.json").write_text(json.dumps(rows, indent=1))

    def show(sel, name):
        g = [r for r in rows if sel(r)]
        if not g:
            return
        f = lambda k: np.nanmean([r[k] for r in g])  # noqa: E731
        print(f"  {name:34s} n={len(g):3d}  recall_par {f('rec_hoje'):.3f} -> {f('rec_zoom'):.3f}"
              f"   pureza {f('pur_hoje'):.3f} -> {f('pur_zoom'):.3f}")

    print(f"\n=== v4 hoje -> v4 no recorte {ZOOM}x  ({len(rows)} quadros) ===")
    print("  recall_par: fracao da caixa do par coberta pela mascara (a caixa tem fundo")
    print("  dentro, entao o valor absoluto e baixo de proposito -- o que vale e o delta)")
    show(lambda r: r["h"] < WIDE and r["pair_in_crop"], "WIDE, par inteiro no recorte")
    show(lambda r: r["h"] < WIDE and not r["pair_in_crop"], "WIDE, par cortado (ROI errada)")
    show(lambda r: r["h"] >= WIDE and r["pair_in_crop"], "MEDIO, par inteiro no recorte")
    show(lambda r: r["h"] < WIDE and r["pair_in_crop"] and r["iou"] >= 0.25, "WIDE + par entrelacado")
    g = [r for r in rows if r["h"] < WIDE and r["pair_in_crop"]]
    print(f"\n  wide com ROI certa (n={len(g)}): zoom melhora recall em "
          f"{sum(r['rec_zoom'] > r['rec_hoje'] + 0.05 for r in g)}, "
          f"piora em {sum(r['rec_zoom'] < r['rec_hoje'] - 0.05 for r in g)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ruler", "cross", "zoom", "quant"])
    a = ap.parse_args()
    {"ruler": phase_ruler, "cross": phase_cross,
     "zoom": phase_zoom, "quant": phase_quant}[a.phase]()


if __name__ == "__main__":
    main()
