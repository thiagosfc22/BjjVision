"""LinkedIn card for the distilled student: the input->output pair as the hero.

1200x1500 (4:5), the widest-reaching feed format. Unlike the Stories there is
no phone UI covering the edges, so content runs from y=56 down to y=1444.

Layout decisions, top to bottom:

* The hero is a real frame of the fight next to the same frame with the two
  predicted masks, an arrow in between. A recruiter who will never read "IoU"
  still gets it: one image in, two athletes out. Panels are re-rendered from
  the 720p source and the student's 320x180 output is nearest-upsampled onto
  it -- the slightly blocky mask edge is the model's true resolution, not a
  rendering bug, so it stays.
* The parameter count is the second-loudest element and is written out in
  full: "3.350.339" reads as a fact where "3,35M" reads as marketing. It is
  counted from the instantiated model at render time, and every other number
  that can be read from an artifact is -- checkpoint sizes via getsize, the
  SAM2 parameter count from its state dict, the held-out IoU and shots from
  history.json, frame counts from the video and manifest -- so the card
  cannot silently drift from the code. Numbers that exist only as one-off
  measurements (fps, training wall time, the occlusion sweep) live in
  MEASURED with a note saying where each came from.
* Then four more masked positions in a strip, one compact metric row, and a
  single technical flex: IoU *rising* with occlusion once athlete scale is
  held fixed, the most surprising measured fact about this model.

All copy sits in TEXTS, so `--lang en` renders the English version.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.student import UNetStudent, normalise   # noqa: E402
from bjjvision.studentdata import StudentSet           # noqa: E402

W, H = 1200, 1500
BG = (9, 13, 19)
PANEL = (15, 20, 31)
BLUE = (57, 131, 240)
BLUE_DIM = (127, 176, 247)
WHITE = (239, 239, 240)
GREY = (99, 111, 126)
FAINT = (62, 74, 92)
RULE = (28, 36, 48)
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
M = 64
RW = W - 2 * M
RED_BGR, GREEN_BGR = (0, 0, 255), (0, 255, 0)   # BGR, as in every other render
RED_RGB, GREEN_RGB = (255, 0, 0), (0, 255, 0)   # the same colours, for PIL

HERO = 13797                                    # white gi on top, IBJJF banners
THUMBS = [13347, 17867, 17951, 13690]           # turtle, over-under, two more

# Measured once, not recomputable at render time. The source of each number is
# named so it can be re-checked instead of trusted.
MEASURED = {
    "fps_student": 251,      # scripts/render_student.py, MPS on this laptop
    "fps_sam2": 4.34,        # the teacher run on the rented GPU (vast profile)
    "train_min": 17,         # wall time of scripts/train_student.py, MPS
    "occl_low": 0.773,       # held-out IoU at 0-40% occlusion, athlete scale
    "occl_high": 0.844,      # held fixed; 80-100% band. From the eval sweep.
}

TEXTS = {
    "pt": {
        "match": "GALVÃO × RIBEIRO · IBJJF PRO LEAGUE GRAND PRIX 2017",
        "headline": "A REDE NEURAL QUE SEPARA OS DOIS ATLETAS",
        "subtitle": "Segmentação em vídeo para **jiu-jitsu brasileiro**, destilada do SAM2.",
        "in_label": "entrada · 1 frame",
        "out_label": "saída · 2 máscaras",
        "frame_chip": "frame {frame} de {total}",
        "legend_a": "atleta A",
        "legend_b": "atleta B",
        "param_caption": "PARÂMETROS TREINADOS DO ZERO",
        "note1": "sem backbone pré-treinado",
        "note2": "{steps} passos · {mins} minutos num MacBook",
        "strip_label": "O MESMO MODELO EM OUTRAS QUATRO POSIÇÕES",
        "m1_label": ("IoU em {n} frames", "de shots fora do treino"),
        "m2_value": "{fps} fps",
        "m2_label": ("no laptop", "SAM2: {fps_sam} em GPU"),
        "m3_value": "{mb} MB",
        "m3_label": ("em disco", "SAM2: {sam_mb} MB"),
        "m4_value": "{ratio}×",
        "m4_label": ("menos parâmetros", "que o professor SAM2"),
        "takeaway": "com o tamanho do atleta fixo, o IoU sobe com a oclusão: "
                    "{lo} (0–40% escondido) → {hi} (80–100%)",
        "credit": "estudo de caso por",
        "name": "THIAGO ABREU",
    },
    "en": {
        "match": "GALVÃO × RIBEIRO · IBJJF PRO LEAGUE GRAND PRIX 2017",
        "headline": "THE NEURAL NET THAT SEPARATES BOTH ATHLETES",
        "subtitle": "Video segmentation for **Brazilian jiu-jitsu**, distilled from SAM2.",
        "in_label": "input · 1 frame",
        "out_label": "output · 2 masks",
        "frame_chip": "frame {frame} of {total}",
        "legend_a": "athlete A",
        "legend_b": "athlete B",
        "param_caption": "PARAMETERS TRAINED FROM SCRATCH",
        "note1": "no pretrained backbone",
        "note2": "{steps} steps · {mins} minutes on a MacBook",
        "strip_label": "THE SAME MODEL ON FOUR MORE POSITIONS",
        "m1_label": ("IoU on {n} frames", "from held-out shots"),
        "m2_value": "{fps} fps",
        "m2_label": ("on a laptop", "SAM2: {fps_sam} on GPU"),
        "m3_value": "{mb} MB",
        "m3_label": ("on disk", "SAM2: {sam_mb} MB"),
        "m4_value": "{ratio}×",
        "m4_label": ("fewer parameters", "than the SAM2 teacher"),
        "takeaway": "with athlete scale held fixed, IoU rises with occlusion: "
                    "{lo} (0–40% hidden) → {hi} (80–100%)",
        "credit": "case study by",
        "name": "THIAGO ABREU",
    },
}


def f(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def tracked(d, xy, text, font, fill, extra=0.0):
    """Letter-spaced draw; PIL has no tracking and the brand look needs it."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + extra
    return x


def rich(d, xy, text, size, base_fill, hot_fill=BLUE):
    """Draw `text`, rendering **spans** in bold + the accent colour."""
    x, y = xy
    for i, seg in enumerate(text.split("**")):
        hot = i % 2 == 1
        fo = f(BOLD if hot else REG, size)
        d.text((x, y), seg, font=fo, fill=hot_fill if hot else base_fill)
        x += d.textlength(seg, font=fo)
    return x


def fit(d, text, size, maxw, path=BOLD):
    """Largest font <= size that keeps `text` inside `maxw`."""
    while size > 12 and d.textlength(text, font=f(path, size)) > maxw:
        size -= 2
    return f(path, size)


def group(n: int, lang: str) -> str:
    s = f"{n:,}"
    return s.replace(",", ".") if lang == "pt" else s


def dec(x: float, nd: int, lang: str) -> str:
    s = f"{x:.{nd}f}"
    return s.replace(".", ",") if lang == "pt" else s


def paint(img: np.ndarray, plane: np.ndarray) -> np.ndarray:
    out = img.copy()
    for cid, col in ((1, RED_BGR), (2, GREEN_BGR)):
        m = plane == cid
        if not m.any():
            continue
        out[m] = (0.45 * np.array(col) + 0.55 * out[m]).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, col, 3)
    return out


def crop_box(union: np.ndarray, pad: float = 0.34) -> tuple[int, int, int, int]:
    """16:9 box around the mask union so the athletes are actually big."""
    ys, xs = np.nonzero(union)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    h = (y1 - y0) * (1 + 2 * pad)
    w = (x1 - x0) * (1 + 2 * pad)
    h = max(h, w * 9 / 16)
    w = h * 16 / 9
    H0, W0 = union.shape
    w, h = min(w, W0), min(h, H0)
    x0 = int(np.clip(cx - w / 2, 0, W0 - w))
    y0 = int(np.clip(cy - h / 2, 0, H0 - h))
    return x0, y0, int(w), int(h)


def chip(d, x, y, parts):
    """Small overlay label on a panel corner. `parts` mixes text and swatches."""
    fb = f(BOLD, 20)
    wtot = 14
    for kind, val in parts:
        wtot += d.textlength(val, font=fb) + 8 if kind == "t" else 14 + 8
    wtot += 6
    d.rectangle([x, y, x + wtot, y + 32], fill=BG)
    cx = x + 14
    for kind, val in parts:
        if kind == "t":
            d.text((cx, y + 6), val, font=fb, fill=WHITE)
            cx += d.textlength(val, font=fb) + 8
        else:
            d.rectangle([cx, y + 9, cx + 14, y + 23], fill=val)
            cx += 14 + 8


def harrow(d, x0, x1, y, col=BLUE):
    d.line([(x0, y), (x1 - 10, y)], fill=col, width=3)
    d.polygon([(x1, y), (x1 - 14, y - 9), (x1 - 14, y + 9)], fill=col)


def gather(a) -> dict:
    """Everything readable from artifacts, read from them."""
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = UNetStudent(ck["width"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    stu_params = sum(p.numel() for p in model.parameters())

    ds = StudentSet(a.data)
    pos = {int(fr): i for i, fr in enumerate(ds.frames)}
    frames = [HERO] + THUMBS
    batch = np.stack([np.asarray(ds.img[pos[n]]) for n in frames])
    with torch.no_grad():
        planes = model(normalise(batch, dev)).argmax(1).cpu().numpy().astype(np.uint8)

    cap = cv2.VideoCapture(a.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    panels = {}
    for n, plane in zip(frames, planes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, full = cap.read()
        assert ok, f"could not read frame {n}"
        up = cv2.resize(plane, (full.shape[1], full.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        x0, y0, w, h = crop_box(up > 0)
        raw = full[y0:y0 + h, x0:x0 + w]
        masked = paint(full, up)[y0:y0 + h, x0:x0 + w]
        panels[n] = (raw, masked)
    cap.release()

    hist = json.loads((Path(a.ckpt).parent / "history.json").read_text())
    n_val = int(np.isin(ds.shots, hist["test_shots"]).sum())

    sam = torch.load(a.sam, map_location="cpu", weights_only=False)
    sam = sam.get("model", sam)
    sam_params = sum(v.numel() for v in sam.values() if hasattr(v, "numel"))

    return {
        "panels": panels,
        "total_frames": total_frames,
        "stu_params": stu_params,
        "stu_mb": os.path.getsize(a.ckpt) / 1e6,
        "sam_params": sam_params,
        "sam_mb": os.path.getsize(a.sam) / 1e6,
        "iou": hist["final"]["assigned"],
        "n_val": n_val,
        "steps": ck["args"]["steps"],
    }


def to_panel(img_bgr: np.ndarray, w: int, h: int) -> Image.Image:
    t = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))


def compose(T: dict, R: dict, lang: str, out: str) -> None:
    card = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(card)

    # -- header ---------------------------------------------------------------
    tracked(d, (M, 56), "BJJVISION", f(BOLD, 28), BLUE, 2.2)
    fm = f(REG, 20)
    d.text((W - M - d.textlength(T["match"], font=fm), 62), T["match"],
           font=fm, fill=FAINT)
    d.text((M, 116), T["headline"], font=fit(d, T["headline"], 46, RW), fill=WHITE)
    rich(d, (M, 180), T["subtitle"], 27, GREY)

    # -- hero: raw frame -> masked frame -------------------------------------
    pw = 504
    ph = round(pw * 9 / 16)
    x_r = W - M - pw
    fl = f(BOLD, 26)
    d.text((M, 244), T["in_label"], font=fl, fill=GREY)
    d.text((x_r, 244), T["out_label"], font=fl, fill=GREY)
    y = 285
    raw, masked = R["panels"][HERO]
    card.paste(to_panel(raw, pw, ph), (M, y))
    card.paste(to_panel(masked, pw, ph), (x_r, y))
    harrow(d, M + pw + 8, x_r - 8, y + ph // 2)
    chip(d, M, y, [("t", T["frame_chip"].format(
        frame=group(HERO, lang), total=group(R["total_frames"], lang)))])
    chip(d, x_r, y, [("s", RED_RGB), ("t", T["legend_a"]),
                     ("s", GREEN_RGB), ("t", T["legend_b"])])
    y += ph

    # -- the number -----------------------------------------------------------
    y += 24
    d.line([(M, y), (W - M, y)], fill=RULE, width=2)
    y += 28
    d.text((M, y), group(R["stu_params"], lang), font=f(BOLD, 128), fill=WHITE)
    fn = f(REG, 26)
    n1, n2 = T["note1"], T["note2"].format(
        steps=group(R["steps"], lang), mins=MEASURED["train_min"])
    d.text((W - M - d.textlength(n1, font=fn), y + 66), n1, font=fn, fill=GREY)
    d.text((W - M - d.textlength(n2, font=fn), y + 104), n2, font=fn, fill=GREY)
    y += 156
    tracked(d, (M, y), T["param_caption"], f(BOLD, 30), BLUE, 1.8)
    y += 34

    # -- thumbnail strip ------------------------------------------------------
    y += 26
    d.line([(M, y), (W - M, y)], fill=RULE, width=2)
    y += 26
    tracked(d, (M, y), T["strip_label"], f(BOLD, 24), BLUE, 1.6)
    y += 40
    tw = (RW - 3 * 16) // 4
    th = round(tw * 9 / 16)
    for i, n in enumerate(THUMBS):
        card.paste(to_panel(R["panels"][n][1], tw, th), (M + i * (tw + 16), y))
    y += th

    # -- metric row -----------------------------------------------------------
    y += 24
    d.line([(M, y), (W - M, y)], fill=RULE, width=2)
    y += 26
    ratio = round(R["sam_params"] / R["stu_params"])
    cols = [
        (dec(R["iou"], 3, lang),
         T["m1_label"][0].format(n=group(R["n_val"], lang)), T["m1_label"][1]),
        (T["m2_value"].format(fps=MEASURED["fps_student"]),
         T["m2_label"][0],
         T["m2_label"][1].format(fps_sam=dec(MEASURED["fps_sam2"], 2, lang))),
        (T["m3_value"].format(mb=dec(R["stu_mb"], 1, lang)),
         T["m3_label"][0],
         T["m3_label"][1].format(sam_mb=round(R["sam_mb"]))),
        (T["m4_value"].format(ratio=ratio), T["m4_label"][0], T["m4_label"][1]),
    ]
    cw = RW // 4
    for i, (v, l1, l2) in enumerate(cols):
        x = M + i * cw
        if i:
            d.line([(x - 16, y + 4), (x - 16, y + 106)], fill=RULE, width=2)
        d.text((x, y), v, font=f(BOLD, 44), fill=WHITE)
        d.text((x, y + 58), l1, font=f(REG, 21), fill=GREY)
        d.text((x, y + 85), l2, font=f(REG, 21), fill=FAINT)
    y += 110

    # -- one technical flex ---------------------------------------------------
    y += 26
    tk = T["takeaway"].format(lo=dec(MEASURED["occl_low"], 3, lang),
                              hi=dec(MEASURED["occl_high"], 3, lang))
    d.text((M, y), tk, font=fit(d, tk, 25, RW), fill=BLUE)
    y += 34

    # -- footer ---------------------------------------------------------------
    fy = 1300
    d.line([(M, fy), (W - M, fy)], fill=RULE, width=2)
    d.text((M, fy + 28), T["credit"], font=f(REG, 27), fill=GREY)
    d.text((M, fy + 66), T["name"], font=f(BOLD, 52), fill=WHITE)
    d.rectangle([M, fy + 138, M + 340, fy + 144], fill=BLUE)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    card.save(out)
    print("%s %s | conteudo termina em y=%d, rodape em %d..%d" %
          (out, card.size, y, fy, fy + 144))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/out/student_ckpt_v1/student.pt")
    ap.add_argument("--data", default="data/out/student_gx_320")
    ap.add_argument("--video", default="data/interim/galvao-xande_norm.mp4")
    ap.add_argument("--sam", default="checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--lang", choices=("pt", "en"), default="pt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = ("data/out/linkedin_student.png" if a.lang == "pt"
                 else "data/out/linkedin_student_en.png")

    R = gather(a)
    print("params %s | iou %.4f em %d frames | steps %s | aluno %.1f MB | "
          "sam %s params, %.0f MB" %
          (group(R["stu_params"], a.lang), R["iou"], R["n_val"], R["steps"],
           R["stu_mb"], group(R["sam_params"], a.lang), R["sam_mb"]))
    compose(TEXTS[a.lang], R, a.lang, a.out)


if __name__ == "__main__":
    main()
