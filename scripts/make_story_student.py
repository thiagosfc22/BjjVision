"""Instagram Stories card for the distilled student, in the BjjVision look.

Panels are re-rendered from the 720p source rather than upscaled from the
320x180 training crops: a Story is viewed full-bleed on a phone, and a 3.4x
upscale of a debug figure reads as a screenshot, not as a result. The student's
own output stays at its true 320x180 and is nearest-upsampled, which is what it
would do in production anyway -- the slightly blockier outline on the left is
honest.

Palette and type are sampled from `story_capa_abertura.png` so this sits in the
same feed as the earlier covers.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.maskstore import MaskReader                       # noqa: E402
from bjjvision.student import UNetStudent, metrics, normalise    # noqa: E402
from bjjvision.studentdata import StudentSet                     # noqa: E402

W, H = 1080, 1920
BG = (9, 13, 19)
BLUE = (57, 131, 240)
WHITE = (239, 239, 240)
GREY = (99, 111, 126)
RULE = (28, 36, 48)
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
RED, GREEN = (0, 0, 255), (0, 255, 0)          # BGR, as in every other render


def f(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def tracked(d: ImageDraw.ImageDraw, xy, text, font, fill, extra=0.0):
    """Letter-spaced draw; PIL has no tracking and the brand look needs it."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + extra
    return x


def paint(img: np.ndarray, plane: np.ndarray) -> np.ndarray:
    out = img.copy()
    for cid, col in ((1, RED), (2, GREEN)):
        m = plane == cid
        if not m.any():
            continue
        out[m] = (0.42 * np.array(col) + 0.58 * out[m]).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, col, 3)
    return out


def crop_169(img: np.ndarray, union: np.ndarray, pad: float = 0.34) -> np.ndarray:
    """Crop 16:9 around the action so the athletes are actually big on a phone."""
    ys, xs = np.nonzero(union)
    if len(ys) == 0:
        return img
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    h = (y1 - y0) * (1 + 2 * pad)
    w = (x1 - x0) * (1 + 2 * pad)
    h = max(h, w * 9 / 16)
    w = h * 16 / 9
    H0, W0 = img.shape[:2]
    w, h = min(w, W0), min(h, H0)
    x0 = int(np.clip(cx - w / 2, 0, W0 - w)); y0 = int(np.clip(cy - h / 2, 0, H0 - h))
    return img[y0:y0 + int(h), x0:x0 + int(w)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/out/student_ckpt_v1/student.pt")
    ap.add_argument("--data", default="data/out/student_gx_320")
    ap.add_argument("--video", default="data/interim/galvao-xande_norm.mp4")
    ap.add_argument("--run", default="data/out/galvao-xande_scale")
    ap.add_argument("--out", default="data/out/story_student.png")
    a = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = UNetStudent(ck["width"]).to(dev); model.load_state_dict(ck["model"]); model.eval()

    ds = StudentSet(a.data)
    idx = np.where(np.isin(ds.shots, [6, 10, 12, 2]))[0]
    sc, pr = [], []
    with torch.no_grad():
        for s in range(0, len(idx), 32):
            sel = idx[s:s + 32]
            lg = model(normalise(np.asarray(ds.img[sel]), dev))
            y = torch.from_numpy(np.asarray(ds.lab[sel])).long().to(dev)
            sc.append(metrics(lg, y)["assigned"])
            pr.append(lg.argmax(1).cpu().numpy().astype(np.uint8))
    sc, pr = np.concatenate(sc), np.concatenate(pr)
    order = np.argsort(sc)
    picks = [("PIOR 10%", order[int(len(order) * .10)]),
             ("TÍPICO",   order[int(len(order) * .50)]),
             ("MELHOR 10%", order[int(len(order) * .90)])]

    readers = []
    for d in sorted(glob.glob(str(Path(a.run) / "chunk_*"))):
        R = MaskReader(Path(d) / "masks.bin")
        readers.append((set(R.frames), R))

    cap = cv2.VideoCapture(a.video)
    panels = []
    for tag, k in picks:
        fi = int(ds.frames[idx[k]])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, full = cap.read()
        teacher = np.zeros(full.shape[:2], np.uint8)
        for fs, R in readers:
            if fi in fs:
                m = R.get(fi)
                if m.get("B") is not None:
                    teacher[m["B"]] = 2
                if m.get("A") is not None:
                    teacher[m["A"]] = 1
                break
        stu = cv2.resize(pr[k], (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
        union = (stu > 0) | (teacher > 0)
        L = crop_169(paint(full, stu), union)
        Rr = crop_169(paint(full, teacher), union)
        panels.append((tag, float(sc[k]), L, Rr))
    cap.release()
    for _, R in readers:
        R.close()

    card = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(card)
    M = 64
    # Instagram covers roughly the top 220 px with the avatar/progress bar and
    # the bottom 200 px with the reply field. Nothing that must be read goes there.
    SAFE_TOP, SAFE_BOT = 232, 1712

    y = SAFE_TOP
    tracked(d, (M, y), "BJJVISION", f(BOLD, 27), BLUE, 2.2)
    y += 54
    for line in ("77 MINUTOS DE GPU", "ALUGADA VIRARAM", "1,3 MINUTO DE LAPTOP"):
        d.text((M, y), line, font=f(BOLD, 60), fill=WHITE)
        y += 71
    y += 16
    d.line([(M, y), (W - M, y)], fill=RULE, width=2)
    y += 26
    for line in ("Destilei o SAM2 num U-Net de 3,35M par\u00e2metros.",
                 "IoU 0.798 em 3.754 frames que ele nunca viu. Zero trocas de identidade."):
        d.text((M, y), line, font=f(REG, 28), fill=GREY)
        y += 37

    y += 24
    GAP = 10
    tracked(d, (M, y), "ALUNO", f(BOLD, 28), BLUE, 1.6)
    tracked(d, (M + (W - 2 * M + GAP) // 2, y), "PROFESSOR SAM2", f(BOLD, 28), (150, 160, 175), 1.6)
    y += 32
    d.text((M, y), "3,35M params  \u00b7  251 fps no laptop", font=f(REG, 23), fill=GREY)
    d.text((M + (W - 2 * M + GAP) // 2, y), "GPU alugada  \u00b7  4,34 fps", font=f(REG, 23), fill=GREY)
    y += 38

    # Size the strip from what is left inside the safe area rather than from a
    # constant, so the footer can never be pushed under Instagram's reply bar.
    FOOT = 190
    avail = (SAFE_BOT - FOOT) - y - 2 * GAP
    pw = min((W - 2 * M - GAP) // 2, int(avail / 3 * 16 / 9))
    ph = min(avail // 3, int(pw * 9 / 16))

    for tag, iou, L, Rr in panels:
        for j, im in enumerate((L, Rr)):
            t = cv2.resize(im, (pw, ph), interpolation=cv2.INTER_AREA)
            card.paste(Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)),
                       (M + j * (pw + GAP), y))
        # One chip, on the student's panel: the IoU is the student's score, so
        # hanging it off the teacher's frame read as if it graded the teacher.
        fb, fr = f(BOLD, 22), f(REG, 22)
        tw = d.textlength(tag, font=fb)
        vw = d.textlength("IoU %.2f" % iou, font=fb)
        box = int(14 + tw + 10 + 8 + 10 + vw + 14)
        d.rectangle([M, y, M + box, y + 34], fill=BG)
        d.rectangle([M, y, M + 4, y + 34], fill=BLUE)
        d.text((M + 14, y + 7), tag, font=fb, fill=WHITE)
        d.text((M + 14 + tw + 10, y + 7), "\u00b7", font=fr, fill=GREY)
        d.text((M + 14 + tw + 28, y + 7), "IoU %.2f" % iou, font=fb, fill=BLUE)
        y += ph + GAP

    fy = y + 16
    d.line([(M, fy), (W - M, fy)], fill=RULE, width=2)
    d.text((M, fy + 28), "estudo de caso por", font=f(REG, 27), fill=GREY)
    d.text((M, fy + 66), "THIAGO ABREU", font=f(BOLD, 52), fill=WHITE)
    d.rectangle([M, fy + 138, M + 340, fy + 144], fill=BLUE)
    print("footer bottom at y=%d (safe limit %d)" % (fy + 144, SAFE_BOT))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    card.save(a.out)
    print(a.out, card.size)


if __name__ == "__main__":
    main()
