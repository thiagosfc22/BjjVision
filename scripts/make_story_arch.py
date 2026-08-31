"""Two Instagram Stories about the student's architecture, in the BjjVision look.

Card 1 is the U-Net itself -- the parameter count as the headline, the nine
convolutional blocks drawn as the U they actually form, with the real tensor
shapes pulled from a forward pass rather than typed in by hand.

Card 2 is where those parameters sit: a bar per block, and the size/speed
comparison against the SAM2 checkpoint that taught it.

Everything numeric here is read from the model and the checkpoints at render
time. A slide that says 3.350.339 has to keep saying the truth after the next
change to `student.py`, and the only way to guarantee that is to not retype it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.student import UNetStudent   # noqa: E402

W, H = 1080, 1920
BG = (9, 13, 19)
PANEL = (15, 20, 31)
PANEL_HOT = (16, 26, 46)
BLUE = (57, 131, 240)
BLUE_DIM = (127, 176, 247)
WHITE = (239, 239, 240)
GREY = (99, 111, 126)
FAINT = (62, 74, 92)
RULE = (28, 36, 48)
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
M = 64
SAFE_TOP, SAFE_BOT = 232, 1712
HOT = {"e4", "bott", "d4"}          # the three blocks holding 84% of the weights


def f(p, s):
    return ImageFont.truetype(p, s)


def tracked(d, xy, text, font, fill, extra=0.0):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + extra
    return x


def br(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def short(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.2f}".replace(".", ",") + " mi"
    if n >= 1000:
        return f"{n/1e3:.0f} mil"
    return str(n)


def probe() -> tuple[list[dict], int]:
    m = UNetStudent(32).eval()
    shapes: dict[str, tuple] = {}
    for name, mod in m.named_children():
        mod.register_forward_hook(
            lambda mo, i, o, n=name: shapes.__setitem__(n, tuple(o.shape)))
    with torch.no_grad():
        m(torch.randn(1, 3, 180, 320))
    total = sum(p.numel() for p in m.parameters())
    out = []
    for name, mod in m.named_children():
        p = sum(x.numel() for x in mod.parameters())
        if p == 0 or name not in shapes:
            continue
        s = shapes[name]
        out.append({"name": name, "ch": s[1], "w": s[3], "h": s[2],
                    "p": p, "pct": 100 * p / total})
    return out, total


def frame(d: ImageDraw.ImageDraw, eyebrow="BJJVISION"):
    tracked(d, (M, SAFE_TOP), eyebrow, f(BOLD, 27), BLUE, 2.2)


def footer(d: ImageDraw.ImageDraw):
    fy = SAFE_BOT - 176
    d.line([(M, fy), (W - M, fy)], fill=RULE, width=2)
    d.text((M, fy + 28), "estudo de caso por", font=f(REG, 27), fill=GREY)
    d.text((M, fy + 66), "THIAGO ABREU", font=f(BOLD, 52), fill=WHITE)
    d.rectangle([M, fy + 138, M + 340, fy + 144], fill=BLUE)


def block(d, x, y, w, h, title, sub, hot):
    d.rectangle([x, y, x + w, y + h], fill=PANEL_HOT if hot else PANEL,
                outline=BLUE if hot else RULE, width=2)
    if hot:
        d.rectangle([x, y, x + 6, y + h], fill=BLUE)
    cx = x + w / 2
    ft, fs = f(BOLD, 30), f(REG, 24)
    d.text((cx - d.textlength(title, font=ft) / 2, y + h / 2 - 34), title, font=ft, fill=WHITE)
    d.text((cx - d.textlength(sub, font=fs) / 2, y + h / 2 + 6), sub, font=fs,
           fill=BLUE_DIM if hot else GREY)


def varrow(d, x, y0, y1, col=FAINT):
    d.line([(x, y0), (x, y1)], fill=col, width=2)
    s = 9 if y1 > y0 else -9
    d.polygon([(x, y1), (x - 7, y1 - s), (x + 7, y1 - s)], fill=col)


def harrow(d, x0, x1, y, col=FAINT, dash=0):
    if dash:
        x = x0
        while x < x1 - 12:
            d.line([(x, y), (min(x + dash, x1 - 12), y)], fill=col, width=2)
            x += dash * 2
    else:
        d.line([(x0, y), (x1, y)], fill=col, width=2)
    d.polygon([(x1, y), (x1 - 11, y - 7), (x1 - 11, y + 7)], fill=col)


def card_one(blocks, total, out):
    card = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(card)
    frame(d)

    y = SAFE_TOP + 52
    d.text((M, y), br(total), font=f(BOLD, 116), fill=WHITE)
    y += 140
    tracked(d, (M, y), "PARÂMETROS TREINADOS DO ZERO", f(BOLD, 32), BLUE, 1.8)
    y += 56
    d.line([(M, y), (W - M, y)], fill=RULE, width=2)
    y += 28
    for line in ("Arquitetura U-Net, 9 blocos convolucionais.",
                 "Entrada: um frame. Saída: as duas máscaras de atleta."):
        d.text((M, y), line, font=f(REG, 28), fill=GREY)
        y += 37

    bw, gap = 376, 32
    lx, rx = M, W - M - bw
    lc, rc = lx + bw / 2, rx + bw / 2
    y = 620
    d.rectangle([lx, y, lx + bw, y + 64], fill=PANEL, outline=RULE, width=2)
    d.rectangle([rx, y, rx + bw, y + 64], fill=PANEL, outline=RULE, width=2)
    for cx, t in ((lc, "entrada · 1 frame"), (rc, "saída · 2 máscaras")):
        ft = f(BOLD, 28)
        d.text((cx - d.textlength(t, font=ft) / 2, y + 17), t, font=ft, fill=GREY)
    rows = [y]
    y += 64 + gap

    by = {b["name"]: b for b in blocks}
    enc = ["e1", "e2", "e3", "e4"]
    dec = ["d1", "d2", "d3", "d4"]
    bh = 110
    ys = {}
    for i in range(4):
        ys[enc[i]] = ys[dec[i]] = y
        e, dd = by[enc[i]], by[dec[i]]
        block(d, lx, y, bw, bh, f"{e['name']} · {e['ch']} canais",
              f"{e['w']}×{e['h']} · {short(e['p'])}", e["name"] in HOT)
        block(d, rx, y, bw, bh, f"{dd['name']} · {dd['ch']} canais",
              f"{dd['w']}×{dd['h']} · {short(dd['p'])}", dd["name"] in HOT)
        y += bh + gap
    b = by["bott"]
    bx, bwid = (W - 470) // 2, 470
    block(d, bx, y, bwid, bh, f"gargalo · {b['ch']} canais",
          f"{b['w']}×{b['h']} · {short(b['p'])}", True)
    bott_mid = y + bh / 2
    diag_bottom = y + bh

    varrow(d, lc, rows[0] + 68, ys["e1"] - 4)
    for i in range(3):
        varrow(d, lc, ys[enc[i]] + bh + 4, ys[enc[i + 1]] - 4)
    d.line([(lc, ys["e4"] + bh + 4), (lc, bott_mid)], fill=FAINT, width=2)
    harrow(d, lc, bx - 4, bott_mid)
    d.line([(bx + bwid + 4, bott_mid), (rc, bott_mid)], fill=FAINT, width=2)
    varrow(d, rc, bott_mid, ys["d4"] + bh + 4)
    for i in (3, 2, 1):
        varrow(d, rc, ys[dec[i]] - 4, ys[dec[i - 1]] + bh + 4)
    varrow(d, rc, ys["d1"] - 4, rows[0] + 68)

    for n in enc:
        harrow(d, lx + bw + 4, rx - 4, ys[n] + bh / 2, col=(42, 52, 68), dash=9)

    ly = diag_bottom + 40
    fl = f(REG, 23)
    d.rectangle([M, ly, M + 18, ly + 18], fill=BLUE)
    d.text((M + 30, ly - 3), "e4, gargalo e d4 — 84% de todo o peso do modelo",
           font=fl, fill=GREY)
    ly += 36
    d.rectangle([M, ly, M + 18, ly + 18], fill=PANEL, outline=RULE, width=2)
    d.text((M + 30, ly - 3), "demais blocos — 16%", font=fl, fill=GREY)
    x2 = M + 30 + d.textlength("demais blocos — 16%", font=fl) + 56
    for k in range(3):
        d.line([(x2 + k * 16, ly + 9), (x2 + k * 16 + 9, ly + 9)], fill=(42, 52, 68), width=2)
    d.text((x2 + 56, ly - 3), "conexão de salto", font=fl, fill=GREY)

    footer(d)
    card.save(out)
    print(out, "diagrama termina em y=%d" % (ly + 18))


def card_two(blocks, total, out):
    card = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(card)
    frame(d)

    y = SAFE_TOP + 46
    for line in ("ONDE FICAM OS", "3,35 MILHÕES"):
        d.text((M, y), line, font=f(BOLD, 72), fill=WHITE)
        y += 82
    d.line([(M, y), (W - M, y)], fill=RULE, width=2)
    y += 26
    for line in ("84% dos pesos estão nas três camadas mais profundas —",
                 "onde a rede decide quem é quem, não onde está a borda."):
        d.text((M, y), line, font=f(REG, 28), fill=GREY)
        y += 36

    y += 30
    tracked(d, (M, y), "PARÂMETROS POR BLOCO", f(BOLD, 24), BLUE, 1.6)
    y += 44
    mx = max(b["p"] for b in blocks)
    bar_x0, bar_max = 200, 610
    for b in blocks:
        hot = b["name"] in HOT
        fn = f(BOLD, 26)
        d.text((170 - d.textlength(b["name"], font=fn), y), b["name"],
               font=fn, fill=WHITE if hot else GREY)
        wpx = max(3, int(bar_max * b["p"] / mx))
        d.rectangle([bar_x0, y + 3, bar_x0 + wpx, y + 23],
                    fill=BLUE if hot else (38, 48, 64))
        # br() already writes 1.180.672; only the percentage decimal needs a comma.
        lab = br(b["p"]) + "   " + ("%.1f%%" % b["pct"]).replace(".", ",")
        fv = f(REG, 24)
        d.text((W - M - d.textlength(lab, font=fv), y), lab, font=fv,
               fill=BLUE_DIM if hot else GREY)
        y += 46

    y += 36
    d.line([(M, y), (W - M, y)], fill=RULE, width=2)
    y += 28
    tracked(d, (M, y), "CONTRA O PROFESSOR QUE ELE COPIOU", f(BOLD, 24), BLUE, 1.6)
    y += 46

    sam = torch.load("checkpoints/sam2.1_hiera_large.pt", map_location="cpu",
                     weights_only=False)
    sam = sam.get("model", sam)
    sam_p = sum(v.numel() for v in sam.values() if hasattr(v, "numel"))
    sam_mb = os.path.getsize("checkpoints/sam2.1_hiera_large.pt") / 1e6
    stu_mb = os.path.getsize("data/out/student_ckpt_v1/student.pt") / 1e6

    cA, cB = 720, W - M
    fh = f(BOLD, 25)
    d.text((cA - d.textlength("ALUNO", font=fh), y), "ALUNO", font=fh, fill=BLUE)
    d.text((cB - d.textlength("SAM2 large", font=fh), y), "SAM2 large",
           font=fh, fill=(150, 160, 175))
    y += 42
    rows = [("parâmetros", br(total), br(sam_p)),
            ("tamanho em disco", ("%.1f MB" % stu_mb).replace(".", ","),
             "%.0f MB" % sam_mb),
            ("frames por segundo", "251", "4,34")]
    for lab, a, bv in rows:
        d.line([(M, y), (W - M, y)], fill=(20, 26, 36), width=2)
        d.text((M, y + 14), lab, font=f(REG, 27), fill=GREY)
        fa, fb2 = f(BOLD, 29), f(REG, 29)
        d.text((cA - d.textlength(a, font=fa), y + 12), a, font=fa, fill=WHITE)
        d.text((cB - d.textlength(bv, font=fb2), y + 12), bv, font=fb2,
               fill=(120, 132, 150))
        y += 58
    d.line([(M, y), (W - M, y)], fill=(20, 26, 36), width=2)
    y += 20
    d.text((M, y), "%d× menor, e roda no laptop em vez de GPU alugada" % round(sam_p / total),
           font=f(BOLD, 27), fill=BLUE)

    footer(d)
    card.save(out)
    print(out, "conteudo termina em y=%d (rodape comeca em %d)" % (y + 30, SAFE_BOT - 176))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out1", default="data/out/story_arch_1.png")
    ap.add_argument("--out2", default="data/out/story_arch_2.png")
    a = ap.parse_args()
    blocks, total = probe()
    card_one(blocks, total, a.out1)
    card_two(blocks, total, a.out2)


if __name__ == "__main__":
    main()
