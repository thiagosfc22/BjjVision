"""Export the student to ONNX, with the parity proof built in.

    python scripts/export_student.py --ckpt data/out/student_ckpt_v3/student.pt

The student was distilled to end up on a phone: 3.35M parameters, 6.7 MB in
fp16, standard ops only (conv3x3, GroupNorm, SiLU, maxpool, nearest
upsample). ONNX is the platform-neutral artifact -- ONNX Runtime Mobile runs
it on both Android (NNAPI) and iOS (CoreML EP); a native CoreML export can
come later if the target settles on iPhone.

An export without a parity check is a guess with a file extension, so this
script refuses to just write the file: it runs the same frames through torch
and onnxruntime and reports the worst absolute logit difference and any
argmax disagreement. Conversion bugs (a mis-lowered GroupNorm, a resize mode
mismatch) show up HERE, on a disposable checkpoint, not in a phone demo with
the model that finally works.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.student import UNetStudent  # noqa: E402

W, H = 320, 180


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/out/student_ckpt_v3/student.pt")
    ap.add_argument("--out", default=None, help="default: <ckpt_dir>/student.onnx")
    ap.add_argument("--frames", type=int, default=32,
                    help="random frames for the parity check")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = UNetStudent(ck["width"])
    model.load_state_dict(ck["model"])
    model.eval()
    out = Path(a.out or Path(a.ckpt).parent / "student.onnx")

    x = torch.randn(1, 3, H, W)
    torch.onnx.export(model, (x,), str(out), input_names=["image"],
                      output_names=["logits"], opset_version=18,
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    # The dynamo exporter externalises weights into a sidecar .onnx.data; a
    # phone deployment wants ONE file, so fold them back in.
    import onnx
    m = onnx.load(str(out))
    sidecar = out.with_suffix(".onnx.data")
    onnx.save(m, str(out), save_as_external_data=False)
    sidecar.unlink(missing_ok=True)
    print(f"exportado {out}  ({out.stat().st_size / 1e6:.1f} MB, opset 18, arquivo unico)")
    print(f"  origem: {a.ckpt}  git_sha {ck.get('git_sha', 'desconhecido')}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst, disagree = 0.0, 0
    for _ in range(a.frames):
        xi = rng.standard_normal((1, 3, H, W)).astype(np.float32)
        with torch.no_grad():
            yt = model(torch.from_numpy(xi)).numpy()
        yo = sess.run(None, {"image": xi})[0]
        worst = max(worst, float(np.abs(yt - yo).max()))
        disagree += int((yt.argmax(1) != yo.argmax(1)).sum())
    print(f"paridade em {a.frames} frames: max|dif logits| {worst:.2e}, "
          f"pixels com argmax divergente: {disagree}")
    if worst > 1e-3 or disagree:
        raise SystemExit("PARIDADE FALHOU -- nao use este onnx")

    n, t0 = 60, time.time()
    xi = rng.standard_normal((1, 3, H, W)).astype(np.float32)
    for _ in range(n):
        sess.run(None, {"image": xi})
    print(f"onnxruntime CPU: {n / (time.time() - t0):.0f} fps "
          "(celular com NPU tende a superar isso)")


if __name__ == "__main__":
    main()
