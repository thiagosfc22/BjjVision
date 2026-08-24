#!/usr/bin/env bash
# Provision a VAST.AI GPU instance for BjjVision.
# Run ON the instance:  bash remote/bootstrap_vast.sh
set -euo pipefail

# -- interpreter resolution ---------------------------------------------------
# Do not assume `python`/`pip` on PATH point at the right environment. The VAST
# PyTorch image keeps its environment at /venv/main and does NOT activate it,
# not even under `bash -lc`: `python` does not exist at all, and bare `pip`
# resolves to the system one, which dies on `--upgrade pip` with
# "Cannot uninstall pip 24.0, RECORD file not found". So: detect, then always
# go through `$PY -m pip`, and never touch pip itself.
if [ -n "${BJJ_PYTHON:-}" ] && [ -x "${BJJ_PYTHON}" ]; then
  PY="$BJJ_PYTHON"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PY="$VIRTUAL_ENV/bin/python"
elif [ -x /venv/main/bin/python ]; then
  PY="/venv/main/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "no python interpreter found (tried \$BJJ_PYTHON, \$VIRTUAL_ENV, /venv/main, python3, python)" >&2
  exit 1
fi
# Everything downstream -- ./bjj, ultralytics, any shelled-out console script --
# resolves through PATH, so put the chosen environment in front of it.
export PATH="$(dirname "$PY"):$PATH"
export BJJ_PYTHON="$PY"
# --no-cache-dir: the container disk on these boxes is 16 GB total and the frame
# directory is what it is for; a 4.6 GB wheel cache is not.
PIP=("$PY" -m pip --disable-pip-version-check --no-input --no-cache-dir)

echo "== interpreter =="
echo "  $PY  ($("$PY" -c 'import sys; print(sys.version.split()[0], "prefix", sys.prefix)'))"

echo "== system deps =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg git curl libgl1 libglib2.0-0 rsync

echo "== torch =="
# Only install if genuinely absent. The stock image already carries a CUDA build
# newer than any pin we would write here; reinstalling from the cu124 index
# would downgrade a working install.
if ! "$PY" -c "import torch" 2>/dev/null; then
  "${PIP[@]}" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
"$PY" - <<'PY'
import sys, torch
ok = torch.cuda.is_available()
print(f"  torch {torch.__version__}  cuda={ok}  {torch.cuda.get_device_name(0) if ok else '-'}")
if not ok:
    sys.exit("CUDA not visible to torch -- this is a paid GPU box; stop and fix that "
             "before running anything, do not fall through to CPU")
PY

echo "== python deps =="
# No `pip install --upgrade pip`: on this image it breaks (see above), and a
# working pip is not something this bootstrap needs to improve.
"${PIP[@]}" install -q numpy opencv-python-headless pyyaml tqdm rich typer pillow scipy scikit-learn \
                     ultralytics anthropic lap pyarrow

echo "== SAM2 =="
if ! "$PY" -c "import sam2" 2>/dev/null; then
  [ -d /opt/sam2/.git ] || git clone -q https://github.com/facebookresearch/sam2.git /opt/sam2
  # --no-build-isolation is not optional here. SAM2 lists torch in its
  # build-system requires, so an isolated build downloads a SECOND torch plus
  # its CUDA wheels (~2.5 GB) into a temp env -- which is exactly how this
  # bootstrap filled a 16 GB container disk and died on ENOSPC. The image's
  # torch is already installed and is the one we want to build against.
  # SAM2_BUILD_CUDA=0 skips the optional connected-components extension: it
  # needs nvcc, it is only used by mask postprocessing we do not enable, and
  # every number in PROMPT.md was measured on Metal without it.
  SAM2_BUILD_CUDA=0 "${PIP[@]}" install -q -e /opt/sam2 --no-build-isolation
fi
"$PY" -c "import sam2; print('  sam2 ok')"

echo "== checkpoints =="
mkdir -p checkpoints
CKPT=checkpoints/sam2.1_hiera_large.pt
# 898 MB. Container disk on these boxes is small and the frame directory is the
# real consumer, so say so before spending it.
echo "  disk free on $(df -P . | awk 'NR==2{print $6}'): $(df -Ph . | awk 'NR==2{print $4}')"
if [ ! -s "$CKPT" ]; then
  curl -fL --progress-bar -o "$CKPT" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi
ls -lh "$CKPT"

echo "== yolo weights =="
# sync-up ships yolo11x-pose.pt in the repo root; ultralytics would otherwise
# fetch it on first use and stall the run mid-calibration.
"$PY" - <<'PY'
from ultralytics import YOLO
YOLO("yolo11x-pose.pt")
print("  yolo weights ready")
PY

echo "== bf16 autocast smoke =="
# The torch.autocast("cuda", bfloat16) branch in segment.py has never executed --
# every run so far was Metal. Prove the branch works on a 3-line matmul now,
# rather than discovering it 40 minutes into a paid run.
"$PY" - <<'PY'
import torch
with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
    y = (torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda"))
print(f"  autocast ok, dtype={y.dtype}")
PY

echo "== package =="
# --no-deps deliberately: the deps are installed above against the image's torch,
# and letting a resolver loose here can pull a numpy that breaks it. The editable
# install is a convenience only; ./bjj sets PYTHONPATH and does not rely on it.
"${PIP[@]}" install -q -e . --no-deps || echo "  editable install failed (non-fatal; ./bjj sets PYTHONPATH)"
chmod +x ./bjj 2>/dev/null || true

echo
./bjj doctor
echo
echo "ready. next:"
echo "  ./bjj frames <slug> --frames 10777:11077 --calib-frames 1350:1500"
echo "  ./bjj run <slug> --config config/vast.yaml --frames 10777:11077 --no-llm"
