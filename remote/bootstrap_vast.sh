#!/usr/bin/env bash
# Provision a VAST.AI GPU instance for BjjVision.
# Run ON the instance:  bash remote/bootstrap_vast.sh
set -euo pipefail

echo "== system deps =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg git curl libgl1 libglib2.0-0 rsync

echo "== python deps =="
python -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -q --upgrade pip
pip install -q numpy opencv-python-headless pyyaml tqdm rich typer pillow scipy scikit-learn \
                ultralytics anthropic lap

echo "== SAM2 =="
if ! python -c "import sam2" 2>/dev/null; then
  git clone -q https://github.com/facebookresearch/sam2.git /opt/sam2
  pip install -q -e /opt/sam2
fi

echo "== checkpoints =="
mkdir -p checkpoints
CKPT=checkpoints/sam2.1_hiera_large.pt
[ -f "$CKPT" ] || curl -L --progress-bar -o "$CKPT" \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# ultralytics fetches yolo11x-pose.pt on first use, but pre-pull so the run
# does not stall on a download mid-calibration
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo11x-pose.pt")
print("yolo weights ready")
PY

echo "== install package =="
pip install -q -e .

echo
python -m bjjvision.cli doctor
echo
echo "ready. next:"
echo "  python -m bjjvision.cli frames <slug>"
echo "  python -m bjjvision.cli run <slug>"
