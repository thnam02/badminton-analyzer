#!/usr/bin/env bash
# Install Detectron2 + DensePose for body-surface muscle overlay.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Use python -m pip (avoids shell alias conflicts with the wheel CLI on some setups).
python -m pip install -U pip 'setuptools<81'
python -m pip install wheel ninja

# Torch must be installed before detectron2 setup.py runs (not visible in pip build isolation).
python -m pip install 'torch==2.4.1' 'torchvision==0.19.1'

echo "Installing Detectron2 (CPU/macOS build may take several minutes)..."
# --no-build-isolation: detectron2 setup.py imports torch during metadata/build.
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6' \
  --no-build-isolation \
  --no-cache-dir

echo "Installing DensePose project package..."
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6#subdirectory=projects/DensePose' \
  --no-build-isolation \
  --no-cache-dir

ASSETS="$ROOT/app/cv/densepose/assets"
mkdir -p "$ASSETS"
for name in densepose_rcnn_R_50_FPN_s1x.yaml Base-DensePose-RCNN-FPN.yaml; do
  CONFIG="$ASSETS/$name"
  if [[ ! -f "$CONFIG" ]]; then
    curl -fsSL \
      "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/${name}" \
      -o "$CONFIG"
  fi
done

python - <<'PY'
from app.cv.densepose.compat import apply_pillow_shims

apply_pillow_shims()
from detectron2.config import get_cfg
from densepose.config import add_densepose_config

cfg = get_cfg()
add_densepose_config(cfg)
print("DensePose ready.")
PY

echo "DensePose ready. Weights download on first analyze (or set DENSEPOSE_WEIGHTS)."
