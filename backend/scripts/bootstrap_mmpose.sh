#!/usr/bin/env bash
# Bootstrap OpenMMLab deps. Prefers full mmcv (with ops); falls back to lite + shims.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U 'pip' 'setuptools<81' 'wheel'
pip install -r requirements.txt

# Prefer full mmcv with ops (needed by RTMDet NMS). Requires a working C++ toolchain.
if ! python -c "import mmcv._ext" 2>/dev/null; then
  echo "Building mmcv==2.1.0 with ops (this can take several minutes)..."
  if ! MMCV_WITH_OPS=1 pip install 'mmcv==2.1.0' --no-build-isolation --no-cache-dir; then
    echo "Full mmcv build failed; installing mmcv-lite + shims."
    pip install 'mmcv-lite==2.1.0'
    python scripts/patch_mmcv_lite_ops.py
  fi
fi

pip install 'mmdet==3.3.0'
pip install 'mmpose==1.3.2' --no-deps
pip install json-tricks munkres scipy pillow
pip install 'chumpy @ git+https://github.com/mattloper/chumpy.git' --no-build-isolation
pip install -e ./vendor/xtcocotools_shim

python scripts/patch_mmpose_for_mmcv_lite.py

python -c "from mmpose.apis import MMPoseInferencer; print('MMPose ready')"
