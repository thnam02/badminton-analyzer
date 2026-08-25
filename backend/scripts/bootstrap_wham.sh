#!/usr/bin/env bash
# Clone WHAM into vendor/ and document SMPL model setup for mesh recovery.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
VENDOR="$REPO_ROOT/vendor"
WHAM_DIR="$VENDOR/WHAM"
ASSETS_DIR="$ROOT/app/cv/mesh/assets"

mkdir -p "$VENDOR"
mkdir -p "$ASSETS_DIR"

if [[ ! -d "$WHAM_DIR/.git" ]]; then
  echo "Cloning WHAM into $WHAM_DIR ..."
  git clone --depth 1 https://github.com/yohanshin/WHAM.git "$WHAM_DIR"
else
  echo "WHAM already present at $WHAM_DIR"
fi

cat <<EOF

WHAM cloned at: $WHAM_DIR

Next steps (manual — requires SMPL registration):

  1. cd $WHAM_DIR
  2. Follow WHAM README / INSTALL.md (conda or pip deps for this machine).
  3. Register at https://smpl.is.tue.mpg.de/ and https://smplify.is.tue.mpg.de/
  4. Place SMPL body models under WHAM's dataset/body_models (or run WHAM's
     fetch_demo_data script for checkpoints + models).
  5. Optionally export faces for faster overlay startup:

       cd $ROOT
       source .venv/bin/activate   # if using the app venv
       python - <<'PY'
from pathlib import Path
import numpy as np
from smplx import SMPL
root = Path("$WHAM_DIR")
for cand in [
    root / "dataset" / "body_models" / "smpl",
    root / "data" / "body_models" / "smpl",
]:
    if cand.is_dir() and any(cand.glob("*.pkl")):
        model = SMPL(model_path=str(cand), gender="neutral")
        out = Path("$ASSETS_DIR")
        np.save(out / "smpl_faces.npy", np.asarray(model.faces, dtype=np.int32))
        reg = model.J_regressor
        arr = reg.toarray() if hasattr(reg, "toarray") else np.asarray(reg.detach().cpu() if hasattr(reg, "detach") else reg)
        np.save(out / "smpl_j_regressor.npy", arr.astype(np.float64))
        print("Wrote", out / "smpl_faces.npy", "and smpl_j_regressor.npy")
        break
else:
    print("SMPL pkl not found yet — finish WHAM body-model setup first.")
PY

  6. Add to badminton-analytics/.env:

       MESH_ENABLED=true
       MESH_BACKEND=wham
       MESH_WHAM_ROOT=$WHAM_DIR
       # optional if models live outside WHAM:
       # MESH_SMPL_MODEL_PATH=/path/to/smpl

This milestone only produces outputs/{uuid}_mesh.mp4 (semi-transparent SMPL
over the original video) plus RTMPose shoulder/elbow/hip alignment logs.
No SMPLer-X, muscle meshes, or activation yet.

EOF
