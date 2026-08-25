"""WHAM recovery that reuses RTMPose tracks (skips ViTPose/YOLO)."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.cv.mesh.backends.base import MeshRecoveryError
from app.cv.skeleton import COCO_KEYPOINT_NAMES
from app.schemas.pose import PoseSequence

logger = logging.getLogger(__name__)

# RTMPose / COCO-17 names → WHAM expects (T, 17, 3) in COCO order.
_MIN_TRACK_FRAMES = 8


@contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def pose_sequence_to_tracking(
    pose_sequence: PoseSequence,
    *,
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> dict[int, dict[str, Any]]:
    """Build a single-person WHAM tracking dict from RTMPose."""
    from collections import defaultdict

    frame_ids: list[int] = []
    keypoints: list[np.ndarray] = []
    bboxes: list[np.ndarray] = []

    for frame in pose_sequence.frames:
        kp = np.zeros((17, 3), dtype=np.float64)
        valid = 0
        xs: list[float] = []
        ys: list[float] = []
        for i, name in enumerate(COCO_KEYPOINT_NAMES):
            joint = frame.keypoints.get(name)
            if joint is None:
                continue
            conf = float(joint.confidence)
            x = float(joint.x) * image_width
            y = float(joint.y) * image_height
            kp[i, 0] = x
            kp[i, 1] = y
            kp[i, 2] = conf
            if conf >= min_confidence:
                valid += 1
                xs.append(x)
                ys.append(y)
        if valid < 6 or not xs:
            continue
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))
        scale = max(max(xs) - min(xs), max(ys) - min(ys)) / 200.0 * 1.2
        scale = max(scale, 0.05)
        frame_ids.append(int(frame.frame_index))
        keypoints.append(kp)
        bboxes.append(np.array([cx, cy, scale], dtype=np.float64))

    if len(frame_ids) < _MIN_TRACK_FRAMES:
        raise MeshRecoveryError(
            f"RTMPose track too short for WHAM ({len(frame_ids)} frames; "
            f"need >= {_MIN_TRACK_FRAMES})."
        )

    bbox_arr = np.asarray(bboxes, dtype=np.float64)
    fps_guess = 30.0
    kernel = int(int(fps_guess / 2) / 2) * 2 + 1
    kernel = max(kernel, 3)
    if bbox_arr.shape[0] >= kernel:
        from scipy.signal import medfilt

        bbox_arr = np.array([medfilt(param, kernel) for param in bbox_arr.T]).T

    # defaultdict(list) so FeatureExtractor can append features / init keys.
    track: dict[str, Any] = defaultdict(list)
    track["frame_id"] = np.asarray(frame_ids, dtype=np.int64)
    track["keypoints"] = np.asarray(keypoints, dtype=np.float64)
    track["bbox"] = bbox_arr
    return {0: track}


def run_wham_with_rtmpose(
    *,
    wham_root: Path,
    video_path: Path,
    output_dir: Path,
    pose_sequence: PoseSequence,
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> dict:
    """Run FeatureExtractor + WHAM network using RTMPose tracks (no ViTPose)."""
    root = wham_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    device = _resolve_device()
    _ensure_checkpoints(root)

    with _chdir(root):
        # Newer SciPy moved gaussian_filter1d; WHAM still imports the old path.
        try:
            import scipy.ndimage.filters  # noqa: F401
        except ImportError:
            import types

            import scipy.ndimage as _ndi

            shim = types.ModuleType("scipy.ndimage.filters")
            shim.gaussian_filter1d = _ndi.gaussian_filter1d
            sys.modules["scipy.ndimage.filters"] = shim

        # CPU-safe torch.load (WHAM checkpoints were saved on CUDA machines).
        import torch

        _orig_load = torch.load

        def _cpu_load(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            try:
                return _orig_load(*args, **{**kwargs, "weights_only": False})
            except TypeError:
                return _orig_load(*args, **kwargs)

        torch.load = _cpu_load  # type: ignore[assignment]

        try:
            from configs.config import get_cfg_defaults
            from lib.data.datasets import CustomDataset
            from lib.models import build_body_model, build_network
            from lib.models.preproc.extractor import FeatureExtractor
        except Exception as exc:  # noqa: BLE001
            raise MeshRecoveryError(
                f"Could not import WHAM modules from {root}: {exc}. "
                "Install soft deps: pip install joblib yacs smplx loguru "
                "progress einops timm==0.4.9 scikit-image"
            ) from exc

        cfg = get_cfg_defaults()
        cfg.merge_from_file("configs/yamls/demo.yaml")
        cfg.DEVICE = device
        cfg.FLIP_EVAL = False  # faster / simpler on CPU

        tracking = pose_sequence_to_tracking(
            pose_sequence,
            image_width=image_width,
            image_height=image_height,
            min_confidence=min_confidence,
        )

        logger.info(
            "WHAM (RTMPose tracks) video=%s frames=%d device=%s",
            video_path.name,
            len(tracking[0]["frame_id"]),
            device,
        )

        extractor = FeatureExtractor(device.lower())
        # Quiet progress.bar spam (can flood uvicorn logs / terminals).
        try:
            from progress.bar import Bar

            class _QuietBar(Bar):
                def next(self, n=1):  # noqa: A003
                    self.index = self.index + n
                    if self.index >= self.max:
                        self.finish()

            import lib.models.preproc.extractor as _ext_mod

            _ext_mod.Bar = _QuietBar  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        tracking = extractor.run(str(video_path), tracking)

        length = _video_frame_count(video_path)
        slam = np.zeros((max(length, 1), 7), dtype=np.float64)
        slam[:, 3] = 1.0

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()

        network = build_network(
            cfg,
            build_body_model(cfg.DEVICE, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN),
        )
        network.eval()

        dataset = CustomDataset(cfg, tracking, slam, image_width, image_height, fps)
        results: dict = {}
        import torch

        with torch.no_grad():
            for batch in dataset:
                if batch is None:
                    break
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = batch
                pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )
                results[_id] = {
                    "poses_body": pred["poses_body"].cpu().squeeze(0).numpy(),
                    "poses_root_cam": pred["poses_root_cam"].cpu().squeeze(0).numpy(),
                    "betas": pred["betas"].cpu().squeeze(0).numpy(),
                    "verts_cam": (
                        pred["verts_cam"] + pred["trans_cam"].unsqueeze(1)
                    )
                    .cpu()
                    .numpy(),
                    "frame_id": frame_id,
                }

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import joblib

            joblib.dump(results, output_dir / "wham_results.pth")
        except Exception:  # noqa: BLE001
            pass

        if not results:
            raise MeshRecoveryError("WHAM network returned no person tracks")
        return results


def _resolve_device() -> str:
    import torch

    configured = (settings.device or "cpu").strip().lower()
    if configured.startswith("cuda") and torch.cuda.is_available():
        return configured
    return "cpu"


def _ensure_checkpoints(root: Path) -> None:
    needed = [
        root / "checkpoints" / "hmr2a.ckpt",
        root / "checkpoints" / "wham_vit_bedlam_w_3dpw.pth.tar",
        root / "dataset" / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl",
        root / "dataset" / "body_models" / "J_regressor_wham.npy",
        root / "dataset" / "body_models" / "smpl_mean_params.npz",
    ]
    missing = [str(p.relative_to(root)) for p in needed if not p.is_file()]
    if missing:
        raise MeshRecoveryError(
            "WHAM checkpoints/body models incomplete. Missing: "
            + ", ".join(missing)
            + ". Re-run fetch_demo_data.sh (needs wget) or "
            "backend/scripts/fetch_wham_demo_data.sh."
        )


def _video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n
