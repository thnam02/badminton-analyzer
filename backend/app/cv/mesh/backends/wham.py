"""WHAM backend: temporally stable camera-space SMPL mesh from video."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from app.config import ROOT_DIR, settings
from app.cv.mesh.backends.base import MeshRecoveryBackend, MeshRecoveryError
from app.cv.mesh.types import CameraParams, MeshFrame, MeshSequence
from app.schemas.pose import PoseSequence

logger = logging.getLogger(__name__)

_DEFAULT_WHAM_ROOT = ROOT_DIR / "vendor" / "WHAM"
_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


class WhamBackend(MeshRecoveryBackend):
    """Wraps WHAM; prefers RTMPose tracks (no ViTPose) when pose is available."""

    name = "wham"

    def __init__(self, *, wham_root: Path | None = None) -> None:
        configured = (wham_root or settings.mesh_wham_root or "").strip()
        self._wham_root = Path(configured).expanduser() if configured else _DEFAULT_WHAM_ROOT

    def is_available(self) -> bool:
        root = self._wham_root
        if not root.is_dir():
            return False
        return (root / "wham_api.py").is_file() or (root / "lib" / "models" / "wham.py").is_file()

    def recover(
        self,
        video_path: Path,
        *,
        pose_sequence: PoseSequence | None = None,
        image_width: int,
        image_height: int,
    ) -> MeshSequence:
        if not self.is_available():
            raise MeshRecoveryError(
                f"WHAM backend unavailable at {self._wham_root}. "
                "Run backend/scripts/bootstrap_wham.sh and finish body-model setup."
            )

        video_path = video_path.resolve()
        output_dir = settings.output_dir / f"{video_path.stem}_wham_work"
        output_dir.mkdir(parents=True, exist_ok=True)

        if pose_sequence is not None and pose_sequence.frame_count > 0:
            from app.cv.mesh.backends.wham_rtmpose import run_wham_with_rtmpose

            results = run_wham_with_rtmpose(
                wham_root=self._wham_root,
                video_path=video_path,
                output_dir=output_dir,
                pose_sequence=pose_sequence,
                image_width=image_width,
                image_height=image_height,
                min_confidence=settings.pose_confidence_threshold,
            )
        else:
            results = self._run_wham_api(video_path, output_dir)

        faces = self._load_smpl_faces()
        return _parse_wham_results(
            results,
            video_name=video_path.name,
            image_width=image_width,
            image_height=image_height,
            faces=faces,
            j_regressor=self._load_j_regressor(),
        )

    def _run_wham_api(self, video_path: Path, output_dir: Path) -> dict:
        root = self._wham_root.resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        with _chdir(root):
            try:
                from wham_api import WHAM_API  # type: ignore
            except ImportError as exc:
                raise MeshRecoveryError(
                    f"Could not import WHAM_API from {root}: {exc}. "
                    "Provide RTMPose pose_sequence (normal analyze path), or finish "
                    "full WHAM+ViTPose install (see WHAM INSTALL.md)."
                ) from exc

            logger.info("Running WHAM_API on %s (cwd=%s)", video_path, root)
            try:
                api = WHAM_API()
                results, _tracking, _slam = api(
                    str(video_path),
                    output_dir=str(output_dir),
                    run_global=False,
                    visualize=False,
                )
            except Exception as exc:  # noqa: BLE001
                raise MeshRecoveryError(f"WHAM inference failed: {exc}") from exc

        if not results:
            raise MeshRecoveryError("WHAM returned no person tracks")
        return results

    def _load_smpl_faces(self) -> np.ndarray:
        bundled = _ASSETS_DIR / "smpl_faces.npy"
        if bundled.is_file():
            return np.load(bundled).astype(np.int32)

        faces = _try_faces_from_wham(self._wham_root)
        if faces is not None:
            _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(bundled, faces)
            return faces

        faces = _try_faces_from_smplx()
        if faces is not None:
            _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(bundled, faces)
            return faces

        # WHAM auxiliary pack often ships faces here after body_models.tar.gz
        aux = self._wham_root / "dataset" / "body_models" / "smpl_faces.npy"
        if aux.is_file():
            faces = np.load(aux).astype(np.int32)
            _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(bundled, faces)
            return faces

        raise MeshRecoveryError(
            "SMPL faces not found. Finish WHAM body-model setup "
            "(dataset/body_models.tar.gz) or place smpl_faces.npy under "
            f"{_ASSETS_DIR}."
        )

    def _load_j_regressor(self) -> np.ndarray | None:
        bundled = _ASSETS_DIR / "smpl_j_regressor.npy"
        if bundled.is_file():
            return np.load(bundled).astype(np.float64)
        wham_reg = self._wham_root / "dataset" / "body_models" / "J_regressor_wham.npy"
        if wham_reg.is_file():
            return np.load(wham_reg).astype(np.float64)
        return _try_j_regressor_from_smplx()


@contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _try_faces_from_wham(wham_root: Path) -> np.ndarray | None:
    candidates = list((wham_root / "dataset" / "body_models" / "smpl").glob("*.pkl"))
    candidates += list((wham_root / "data" / "body_models" / "smpl").glob("*.pkl"))
    if not candidates:
        return None
    try:
        from smplx import SMPL  # type: ignore

        model = SMPL(model_path=str(candidates[0].parent), gender="neutral")
        return np.asarray(model.faces, dtype=np.int32)
    except Exception:  # noqa: BLE001
        return None


def _try_faces_from_smplx() -> np.ndarray | None:
    model_path = (settings.mesh_smpl_model_path or "").strip()
    if not model_path:
        return None
    try:
        from smplx import SMPL  # type: ignore

        path = Path(model_path)
        model = SMPL(
            model_path=str(path.parent if path.is_file() else path),
            gender="neutral",
        )
        return np.asarray(model.faces, dtype=np.int32)
    except Exception:  # noqa: BLE001
        return None


def _try_j_regressor_from_smplx() -> np.ndarray | None:
    model_path = (settings.mesh_smpl_model_path or "").strip()
    search_roots = []
    if model_path:
        search_roots.append(Path(model_path))
    search_roots.append(_DEFAULT_WHAM_ROOT / "dataset" / "body_models" / "smpl")
    try:
        from smplx import SMPL  # type: ignore
    except ImportError:
        return None
    for root in search_roots:
        root = root.expanduser()
        if root.is_file():
            root = root.parent
        if not root.is_dir():
            continue
        try:
            model = SMPL(model_path=str(root), gender="neutral")
            reg = model.J_regressor
            if hasattr(reg, "toarray"):
                arr = reg.toarray()
            else:
                arr = np.asarray(reg.detach().cpu() if hasattr(reg, "detach") else reg)
            bundled = _ASSETS_DIR / "smpl_j_regressor.npy"
            _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(bundled, arr.astype(np.float64))
            return arr.astype(np.float64)
        except Exception:  # noqa: BLE001
            continue
    return None


def _parse_wham_results(
    results: object,
    *,
    video_name: str,
    image_width: int,
    image_height: int,
    faces: np.ndarray,
    j_regressor: np.ndarray | None = None,
) -> MeshSequence:
    """Parse WHAM API / demo.py result dicts into MeshSequence."""
    if results is None:
        raise MeshRecoveryError("WHAM returned empty results")

    person_id, payload = _select_primary_person(results)
    verts_seq = _extract_verts(payload)
    frame_ids = _extract_frame_ids(payload, verts_seq.shape[0])

    focal = (
        float(settings.mesh_focal_length)
        if settings.mesh_focal_length > 0
        else float(np.sqrt(image_width**2 + image_height**2))
    )
    camera = CameraParams(
        fx=focal,
        fy=focal,
        cx=image_width * 0.5,
        cy=image_height * 0.5,
        R=np.eye(3, dtype=np.float64),
        t=np.zeros(3, dtype=np.float64),
    )

    frames: list[MeshFrame] = []
    for i in range(verts_seq.shape[0]):
        verts = np.asarray(verts_seq[i], dtype=np.float64)
        joints = None
        if j_regressor is not None and verts.shape[0] == j_regressor.shape[1]:
            joints = j_regressor @ verts
        frames.append(
            MeshFrame(
                frame_index=int(frame_ids[i]),
                vertices=verts,
                faces=faces,
                camera=camera,
                joints_3d=joints,
                confidence=1.0,
            )
        )

    logger.info(
        "WHAM primary track id=%s frames=%d verts=%d focal=%.1f",
        person_id,
        len(frames),
        verts_seq.shape[1],
        focal,
    )
    return MeshSequence(
        video=video_name,
        frames=frames,
        backend="wham",
        notes=(
            "WHAM camera-space SMPL (RTMPose tracks when available; CLIFF focal). "
            "Overlay-only feasibility milestone."
        ),
    )


def _select_primary_person(results: object) -> tuple[object, dict]:
    if not isinstance(results, dict) or not results:
        raise MeshRecoveryError(f"Unrecognized WHAM result type: {type(results)}")

    scored: list[tuple[int, object, dict]] = []
    for key, value in results.items():
        if not isinstance(value, dict):
            continue
        n = 0
        for name in ("verts_cam", "verts", "vertices", "pred_verts", "poses_body", "pose"):
            arr = value.get(name)
            if arr is not None:
                n = int(np.asarray(arr).shape[0])
                break
        if n > 0:
            scored.append((n, key, value))

    if not scored:
        raise MeshRecoveryError(
            "WHAM results contain no usable person tracks "
            f"(keys={list(results.keys())})"
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    _n, key, payload = scored[0]
    return key, payload


def _extract_verts(payload: dict) -> np.ndarray:
    for name in ("verts_cam", "verts", "vertices", "pred_verts"):
        if name in payload and payload[name] is not None:
            verts = np.asarray(payload[name], dtype=np.float64)
            while verts.ndim == 4 and verts.shape[0] == 1:
                verts = verts[0]
            if verts.ndim == 3 and verts.shape[-1] == 3:
                return verts
            raise MeshRecoveryError(
                f"WHAM '{name}' expected shape (T,V,3), got {verts.shape}"
            )
    raise MeshRecoveryError(
        "WHAM results missing camera-space vertices "
        "(expected verts_cam or verts). Keys="
        f"{sorted(payload.keys())}"
    )


def _extract_frame_ids(payload: dict, t_count: int) -> np.ndarray:
    for name in ("frame_ids", "frame_id", "frames", "frame_idx"):
        if name in payload and payload[name] is not None:
            raw = payload[name]
            if hasattr(raw, "detach"):
                raw = raw.detach().cpu().numpy()
            ids = np.asarray(raw).reshape(-1)
            if ids.size == t_count:
                return ids.astype(np.int64)
    return np.arange(t_count, dtype=np.int64)
