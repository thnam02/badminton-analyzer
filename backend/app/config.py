from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mmpose_config: str = ""
    mmpose_checkpoint: str = ""
    device: str = "cpu"
    pose_confidence_threshold: float = 0.5

    # Temporal preprocessing (filter → short-gap interpolate → Savitzky–Golay)
    pose_interp_max_gap: int = 5
    pose_savgol_window: int = 7
    pose_savgol_polyorder: int = 2

    # Overlay: EMA for body-anchored labels; warn HUD only below this conf
    overlay_anchor_smoothing: float = 0.35
    overlay_low_confidence_warn: float = 0.5
    # DensePose muscle overlay retired for mesh feasibility milestone
    overlay_muscle_enabled: bool = False
    overlay_muscle_base_alpha: float = 0.55
    overlay_muscle_smoothing: float = 0.4

    # DensePose (kept for optional revive; not used by default overlay)
    densepose_config: str = ""
    densepose_weights: str = ""
    densepose_score_threshold: float = 0.5
    densepose_min_person_pixels: int = 400
    densepose_crop_padding: float = 0.20
    densepose_fail_loud: bool = True
    densepose_debug: bool = False
    densepose_debug_frames: int = 5
    densepose_debug_show_parts: bool = False

    # 3D mesh feasibility — WHAM only for this milestone
    mesh_enabled: bool = True
    mesh_backend: str = "wham"
    mesh_wham_root: str = ""
    mesh_smplerx_root: str = ""
    mesh_smpl_model_path: str = ""
    mesh_overlay_alpha: float = 0.45
    mesh_show_reprojection: bool = True
    mesh_focal_length: float = 0.0  # 0 → CLIFF focal sqrt(w^2+h^2)

    # Technique rule thresholds (V1 smash)
    technique_min_contact_elbow_angle_deg: float = 150.0
    technique_min_knee_contribution_deg: float = 12.0
    technique_max_peak_elbow_omega_lead_frames: int = 2
    technique_min_peak_elbow_omega_lead_frames: int = -8
    technique_min_acceleration_phase_fraction: float = 0.12
    technique_max_contact_wrist_y_normalized: float = 0.58
    technique_min_follow_through_speed_ratio: float = 0.30
    technique_min_follow_through_frames: int = 2

    upload_dir: Path = ROOT_DIR / "uploads"
    output_dir: Path = ROOT_DIR / "outputs"
    cors_origins: str = "http://localhost:3000"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
