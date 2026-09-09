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

    # Technique reference profiles (metric bands); severity bands are fixed below.
    technique_reference_profile_id: str = ""
    technique_default_camera_view: str = "SIDE"
    # Seed values for provisional smash profile construction (not scientific norms).
    technique_min_contact_elbow_angle_deg: float = 150.0
    technique_min_knee_contribution_deg: float = 12.0
    technique_max_peak_elbow_omega_lead_frames: int = 2
    technique_min_peak_elbow_omega_lead_frames: int = -8
    technique_min_acceleration_phase_fraction: float = 0.12
    technique_max_contact_wrist_y_normalized: float = 0.58
    technique_min_follow_through_speed_ratio: float = 0.30
    technique_min_follow_through_frames: int = 2

    # Optional OpenAI coaching layer (Responses API)
    openai_coaching_enabled: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Independent shuttlecock tracking (does not alter pose / ESTIMATED_CONTACT)
    shuttle_enabled: bool = False
    shuttle_backend: str = "tracknetv3"  # tracknetv3 | heuristic
    shuttle_tracknet_root: str = ""
    shuttle_tracknet_weights: str = ""
    shuttle_inpaintnet_weights: str = ""
    shuttle_tracknet_large_video: bool = False
    shuttle_keep_raw_csv: bool = False
    shuttle_interp_max_gap: int = 3
    shuttle_debug_trail_length: int = 16

    # Independent racket detection (optional; does not alter contact / coaching)
    racket_enabled: bool = False
    racket_backend: str = "pose_guided"  # pose_guided | yolo
    racket_hitting_hand: str = ""  # LEFT | RIGHT | empty → infer from pose
    racket_pose_confidence_threshold: float = 0.3
    racket_shaft_length_frac: float = 0.18
    racket_roi_pad_frac: float = 0.06
    racket_yolo_weights: str = ""
    racket_yolo_class_id: int = 0
    racket_yolo_conf_threshold: float = 0.25
    racket_track_enabled: bool = True
    racket_track_max_gap: int = 3
    racket_track_max_jump: float = 0.25
    racket_debug_trail_length: int = 12

    # ContactResolver (kinematic + optional shuttle/racket)
    contact_search_radius: int = 8
    contact_min_tracked_confidence: float = 0.55
    contact_max_shuttle_racket_dist: float = 0.12
    contact_max_racket_wrist_dist: float = 0.18
    contact_min_visible_frames: int = 3
    contact_ambiguity_margin: float = 0.08

    upload_dir: Path = ROOT_DIR / "uploads"
    output_dir: Path = ROOT_DIR / "outputs"
    cors_origins: str = "http://localhost:3000"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
