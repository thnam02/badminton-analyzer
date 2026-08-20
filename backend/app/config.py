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

    upload_dir: Path = ROOT_DIR / "uploads"
    output_dir: Path = ROOT_DIR / "outputs"
    cors_origins: str = "http://localhost:3000"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
