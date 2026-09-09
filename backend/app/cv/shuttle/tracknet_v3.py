"""TrackNetV3 adapter — keeps model I/O isolated behind RawShuttleDetection."""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import settings
from app.cv.shuttle.base import RawShuttleDetection, ShuttleTracker

logger = logging.getLogger(__name__)


class TrackNetV3Error(RuntimeError):
    """Raised when TrackNetV3 is misconfigured or inference fails."""


class TrackNetV3Adapter(ShuttleTracker):
    """Invoke an external TrackNetV3 install via ``predict.py`` and parse CSV.

    Expected layout (configurable via settings):
    - ``SHUTTLE_TRACKNET_ROOT`` — cloned qaz812345/TrackNetV3 repo
    - ``SHUTTLE_TRACKNET_WEIGHTS`` — TrackNet_best.pt
    - ``SHUTTLE_INPAINTNET_WEIGHTS`` — optional InpaintNet_best.pt
    """

    name = "tracknetv3"

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        tracknet_weights: str | Path | None = None,
        inpaintnet_weights: str | Path | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.root = Path(root or settings.shuttle_tracknet_root)
        self.tracknet_weights = Path(
            tracknet_weights or settings.shuttle_tracknet_weights
        )
        inpaint = inpaintnet_weights or settings.shuttle_inpaintnet_weights
        self.inpaintnet_weights = Path(inpaint) if inpaint else None
        self.python_executable = python_executable or sys.executable

    def is_available(self) -> bool:
        if not str(self.root).strip() or not str(self.tracknet_weights).strip():
            return False
        predict = self.root / "predict.py"
        return (
            self.root.is_dir()
            and predict.is_file()
            and self.tracknet_weights.is_file()
        )

    def track(self, video_path: Path) -> list[RawShuttleDetection]:
        if not self.is_available():
            raise TrackNetV3Error(
                "TrackNetV3 is not available. Set SHUTTLE_TRACKNET_ROOT to a "
                "cloned TrackNetV3 repo and SHUTTLE_TRACKNET_WEIGHTS to "
                "TrackNet_best.pt (see https://github.com/qaz812345/TrackNetV3)."
            )
        video_path = Path(video_path)
        if not video_path.is_file():
            raise TrackNetV3Error(f"Video not found: {video_path}")

        with tempfile.TemporaryDirectory(prefix="tracknetv3_") as tmp:
            save_dir = Path(tmp)
            cmd = [
                self.python_executable,
                str(self.root / "predict.py"),
                "--video_file",
                str(video_path.resolve()),
                "--tracknet_file",
                str(self.tracknet_weights.resolve()),
                "--save_dir",
                str(save_dir),
            ]
            if self.inpaintnet_weights is not None and self.inpaintnet_weights.is_file():
                cmd.extend(
                    [
                        "--inpaintnet_file",
                        str(self.inpaintnet_weights.resolve()),
                    ]
                )
            if settings.shuttle_tracknet_large_video:
                cmd.append("--large_video")

            logger.info("Running TrackNetV3: %s", " ".join(cmd))
            completed = subprocess.run(
                cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise TrackNetV3Error(
                    "TrackNetV3 predict.py failed "
                    f"(code={completed.returncode}): {completed.stderr[-2000:]}"
                )

            csv_path = _find_prediction_csv(save_dir, video_path)
            if csv_path is None:
                raise TrackNetV3Error(
                    f"TrackNetV3 produced no CSV under {save_dir}. "
                    f"stdout tail: {completed.stdout[-1000:]}"
                )
            # Copy aside for debugging if configured.
            if settings.shuttle_keep_raw_csv:
                dest = video_path.with_name(f"{video_path.stem}_tracknet_raw.csv")
                shutil.copy2(csv_path, dest)
            return parse_tracknet_csv(csv_path)


def parse_tracknet_csv(csv_path: Path) -> list[RawShuttleDetection]:
    """Convert TrackNetV3 CSV (Frame,Visibility,X,Y) into raw detections."""
    rows: list[RawShuttleDetection] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_index = int(float(row["Frame"]))
            visibility = int(float(row.get("Visibility", 0)))
            x_raw = row.get("X", "")
            y_raw = row.get("Y", "")
            try:
                x_px = float(x_raw) if x_raw not in ("", "None", "nan") else None
                y_px = float(y_raw) if y_raw not in ("", "None", "nan") else None
            except ValueError:
                x_px, y_px = None, None
            if visibility <= 0 or x_px is None or y_px is None:
                rows.append(
                    RawShuttleDetection(
                        frame_index=frame_index,
                        x_px=None,
                        y_px=None,
                        visibility=0,
                        confidence=0.0,
                    )
                )
            else:
                # TrackNet does not emit calibrated confidence; treat visible as 1.0.
                rows.append(
                    RawShuttleDetection(
                        frame_index=frame_index,
                        x_px=x_px,
                        y_px=y_px,
                        visibility=visibility,
                        confidence=1.0,
                    )
                )
    rows.sort(key=lambda d: d.frame_index)
    return rows


def _find_prediction_csv(save_dir: Path, video_path: Path) -> Path | None:
    stem = video_path.stem
    candidates = [
        save_dir / f"{stem}_ball.csv",
        save_dir / f"{stem}.csv",
        save_dir / "ball.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    csv_files = sorted(save_dir.rglob("*.csv"))
    return csv_files[0] if csv_files else None
