"""Patch installed mmpose so import works with mmcv-lite (no CUDA/ops)."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    import mmpose

    root = Path(mmpose.__file__).parent
    transformer_init = root / "models/heads/transformer_heads/__init__.py"
    transformer_init.write_text(
        "from typing import Any\n\n"
        "EDPoseHead: Any = None  # requires mmcv.ops; unused for RTMPose\n"
        "__all__ = ['EDPoseHead']\n"
    )

    heads_init = root / "models/heads/__init__.py"
    text = heads_init.read_text()
    needle = "from .transformer_heads import EDPoseHead\n"
    replacement = "EDPoseHead = None  # patched for mmcv-lite\n"
    if needle in text:
        heads_init.write_text(text.replace(needle, replacement))
    print(f"Patched {heads_init}")


if __name__ == "__main__":
    main()
