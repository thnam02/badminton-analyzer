"""Compatibility shims for Detectron2 v0.6 on modern Pillow (>=10)."""

from __future__ import annotations


def apply_pillow_shims() -> None:
    """Restore removed Pillow resampling aliases used by Detectron2 0.6."""
    import PIL.Image as Image

    bilinear = getattr(Image, "BILINEAR", None)
    bicubic = getattr(Image, "BICUBIC", None)
    resampling = getattr(Image, "Resampling", None)
    if bilinear is None and resampling is not None:
        bilinear = resampling.BILINEAR
    if bicubic is None and resampling is not None:
        bicubic = resampling.BICUBIC
    if bilinear is None:
        bilinear = 2
    if bicubic is None:
        bicubic = 3

    aliases = {
        "LINEAR": bilinear,
        "BILINEAR": bilinear,
        "CUBIC": bicubic,
        "BICUBIC": bicubic,
        "ANTIALIAS": getattr(Image, "LANCZOS", bilinear),
    }
    for name, value in aliases.items():
        if not hasattr(Image, name):
            setattr(Image, name, value)


apply_pillow_shims()
