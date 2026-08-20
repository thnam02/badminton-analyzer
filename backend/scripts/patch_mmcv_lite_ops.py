"""Patch mmcv-lite / mmdet so RTMPose inference works without mmcv._ext."""

from __future__ import annotations

import shutil
from pathlib import Path


OPS_INIT = '''# Patched for mmcv-lite CPU inference (no mmcv._ext).
from .nms import batched_nms, nms, soft_nms, nms_match

try:
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention  # type: ignore
except Exception:  # pragma: no cover
    MultiScaleDeformableAttention = None  # type: ignore

__all__ = [
    "nms",
    "soft_nms",
    "batched_nms",
    "nms_match",
    "MultiScaleDeformableAttention",
]
'''

LAYERS_INIT = '''# Patched subset of mmdet.models.layers for RTMDet/RTMPose (no mmcv.ops).
from .activations import SiLU
from .bbox_nms import fast_nms, multiclass_nms
from .brick_wrappers import (AdaptiveAvgPool2d, FrozenBatchNorm2d,
                             adaptive_avg_pool2d)
from .conv_upsample import ConvUpsample
from .csp_layer import CSPLayer
from .dropblock import DropBlock
from .ema import ExpMomentumEMA
from .inverted_residual import InvertedResidual
from .matrix_nms import mask_matrix_nms
from .normed_predictor import NormedConv2d, NormedLinear
from .pixel_decoder import PixelDecoder, TransformerEncoderPixelDecoder
from .positional_encoding import (LearnedPositionalEncoding,
                                  SinePositionalEncoding,
                                  SinePositionalEncoding3D)
from .res_layer import ResLayer, SimplifiedBasicBlock
from .se_layer import ChannelAttention, DyReLU, SELayer

__all__ = [
    'SiLU', 'fast_nms', 'multiclass_nms', 'AdaptiveAvgPool2d',
    'FrozenBatchNorm2d', 'adaptive_avg_pool2d', 'ConvUpsample', 'CSPLayer',
    'DropBlock', 'ExpMomentumEMA', 'InvertedResidual', 'mask_matrix_nms',
    'NormedConv2d', 'NormedLinear', 'PixelDecoder',
    'TransformerEncoderPixelDecoder', 'LearnedPositionalEncoding',
    'SinePositionalEncoding', 'SinePositionalEncoding3D', 'ResLayer',
    'SimplifiedBasicBlock', 'ChannelAttention', 'DyReLU', 'SELayer',
]
'''


def main() -> None:
    import mmcv
    import mmdet

    mmcv_root = Path(mmcv.__file__).parent
    mmdet_root = Path(mmdet.__file__).parent
    shim_nms = Path(__file__).resolve().parents[1] / "vendor" / "mmcv_ops_shim" / "nms.py"

    ops_dir = mmcv_root / "ops"
    shutil.copy2(shim_nms, ops_dir / "nms.py")
    (ops_dir / "__init__.py").write_text(OPS_INIT)

    layers_init = mmdet_root / "models" / "layers" / "__init__.py"
    layers_init.write_text(LAYERS_INIT)

    # Re-apply MMPose EDPoseHead patch if present.
    try:
        from scripts.patch_mmpose_for_mmcv_lite import main as patch_mmpose

        patch_mmpose()
    except Exception as exc:  # pragma: no cover
        print(f"MMPose patch skipped: {exc}")

    print(f"Patched {ops_dir / 'nms.py'}")
    print(f"Patched {ops_dir / '__init__.py'}")
    print(f"Patched {layers_init}")


if __name__ == "__main__":
    main()
