"""Research-only Action Quality Assessment (AQA) scaffold.

Does **not** replace the deterministic biomechanics / technique pipeline.
No training is performed in this package.
"""

from app.research.aqa.factory import get_aqa_model
from app.research.aqa.mock_model import MockActionQualityModel
from app.research.aqa.protocol import ActionQualityModel
from app.research.aqa.schemas import ActionQualityPrediction, AQASample

__all__ = [
    "AQASample",
    "ActionQualityModel",
    "ActionQualityPrediction",
    "MockActionQualityModel",
    "get_aqa_model",
]
