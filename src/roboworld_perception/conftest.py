import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def make_det(box=(100, 100, 200, 180), label="obj", score=0.9):
    """트래커 테스트 공용 검출 dict (스키마 단일 정의)."""
    return {"label": label, "box": np.array(box, dtype=float), "score": score,
            "mask": None}
