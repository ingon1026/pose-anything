import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def make_det(box=(100, 100, 200, 180), label="obj", score=0.9):
    """트래커 테스트 공용 검출 dict (스키마 단일 정의)."""
    return {"label": label, "box": np.array(box, dtype=float), "score": score,
            "mask": None}


def make_filtered_track(z=1.0, extent=(0.2, 0.1, 0.05), converged=True):
    """융합 필터 달린 Track 공용 픽스처 (시드 규약 단일 정의).

    converged=True면 정상 관측 30프레임으로 정직하게 수렴시킨다 —
    P를 직접 찔러 위조하면 실제 정상상태와 어긋난 조건을 검증하게 된다."""
    from roboworld_perception.fusion import TrackFilter
    from roboworld_perception.tracker import Track
    t = Track(1, "obj", np.array([100, 100, 200, 180], float), 0.9)
    t.filter = TrackFilter(np.array([0.0, 0.0, z]), np.log(extent))
    if converged:
        for _ in range(30):
            t.filter.predict(1 / 15)
            t.filter.fuse_pos(np.array([0.0, 0.0, z]))
            t.filter.fuse_extent(np.log(extent))
    return t
