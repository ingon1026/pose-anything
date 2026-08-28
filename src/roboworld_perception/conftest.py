import sys
from pathlib import Path

from pathlib import Path

import numpy as np

# 패키지 루트 — 설정 파일(launch/·rviz/)을 읽는 테스트가 쓴다.
PKG_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(Path(__file__).parent))

# 합성 장면용 내부 파라미터 (테스트 공용 단일 정의)
K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])


def rect_scene(depth_mm, u0, u1, v0, v1):
    """직사각 마스크 + 균일 depth 합성 장면 — 테스트 공용 단일 정의.

    240x320 · uint16 mm. 규약이 두 곳에 있으면 한쪽만 바뀌었을 때 두 테스트가
    서로 다른 세계를 검증하게 되므로 여기 한 벌만 둔다. K(300, 160/120)도
    이 파일 것을 같이 쓴다. 해상도·K 가 다른 장면은 각 파일이 따로 만든다
    (test_geometry.make_box_scene 은 480x640·fx=600 이라 여기 대상이 아니다).
    """
    depth = np.zeros((240, 320), np.uint16)
    mask = np.zeros((240, 320), bool)
    sl = (slice(v0, v1), slice(u0, u1))
    mask[sl] = True
    depth[sl] = depth_mm
    return mask, depth


def make_det(box=(100, 100, 200, 180), label="obj", score=0.9):
    """트래커 테스트 공용 검출 dict (스키마 단일 정의)."""
    return {"label": label, "box": np.array(box, dtype=float), "score": score,
            "mask": None}


def make_filtered_track(z=1.0, extent=(0.2, 0.1, 0.05), converged=True,
                        pub_score_min=0.0):
    """융합 필터 달린 Track 공용 픽스처 (시드 규약 단일 정의).

    converged=True면 정상 관측 30프레임으로 정직하게 수렴시킨다 —
    P를 직접 찔러 위조하면 실제 정상상태와 어긋난 조건을 검증하게 된다."""
    from roboworld_perception.fusion import TrackFilter
    from roboworld_perception.tracker import Track
    t = Track(1, "obj", np.array([100, 100, 200, 180], float), 0.9,
              pub_score_min=pub_score_min)
    t.filter = TrackFilter(np.array([0.0, 0.0, z]), np.log(extent))
    if converged:
        for _ in range(30):
            t.filter.predict(1 / 15)
            t.filter.fuse_pos(np.array([0.0, 0.0, z]))
            t.filter.fuse_extent(np.log(extent))
    return t
