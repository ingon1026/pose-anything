"""fit_plane(stats=...) 이 내놓는 진단값이 실제 잡음 수준과 맞는지."""
import numpy as np

from roboworld_perception.geometry import fit_plane
from roboworld_perception.pipeline import _fit_support_plane


def _ring_depth(sigma_m, z0=1.0, shape=(240, 320), seed=0):
    """z0 m 앞의 fronto-parallel 평면 + 가우스 잡음, 가운데를 판 링 마스크."""
    rng = np.random.default_rng(seed)
    depth = (z0 + rng.normal(0.0, sigma_m, shape)).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[60:180, 80:240] = True
    mask[90:150, 120:200] = False  # 물체 자리 — 링만 남긴다
    return depth, mask


K = np.array([[600.0, 0, 160.0], [0, 600.0, 120.0], [0, 0, 1.0]])


def test_stats_rms_matches_known_noise():
    # 문턱(3mm) 안쪽 σ 라면 인라이어 절단이 거의 없어 RMS ≈ σ 여야 한다.
    # σ 가 문턱에 가까우면 절단 가우스라 RMS < σ 로 나오는 것이 정상이다.
    for sigma in (0.0005, 0.001):
        depth, ring = _ring_depth(sigma)
        stats = {}
        plane = fit_plane(depth, K, mask=ring, dist_thresh=0.003, stats=stats)
        assert plane is not None
        assert 0.0 < stats["inlier"] <= 1.0
        assert abs(stats["rms"] - sigma) < 0.15 * sigma


def test_fit_plane_stats_optional_and_contract_kept():
    depth, ring = _ring_depth(0.0005)
    assert fit_plane(depth, K, mask=ring) is not None  # stats 없이도 그대로


def test_support_plane_returns_stats():
    depth, _ = _ring_depth(0.0005)
    obj = np.zeros(depth.shape, np.uint8)  # 링은 _fit_support_plane 이 만든다
    obj[100:140, 130:190] = 1
    dets = [{"mask": obj}]
    plane, thr, stats = _fit_support_plane(dets, depth, K, 0.001)
    assert plane is not None and thr == 0.003
    assert set(stats) == {"inlier", "rms"}
