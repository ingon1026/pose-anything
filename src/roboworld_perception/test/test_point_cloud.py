"""객체별 점군(/perception/points) 데이터 블록 회귀.

ROS 없이 돈다 — pipeline.cloud_chunk 는 numpy 만 쓴다. 노드 쪽 발행 경로
(publish_points 게이트·회수)는 rclpy 가 필요해 여기서 안 다룬다.
"""
import numpy as np
from conftest import K

from roboworld_perception.geometry import mask_depth_to_points
from roboworld_perception.pipeline import POINT_DTYPE, cloud_chunk

BLUE, GREEN = (66, 133, 244), (52, 168, 83)  # overlay.PALETTE[0], [1] (BGR)


def make_scene(u0=100, u1=140, v0=80, v1=110, z=1.0):
    """상면 하나짜리 합성 씬 — uint16 mm depth + 마스크."""
    depth = np.zeros((240, 320), dtype=np.uint16)
    mask = np.zeros((240, 320), dtype=bool)
    depth[v0:v1, u0:u1] = int(z * 1000)
    mask[v0:v1, u0:u1] = True
    return mask, depth


def test_chunk_is_the_point_set_the_obb_used():
    """점 개수·좌표가 mask_depth_to_points 와 정확히 같아야 한다 —
    노브를 따로 정하면 상자와 점군이 다른 것을 보여준다."""
    mask, depth = make_scene()
    ref = mask_depth_to_points(mask, depth, K, depth_scale=0.001)
    chunk = cloud_chunk(mask, depth, K, BLUE)
    assert len(chunk) == len(ref) > 100
    assert np.allclose(np.stack([chunk["x"], chunk["y"], chunk["z"]], 1),
                       ref, atol=1e-6)


def test_chunk_metric_geometry():
    mask, depth = make_scene(u0=100, u1=140, v0=80, v1=110, z=1.0)
    chunk = cloud_chunk(mask, depth, K, BLUE)
    assert abs(np.median(chunk["z"]) - 1.0) < 1e-6
    # erode_px=3 이 축당 1px 씩 깎으므로 화소중심 간 폭은 40-2-1 = 37px,
    # 높이는 30-2-1 = 27px. 1px = z/fx = 1/300 m.
    assert abs(np.ptp(chunk["x"]) - 37 / 300) < 1e-3
    assert abs(np.ptp(chunk["y"]) - 27 / 300) < 1e-3


def test_color_is_packed_rgb_from_bgr_palette():
    """PALETTE 는 cv2 BGR 이라 팩할 때 뒤집는다 — 마커와 같은 뒤집기다."""
    mask, depth = make_scene()
    for color in (BLUE, GREEN):
        b, g, r = color
        packed = cloud_chunk(mask, depth, K, color)["rgb"].view(np.uint32)
        assert (packed == (r << 16) | (g << 8) | b).all()
    assert POINT_DTYPE.itemsize == 16  # x,y,z,rgb float32 — point_step


def test_empty_mask_yields_empty_chunk():
    """발행할 점이 없어도 dtype 은 유지된다 — 빈 점군이 곧 회수다."""
    mask, depth = make_scene()
    chunk = cloud_chunk(np.zeros_like(mask), depth, K, BLUE)
    assert len(chunk) == 0 and chunk.dtype == POINT_DTYPE
