"""물체별 속도(twist) 를 body frame 으로 옮기는 순수 헬퍼 — ROS 미의존.

칼만 필터(fusion.TrackFilter)의 속도 상태는 카메라 optical frame 이다.
nav_msgs/Odometry 규약은 "twist 는 child_frame_id(여기서는 물체 프레임) 기준"
이라(REP 103), 발행 직전에 물체 자세로 되돌려야 한다. 이 변환과 그에 딸린
6x6 공분산 조립을 perception_node.publish() 밖으로 뺀 것은 ROS 메시지 타입
없이 numpy 만으로 테스트하기 위해서다.
"""
import numpy as np

# 각속도는 추정하지 않는다 — 회전 상태는 필터에 없다(geometry.ObbResult.R 은
# 매 프레임 재적합이지 추적 상태가 아니다). 0 은 ROS 소비자에게 "정확히
# 안다"는 뜻이라 대신 큰 분산으로 "모른다"를 명시한다 —
# pose_covariance.UNESTIMATED_ROTATION_VARIANCE_RAD2 와 같은 취지다. 다만
# 그 값(π² rad²)은 자세 각도 자체의 표현 범위(±180°)에서 나온 것이라 여기
# (rad²/s², 각속도)에는 그 유도가 안 맞는다. 1e3 은 흔한 그랩 플래닝의 신뢰
# 구간이 걸러낼 만큼 크게 잡은 상수다.
ANGULAR_VEL_UNKNOWN = 1e3


def body_twist(R, v, vel_var):
    """물체 프레임(body frame) 속도와 그 6x6 공분산을 만든다.

    R: (3,3) 물체 자세 — 열이 물체 로컬 X/Y/Z 축을 카메라 좌표로 표현한다
       (geometry.ObbResult.R 과 같은 규약). R 은 정규직교라 역행렬 = 전치.
    v: (3,) m/s, optical frame 속도 (fusion.TrackFilter.v).
    vel_var: (3,) m²/s², optical frame 축별 속도 분산
             (fusion.TrackFilter.P[:, 1, 1] — P 는 (3,2,2) 축별 [pos,vel]
             공분산이라 [:, 1, 1] 이 축별 속도 분산이다).

    반환: (v_body(3,) m/s, cov(36,) float64 — 6x6 row-major, 순서 x y z rx ry rz)

    소비자가 v_body 를 다시 optical frame 속도로 되돌리려면 R @ v_body 를
    계산하면 된다(v_body = R.T @ v 의 역변환).
    """
    R = np.asarray(R, dtype=float)
    v = np.asarray(v, dtype=float)
    vel_var = np.asarray(vel_var, dtype=float)

    v_body = R.T @ v

    # Cov(R.T @ v) = R.T @ Cov(v) @ R. 필터가 축을 독립으로 다루므로
    # Cov(v) = diag(vel_var) — hyp.pose.covariance 발행에 쓰는 위치 분산
    # 조합과 같은 형태다.
    linear_cov = R.T @ np.diag(vel_var) @ R

    cov = np.zeros((6, 6), dtype=np.float64)
    cov[:3, :3] = linear_cov
    cov[3, 3] = cov[4, 4] = cov[5, 5] = ANGULAR_VEL_UNKNOWN
    # 선형-각 교차항은 0 그대로 둔다 — 각속도를 추정하지 않으므로 상관도 없다.
    return v_body, cov.reshape(36)
