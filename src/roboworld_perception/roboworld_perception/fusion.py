"""확률 융합 코어 — 이진 게이트(score·depth·size)를 대체하는 축별 칼만 필터.

상태: 위치 3축 × [c, ċ](등속 CV, 축별 해석적 2×2 — 역행렬 없음)
      + log-extent 3 스칼라(내림차순 정렬 — C1 실험: 축대응은 축퇴 축에서 불안정).
게이트: 위치/extent 블록 분리 Mahalanobis, χ²(0.999, 3dof).
탈출: 별도 장치 없음 — 거부 중에도 predict가 P를 t³로 키워 게이트가 스스로
열린다. 탈출 시간이 편차 Δ^(2/3)에 비례해 "거짓말이 클수록 오래 버텨야
믿어주는" 성질이 구조에서 나온다 (docs/fusion_design_2026-08.md).

설계 불변식 (test_fusion.py가 property test로 고정):
 1. 거부가 이어지는 동안 수락 영역은 단조 증가 (교착 불가능)
 2. R̂은 게이트를 통과한 관측만으로 갱신되고 하한(물리 모델) 밑으로 못 내려간다
 3. 탈출 시간은 편차에 단조 — 큰 편차(가리개)일수록 오래 거부된다
"""
import numpy as np

CHI2_3DOF = 16.266   # χ²(0.999, 3) — 프레임당 오기각 0.1%

# ── 보정 상수 (근거: docs/fusion_design_2026-08.md §4, calib 재산출 예정) ──
# 가속도 잡음(DWNA): 지배 요구는 "부분 가림의 중간 편차(~100mm)도 실측 최대
# 가림 2.6s보다 오래 거부"(0.02 → 3.9s ✓). 부수 성질: 554mm 침입 12.5s 거부,
# 정직한 5cm 변화 2.2s 탈출(교착 없음 — 유한이면 충분), 벨트 등속은 CV가
# 거부 없이 추적. 0.07은 554mm 기준만 보고 고른 값이라 부분 가림 꼬리를
# 채택했다 (test4/5 회귀 실측: keyboard z-std 1.7→69mm).
SIGMA_A = 0.02       # m/s²
# log-extent 랜덤워크: 현행 EMA(0.4) 실효 이득을 넘지 않는 수준
Q_LOGE = 2.7e-3      # /s
# 물리 R 하한 (test2 raw 실측 프레임간 잔차 중앙값 수준, m 단위 std)
R_POS_FLOOR = 2.0e-3
R_LOGE_FLOOR = 0.05
# R̂ 상한 배율. 검은 가방류(depth 흡수 표면)의 실측 raw 잡음은 σ 60~150mm로
# 하한의 30~75배 — 상한이 이보다 좁으면 그 물체는 만성 기각된다. 상한에서도
# 554mm 침입은 pooled d²≈31 > χ²로 기각된다 (property test로 고정).
RHAT_CAP = 2500.0    # 분산 배율 (σ로는 50×)
RHAT_ALPHA = 0.05    # R̂ EMA 이득
# 프레임별 가산 측정잡음(r_extra)의 재료 — R 모델 상수는 전부 이 모듈에 둔다:
SYNC_STD = 0.011     # s — color/depth 짝짓기 시차 p95 (test2/4/5 실측 11ms)
FLOW_STEP_STD = 0.002  # m/프레임 — 키프레임 이후 LK 누적 드리프트 (test2 실측)
V0_STD = 0.05        # m/s — 속도 초기·rescue 재팽창 불확실성


class TrackFilter:
    """트랙 1개의 위치+크기 상태. 회전은 필터 밖(기존 match_axes/slerp)."""

    def __init__(self, center, log_extent):
        self.x = np.asarray(center, float).copy()     # (3,) m
        self.v = np.zeros(3)                          # (3,) m/s
        # 축별 2×2 공분산 [[Pcc,Pcv],[Pcv,Pvv]] — 초기 불확실성은 크게
        # (신뢰는 관측이 벌어온다: M-of-N 승격 전까지 발행 안 됨)
        self.P = np.tile(np.diag([R_POS_FLOOR ** 2 * 25, V0_STD ** 2]), (3, 1, 1))
        self.le = np.asarray(log_extent, float).copy()  # (3,) log m, 내림차순
        self.Ple = np.full(3, R_LOGE_FLOOR ** 2 * 25)
        # per-track 측정잡음 적응(분산). 하한=물리 모델, 수락 관측만 갱신 —
        # 게이트를 넓히는 주 채널은 어디까지나 P(거부 시 무조건 성장)이고
        # R̂은 물체별 잡음 규모(실측 40× 편차)를 따라가는 보조 채널이다.
        self.rhat_pos = np.full(3, R_POS_FLOOR ** 2)
        self.rhat_le = np.full(3, R_LOGE_FLOOR ** 2)

    # ── 예측 ──────────────────────────────────────────────
    def predict(self, dt):
        """시간 전진. 관측 유무와 무관하게 매 프레임 호출 — 거부/미관측
        프레임에도 P가 자라는 것이 교착 불가능성의 전부다."""
        dt = float(np.clip(dt, 1e-3, 0.5))  # 스탬프 이상 방어
        self.x += self.v * dt
        F = np.array([[1.0, dt], [0.0, 1.0]])
        g = np.array([0.5 * dt * dt, dt])
        Q = np.outer(g, g) * SIGMA_A ** 2
        self.P = F @ self.P @ F.T + Q
        self.Ple += Q_LOGE * dt

    # ── 융합 (게이트 통과 시에만 상태 갱신) ────────────────
    def fuse_pos(self, z, r_extra=0.0):
        """위치 관측 (3,) m. r_extra: 프레임별 가산 분산(σ_sync² 등).
        반환: 게이트 수락 여부."""
        z = np.asarray(z, float)
        R = self.rhat_pos + r_extra
        nu = z - self.x
        P_prior = self.P[:, 0, 0].copy()
        S = P_prior + R
        if float(np.sum(nu * nu / S)) > CHI2_3DOF:
            return False
        K = self.P[:, 0, :] / S[:, None]              # (3,2) 축별 이득
        self.x += K[:, 0] * nu
        self.v += K[:, 1] * nu
        # (I−KH)P — H=[1,0]이므로 P에서 K·(P의 위치 행)만 빼면 된다
        self.P = self.P - K[:, :, None] * self.P[:, 0:1, :]
        # R̂ 갱신 — 불편형 E[ν²]=P⁻+R̂+r_extra 에서 R̂만 남긴 EMA, 하한/상한 클립
        # (r_extra 몫을 빼야 sync·flow 잡음이 R̂에 이중 계상되지 않는다)
        inno2 = nu * nu - P_prior - r_extra
        self.rhat_pos = np.clip(
            (1 - RHAT_ALPHA) * self.rhat_pos + RHAT_ALPHA * inno2,
            R_POS_FLOOR ** 2, RHAT_CAP * R_POS_FLOOR ** 2)
        return True

    def fuse_extent(self, log_e):
        """정렬 log-extent 관측 (3,). 반환: 수락 여부."""
        le = np.asarray(log_e, float)
        nu = le - self.le
        P_prior = self.Ple.copy()
        S = P_prior + self.rhat_le
        if float(np.sum(nu * nu / S)) > CHI2_3DOF:
            return False
        K = self.Ple / S
        self.le += K * nu
        self.Ple *= (1 - K)
        inno2 = nu * nu - P_prior
        self.rhat_le = np.clip(
            (1 - RHAT_ALPHA) * self.rhat_le + RHAT_ALPHA * inno2,
            R_LOGE_FLOOR ** 2, RHAT_CAP * R_LOGE_FLOOR ** 2)
        return True

    # ── 소비자 뷰 ─────────────────────────────────────────
    @property
    def center(self):
        return self.x

    @property
    def extent_sorted(self):
        return np.exp(self.le)

    @property
    def pos_var(self):
        """축별 위치 분산 (3,) m² — pose.covariance 발행용."""
        return self.P[:, 0, 0]

    @property
    def pos_std(self):
        """축별 위치 표준편차 (3,) m — 발행 판정용."""
        return np.sqrt(self.pos_var)

    def innovation_std(self, axis):
        """해당 축의 혁신 표준편차 sqrt(P+R̂) — 게이트와 같은 척도.
        depth 침입 판정 등 외부 술어가 상태 배열을 직접 인덱싱하지 않게
        하는 유일한 접근 창구다."""
        return float(np.sqrt(self.P[axis, 0, 0] + self.rhat_pos[axis]))

    def inflate_for_rescue(self, radius):
        """가림 후 재등장(rescue) 시 위치·속도 불확실성 재팽창 —
        가림 중 물체가 이동·교체됐을 수 있다 (OC-SORT 재등장 복구 취지)."""
        self.P[:, 0, 0] += radius * radius
        self.P[:, 1, 1] += V0_STD ** 2
