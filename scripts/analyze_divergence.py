"""필터 발산 분석 — raw 관측을 TrackFilter 에 다시 먹여 혁신을 재현한다.

`run_offline.py --raw` 가 남긴 {tag}_raw.csv 는 평활 전 관측이라, 이것만 있으면
필터를 오프라인으로 재현할 수 있다(GPU·모델 불필요). "필터가 틀린 상태에
수렴해 있는가" 를 사후에 재는 용도다.

    python3 scripts/analyze_divergence.py <tag>_raw.csv [라벨 ...]

**해석할 때 주의** — 발행 CSV(raw 아님)의 궤적을 진실로 놓고 필터를 다시
돌리는 방식은 쓰지 말 것. 발행 CSV 는 발행된 프레임만 담아서 공백이 그대로
dt 가 되는데, 필터는 dt 를 0.5s 로 clip 하므로 시간이 어긋나 인위적으로
뒤처진다 — test3 pink block 에서 상태오차가 실제 3.1mm 인데 206mm 로 나왔다.
굳이 그렇게 해야 하면 공백(dt > 0.2s)에서 필터를 재시드할 것.
"""
import csv
import sys

import numpy as np

sys.path.insert(0, "src/roboworld_perception")
from roboworld_perception.fusion import TrackFilter, pos_r_extra  # noqa: E402

WINDOW = 12          # 슬라이딩 창 길이(프레임) — 키프레임 주기 5 의 배수
SIGN_MIN = 0.75      # 창 안 부호 편향 |sum(sign)|/W 하한
BIAS_MIN_MM = 20.0   # 창 안 혁신 평균 크기 하한


def load(path, label):
    rows = [r for r in csv.DictReader(open(path)) if r["label"] == label]
    rows.sort(key=lambda r: float(r["stamp"]))
    return rows


def replay(rows):
    """raw 관측 -> [(t, 혁신(3), 수락, 상태(3))].

    r_extra 는 fusion.pos_r_extra 를 그대로 부른다 — 복사하면 파이프라인이
    항을 추가할 때 이 도구가 조용히 다른 필터를 재현하게 된다."""
    t0 = float(rows[0]["stamp"])
    c0 = np.array([float(rows[0][k]) for k in ("rx", "ry", "rz")])
    e0 = np.sort([float(rows[0][k]) for k in ("re1", "re2", "re3")])[::-1]
    f = TrackFilter(c0, np.log(e0 + 1e-9))
    out, prev, steps = [], t0, 0
    for r in rows[1:]:
        t = float(r["stamp"])
        dt = float(np.clip(t - prev, 1e-3, 0.5))
        prev = t
        steps = 0 if r["keyframe"] == "1" else steps + 1
        f.predict(dt)
        z = np.array([float(r[k]) for k in ("rx", "ry", "rz")])
        nu = z - f.x
        sp = float(np.linalg.norm(f.v))
        ok = f.fuse_pos(z, pos_r_extra(sp, dt, steps))
        out.append((t - t0, nu.copy(), ok, f.x.copy()))
    return out


def windowed(nu, w=WINDOW):
    """축별 슬라이딩 창의 (부호 편향, 혁신 평균 크기 mm) — 축 최대값."""
    s = np.sign(nu)
    sg = np.zeros(len(nu))
    bm = np.zeros(len(nu))
    for i in range(len(nu)):
        lo = max(0, i - w + 1)
        sg[i] = np.abs(s[lo:i + 1].sum(axis=0)).max() / (i - lo + 1)
        bm[i] = np.abs(nu[lo:i + 1].mean(axis=0)).max() * 1000
    return sg, bm


def main(path, labels):
    for lab in labels:
        rows = load(path, lab)
        if len(rows) < 30:
            print(f"{lab:14s} 행 부족({len(rows)})")
            continue
        res = replay(rows)
        t = np.array([r[0] for r in res])
        nu = np.array([r[1] for r in res])
        x = np.array([r[3] for r in res])
        # 정지 씬 가정: 초반 8초 중앙값을 진실로 본다
        truth = np.median(x[t < 8], axis=0)
        err = np.linalg.norm(x - truth, axis=1) * 1000
        sg, bm = windowed(nu)
        flag = (sg >= SIGN_MIN) & (bm >= BIAS_MIN_MM)
        bad, good = err > 30, err <= 10
        det = 100 * flag[bad].mean() if bad.sum() else float("nan")
        fp = 100 * flag[good].mean() if good.sum() else float("nan")
        print(f"{lab:14s} n={len(t):4d} 발산 {bad.sum():3d}({100*bad.mean():3.0f}%) "
              f"| 검출 {det:3.0f}% 오탐 {fp:3.0f}% "
              f"| 상태오차 중앙 {np.median(err):6.1f}mm max {err.max():6.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:] or ["book", "black bag", "keyboard"])
