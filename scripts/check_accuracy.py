#!/usr/bin/env python3
"""정확도 합격선 판정 — 기준(bf16) 실행 대비 후보 실행의 회귀를 PASS/FAIL 로 낸다.

왜 필요한가: TensorRT 의 이득은 대부분 정밀도 강등(FP16/INT8)에서 나오는데,
그게 정확히 이 파이프라인의 급소를 친다. 2026-08-10 실측 — 입력을 560px 로
낮추자 추론은 2.2배 빨라졌지만 스마트폰(154mm, 화면상 42~76px)이 통째로
사라졌다. 그래서 판정해야 하는 것은 "몇 배 빨라졌나"가 아니라
**"속도 대 소물체 생존의 교환비"**다.

두 층을 분리해서 본다 — 한 숫자로 뭉개면 원인을 못 짚는다:
  A. 검출 레벨 — 물체가 살아 있는가 (소물체 소실은 여기서만 잡힌다)
  B. 기하 레벨 — 마스크 경계가 흔들려 OBB 로 번졌는가

사용:
  python3 scripts/check_accuracy.py --ref <기준CSV또는디렉토리> --cand <후보>
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

# ── 판정 상수 ─────────────────────────────────────────────
# 이 파이프라인은 같은 bag 에 대해 결정적이다 — ab/test2_free 와 v2/test2_free 가
# 통계까지 완전히 일치했다(2026-08-21 확인). 즉 실행 간 잡음 바닥이 0 이라
# 반복 실행으로는 문턱을 못 잡는다. 그래서 문턱의 기준은 "이 파이프라인이
# 이미 갖고 있는 프레임간 지터(σ)"로 둔다 — 그보다 작은 차이는 회귀라고
# 부를 근거가 없고, 그보다 크면 강등이 새로 만든 차이다.
#
# ⚠ 이 "잡음 바닥 0" 은 test2_free 에서만 확인됐다 — **무구속 경로라
# fit_plane 을 안 탄다.** 평면 구속 경로(use_belt_plane, 현재 기본 켜짐)에서는
# 0 이 아니다: geometry.fit_plane 의 segment_plane(dist_thresh, 3, 200) 이
# 시드 없는 RANSAC 이고, pipeline 이 그 평면을 실행당 한 번만 맞춰 영구
# 캐시하므로(_plane_tried) 평면 오차가 실행 간에 흔들리면서 프레임간 σ 에는
# 안 잡힌다. **실행 간 평면 변동은 씬마다 자릿수가 다르다** —
# Isaac ±0.40mm / test4 ±2.98mm / test2 ±13mm 급(repeats=21 실측).
# 아래 ABS_FLOOR_MM=2.0 은 **Isaac 만 덮는다** — test2·test4 에는 부족하다.
# **ABS_FLOOR_MM 을 낮추려면 평면 변동을 먼저 재서 그 아래로 내려가지 않게 할 것**
# (같은 조건을 여러 번 돌려 분포를 본다). 근거: docs/README.md §5 "std 는
# 재현성이 아니다", docs/belt_plane_2026-08-21.md.
SIGMA_K = 3.0        # 기준 실행 σ 의 몇 배까지 허용할지. depth_intrusion 이
                     # χ²(0.999,1)=3.29σ 를 쓰는 것과 같은 자릿수다.
# 절대 하한. σ 가 0.2mm 급인 축에서 3σ 를 그대로 쓰면 물리적으로 무의미한
# 차이까지 FAIL 이 된다. fusion.R_POS_FLOOR(2.0mm, "test2 raw 실측 프레임간
# 잔차 중앙값 수준")를 그대로 빌려 쓴다 — 같은 bag 에서 잰 값이다.
ABS_FLOOR_MM = 2.0
# 커버리지 하한. 정지 씬 실측에서 세 실행 모두 라벨당 정확히 230/230 이었다
# (ab/test2_free, v2/test2_free, ab/test2_plane). 즉 편차가 0 이라 5% 는
# 키프레임 경계 한둘이 어긋나는 정도만 허용하는 폭이고, 실제로 막아야 할
# 회귀(560px 사건)는 100% 소실이라 이 문턱 근처에 걸릴 일이 없다.
COVERAGE_MIN = 0.95
# 지터 증가 배율 상한. 평면 구속(실제 변경)이 test2 의 std 를 3~5배 **낮춘**
# 크기의 변화였으므로, 같은 규모의 상승을 잡도록 2.0 으로 둔다. 단 σ 자체가
# ABS_FLOOR_MM 미만이면 배율은 보지 않는다(0.45 -> 0.95mm 를 FAIL 로 만들지
# 않기 위해).
JITTER_MAX = 2.0
# 소물체 관문 밴드(px). 2026-08-10 에 소실된 스마트폰의 겉보기 크기다 —
# 정렬 extent [153.7, 75.8]mm @ z=1.243m 를 역산하면 fx 615~689 에서 정확히
# 이 밴드가 나온다. D455 1280x720 컬러의 fx 가 약 640 이라 앞뒤가 맞는다.
SMALL_PX_MAX = 76.0
DEFAULT_FX = 640.0   # D455 1280x720 컬러 공칭. --fx 로 덮어쓸 것.


def resolve_csv(path):
    """CSV 경로 또는 CSV 가 든 디렉토리 → 실제 CSV 경로.

    `_raw.csv` 제외 규칙의 단일 정의 — 이 규칙이 두 곳에 있으면 한쪽만 바뀔 때
    도구가 조용히 다른 파일을 기준으로 잡는다.
    """
    if not os.path.isdir(path):
        return path
    cands = [p for p in glob.glob(os.path.join(path, "*.csv"))
             if not p.endswith("_raw.csv")]
    if not cands:
        sys.exit(f"CSV 를 찾을 수 없다: {path}")
    return cands[0]


def load(path):
    """CSV 또는 CSV 가 든 디렉토리 → {label: {stamp: row}}. 라벨당 최장 트랙만.

    라벨당 트랙이 여럿일 수 있어(test5 gray notebook 2개) 최장 트랙을 그
    라벨의 대표로 본다 — compare 계열 스크립트와 같은 관용구다.
    """
    path = resolve_csv(path)
    per_track = defaultdict(list)
    for r in csv.DictReader(open(path)):
        per_track[(r["label"], r["track_id"])].append(r)
    out = {}
    for (label, _), rows in per_track.items():
        if label not in out or len(rows) > len(out[label]):
            out[label] = rows
    return {k: {float(r["stamp"]): r for r in v} for k, v in out.items()}, path


def vec(row, keys):
    return np.array([float(row[k]) for k in keys])


def sorted_extent(row):
    """정렬 extent(내림차순, m). 축 배정이 실행마다 바뀌므로 그대로 비교하면
    안 된다 — 크기는 정렬해서 봐야 같은 물리량끼리 붙는다."""
    return np.sort(vec(row, ("w", "d", "h")))[::-1]


def apparent_px(rows, fx):
    """겉보기 최단축 크기(px) — 소물체 관문에 들어가는지 판정용."""
    ext = np.array([sorted_extent(r) for r in rows.values()])
    z = np.array([float(r["z"]) for r in rows.values()])
    # 정렬 extent 의 2번째 = 중간축. 두께 < 폭 인 납작한 물체에서는 이것이
    # 관측 평면의 단축이지만, **두께 > 폭 인 물체에서는 두께다** — isaac 블록
    # [195, 58, 47] 이 그 경우로, 여기서는 47(폭)이 아니라 58(두께)을 돌려준다.
    # 소물체 관문 판정에 쓸 때 그 물체가 어느 쪽인지 확인할 것.
    return float(np.median(ext[:, 1] / z)) * fx


def compare_label(ref, cand):
    """공통 프레임에서 쌍으로 비교. 스탬프는 같은 bag 이라 정확히 일치한다
    (2026-08-21 확인: test2 230/230 완전 일치). 중앙값 비교보다 훨씬 예민하다."""
    common = sorted(set(ref) & set(cand))
    if not common:
        return None
    dc = np.array([vec(cand[s], ("x", "y", "z")) - vec(ref[s], ("x", "y", "z"))
                   for s in common]) * 1000
    de = np.array([sorted_extent(cand[s]) - sorted_extent(ref[s])
                   for s in common]) * 1000
    ref_c = np.array([vec(ref[s], ("x", "y", "z")) for s in common]) * 1000
    ref_e = np.array([sorted_extent(ref[s]) for s in common]) * 1000
    cand_c = np.array([vec(cand[s], ("x", "y", "z")) for s in common]) * 1000
    cand_e = np.array([sorted_extent(cand[s]) for s in common]) * 1000
    return dict(n=len(common),
                d_center=np.abs(np.median(dc, axis=0)),
                d_extent=np.abs(np.median(de, axis=0)),
                ref_cstd=ref_c.std(axis=0), cand_cstd=cand_c.std(axis=0),
                ref_estd=ref_e.std(axis=0), cand_estd=cand_e.std(axis=0))


def gate(sigma):
    """축별 허용 폭(mm) — 기준 σ 의 SIGMA_K 배, 단 ABS_FLOOR_MM 아래로는 안 간다."""
    return np.maximum(SIGMA_K * sigma, ABS_FLOOR_MM)


def jitter_fail(ref_std, cand_std):
    """지터가 유의하게 늘었는가. σ 가 물리 바닥 미만이면 배율은 의미 없다."""
    return bool(np.any((cand_std > ABS_FLOOR_MM)
                       & (cand_std > JITTER_MAX * np.maximum(ref_std, 1e-9))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="기준 실행(현행 bf16)")
    ap.add_argument("--cand", required=True, help="후보 실행(FP16/INT8 등)")
    ap.add_argument("--fx", type=float, default=DEFAULT_FX,
                    help=f"컬러 fx(px). 소물체 밴드 판정용, 기본 {DEFAULT_FX}")
    args = ap.parse_args()

    ref, ref_path = load(args.ref)
    cand, cand_path = load(args.cand)
    print(f"기준   {ref_path}\n후보   {cand_path}\n")

    fails, small_seen = [], False
    print(f"{'라벨':13s} {'겉보기':>7s} {'커버리지':>13s} "
          f"{'|Δ중심| xyz mm':>22s} {'|Δ크기| mm':>22s} {'지터':>6s}")
    for label in sorted(ref):
        rows_ref = ref[label]
        px = apparent_px(rows_ref, args.fx)
        small = px <= SMALL_PX_MAX
        small_seen |= small
        n_ref = len(rows_ref)
        rows_cand = cand.get(label, {})
        n_cand = len(rows_cand)
        cov = n_cand / n_ref if n_ref else 0.0
        tag = "소물체" if small else ""

        if n_cand == 0:
            # 560px 사건의 형태 — 물체가 통째로 사라진다
            print(f"{label:13s} {px:6.0f}p {n_cand:5d}/{n_ref:<5d}      "
                  f"{'— 소실 —':>22s}")
            fails.append(f"{label}: 완전 소실 (기준 {n_ref}프레임)"
                         + (" [소물체]" if small else ""))
            continue

        c = compare_label(rows_ref, rows_cand)
        g_c, g_e = gate(c["ref_cstd"]), gate(c["ref_estd"])
        bad_c = np.any(c["d_center"] > g_c)
        bad_e = np.any(c["d_extent"] > g_e)
        bad_j = (jitter_fail(c["ref_cstd"], c["cand_cstd"])
                 or jitter_fail(c["ref_estd"], c["cand_estd"]))
        mark = lambda b: "!" if b else " "  # noqa: E731
        print(f"{label:13s} {px:6.0f}p {n_cand:5d}/{n_ref:<5d} "
              f"{cov:5.2f}{mark(cov < COVERAGE_MIN)} "
              f"{np.array2string(c['d_center'], precision=2, floatmode='fixed'):>20s}"
              f"{mark(bad_c)} "
              f"{np.array2string(c['d_extent'], precision=2, floatmode='fixed'):>20s}"
              f"{mark(bad_e)} {mark(bad_j):>5s} {tag}")

        if cov < COVERAGE_MIN:
            fails.append(f"{label}: 커버리지 {cov:.2f} < {COVERAGE_MIN}"
                         f" ({n_cand}/{n_ref})" + (" [소물체]" if small else ""))
        if bad_c:
            fails.append(f"{label}: 중심 이동 {np.round(c['d_center'], 2)}mm "
                         f"> 허용 {np.round(g_c, 2)}mm")
        if bad_e:
            fails.append(f"{label}: 크기 변화 {np.round(c['d_extent'], 2)}mm "
                         f"> 허용 {np.round(g_e, 2)}mm")
        if bad_j:
            fails.append(f"{label}: 지터 증가 (중심 "
                         f"{np.round(c['ref_cstd'], 2)}→"
                         f"{np.round(c['cand_cstd'], 2)}, 크기 "
                         f"{np.round(c['ref_estd'], 2)}→"
                         f"{np.round(c['cand_estd'], 2)} mm)")

    for label in sorted(set(cand) - set(ref)):
        print(f"{label:13s} {'':>7s} {len(cand[label]):5d}/{0:<5d}   (기준에 없음)")

    print()
    if not small_seen:
        # 소물체가 없는 검증셋으로 PASS 를 받으면 관문을 안 지난 것이다
        print(f"⚠ 경고: {SMALL_PX_MAX:.0f}px 이하 물체가 검증셋에 없다 — "
              f"소물체 관문을 통과했다고 말할 수 없다 (test2 의 smartphone 을 쓸 것)")
    if fails:
        print(f"FAIL ({len(fails)}건)")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS — 기준 대비 회귀 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
