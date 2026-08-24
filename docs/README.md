# docs 색인

이 폴더의 문서와, **각 문서가 남긴 열린 항목**을 한 곳에 모은다.

> 만든 이유: 2026-08-24 에 `datasets.md` 가 이미 적어둔
> *"weak-prompt 는 detection-threshold 문제지 occlusion logic 이 아니다 —
> 프롬프트 교체가 먼저"* 를 못 보고 다른 데를 팠다. **기록은 있는데 안 읽혔다.**
> 새 문서를 쓰면 아래 표에 한 줄 추가할 것.

---

## 1. 문서 목록

### 시뮬 연동 (Isaac Sim)
| 문서 | 요지 |
|---|---|
| [bridge_contract.md](bridge_contract.md) | Isaac→ROS2 브리지 계약. **입력이 없을 때 확인 순서가 여기 있다** — 퍼블리셔 수부터 볼 것(§3.3~3.4) |
| [isaac_sim_connection_2026-08-13.md](isaac_sim_connection_2026-08-13.md) | 연결 절차·환경변수·실행 방법 |
| [isaac_sim_stability_2026-08-14.md](isaac_sim_stability_2026-08-14.md) | 안정성 이슈와 VRAM 내역 |

### 기하 · 자세 추정
| 문서 | 요지 |
|---|---|
| [belt_plane_2026-08-21.md](belt_plane_2026-08-21.md) | 벨트 평면 구속 OBB. 두께 붕괴·광축 편향 해소, 시뮬 정답 대비 ±0.4mm/+0.7mm **(1008 기준)** |
| [fusion_design_2026-08.md](fusion_design_2026-08.md) | 확률 융합 필터 설계와 불변식 |
| [border_margin_2026-08-21.md](border_margin_2026-08-21.md) | 화면 밖 잘린 물체가 치수를 22% 과소보고. 원인 규명, 처방은 되돌림 |
| [truncation_unify_2026-08-24.md](truncation_unify_2026-08-24.md) | 절단 관측 처리 통일 시도 → 되돌림. **"편향은 잡음이 아니다"** |

### 추적 · 가림
| 문서 | 요지 |
|---|---|
| [footprint_gate_2026-08-21.md](footprint_gate_2026-08-21.md) | 부분 가림으로 잘린 관측 기각. 가림 중 오차 137→5mm. **비대칭 판정이 핵심** |
| [publish_gap_2026-08-24.md](publish_gap_2026-08-24.md) | 발행 공백 원인 분해. `PUB_POS_STD_MAX` 에 σ↔오차 근거표 |
| [filter_divergence_2026-08-21.md](filter_divergence_2026-08-21.md) | 필터 발산 감지 조사 → 보류 |
| [black_bag_limit_2026-08-21.md](black_bag_limit_2026-08-21.md) | black bag 오염 41% 는 **센서 한계**. 코드로 못 고친다 |

### 성능 · 파라미터
| 문서 | 요지 |
|---|---|
| [image_size_2026-08-21.md](image_size_2026-08-21.md) | 672 의 대가 — 소물체 소실 / 폭 −3mm / 그러나 속도는 우세. ④ 는 **오측 정정** |
| [detect_interval_2026-08-21.md](detect_interval_2026-08-21.md) | di 5→10 이 1.7~1.9배. 무텍스처 물체에서 손해라 기본 5 유지 |
| [tensorrt_deploy_2026-08-21.md](tensorrt_deploy_2026-08-21.md) | TensorRT 배포 영향과 철수 기준 → **보류** |

### 데이터셋
| 문서 | 요지 |
|---|---|
| [datasets.md](datasets.md) | bag 목록·실측 사실·**권장 프롬프트**·필요한 데이터 |

### 의사결정
| 문서 | 요지 |
|---|---|
| `open_decisions_2026-09.md` | 로봇(그리퍼) 요구사항이 결정할 항목들 — *별도 작성 중* |

---

## 2. 결정 기록 — 지금 켜져 있는 것과 그 근거

| 설정 | 값 | 근거 |
|---|---|---|
| `use_belt_plane` | **켜짐** | [belt_plane](belt_plane_2026-08-21.md) "재평가" 절 · `ec4fc69`. 오전 판단(꺼짐)이 오후에 뒤집혔다 |
| `enable_footprint_gate` | **켜짐** | [footprint_gate](footprint_gate_2026-08-21.md) "비대칭 판정" 절 · `5a4d173` |
| `enable_merge` | **켜짐** | `5d687bf`. 실기는 `max_per_label==1` 조기 반환으로 구조적 no-op |
| `detect_interval` | **5** (Isaac 은 1) | [detect_interval](detect_interval_2026-08-21.md) |
| `image_size` | 기본 1008, **Isaac 672** | [image_size](image_size_2026-08-21.md) ③ — 근거는 VRAM 이 아니라 **속도**(4.60 vs 2.70 Hz) |
| `MAX_THICKNESS` | 0.35m | `aaf9993`. 비율 규칙은 세운 원통에서 깨진다 |
| `LE_REJECT_STREAK` | 3 | `a65e273`. extent 기각 탈출을 "유한" 이 아니라 **상계**로 |
| `pub_score_min` | 기본 0, **Isaac 0.6** | [datasets](datasets.md) 권장 프롬프트 절 — 어휘가 약하면 이 게이트에 물체가 통째로 날아간다 |
| TensorRT | **보류** | [tensorrt_deploy](tensorrt_deploy_2026-08-21.md) — vision 이 SAM3 의 57% 라 체감 상한 1.2배 |

**권장 프롬프트**([datasets.md](datasets.md)): `gray notebook`→`beige notebook`,
`book`→`manual`, `smartphone`→`cell phone`. 실물과 이름을 맞추는 것이 핵심이다.

---

## 3. 되돌린 것 — **같은 걸 다시 시도하지 않기 위해**

| 시도 | 왜 뺐나 | 출처 |
|---|---|---|
| **게이트 측정량을 면적→점 개수** | raw 시뮬레이션은 전 물체 개선인데 **실기에서 test4 black bag 이 41→49% 악화**. 시뮬레이션이 실기 회귀를 대신하지 못한다 | [black_bag_limit](black_bag_limit_2026-08-21.md) |
| **경계 판정 마진 8px** | 시드는 실제로 차단됐고 실기 오탐도 없었으나 **길이가 154mm 로 그대로**, 쓰레기 트랙 548→906 증가 | [border_margin](border_margin_2026-08-21.md) |
| **절단 관측 처리 통일**(`border` 도 폐기) | 이론은 옳으나 컨베이어는 절단이 상시라 **정상 트랙 #6 이 −84%**. 실기는 무영향 | [truncation_unify](truncation_unify_2026-08-24.md) |
| **필터 발산 감지** | 발산·가림·가속지연이 **같은 서명**이라 P 팽창이 가리개 관측을 수락시킨다. 게다가 풋프린트 게이트가 주 경로를 이미 막아 이득 소멸 | [filter_divergence](filter_divergence_2026-08-21.md) |
| **대칭 풋프린트 판정**(`\|dev\| > TAU`) | 정상 프레임을 통째로 버려 세장 물체가 4→39% 악화. **부호를 봐야 한다**(`dev < -TAU`) | [footprint_gate](footprint_gate_2026-08-21.md) |
| **연결성분 depth 클립(G안)** | Isaac 에선 이기는데 실기에서 중심 흔들림 5배 | 2026-08-20 devlog |
| **`torch.compile`** | 7% 이득에 워밍업 30초 | 2026-08-10 |

> 공통 교훈: **시뮬·합성·raw 결과로 채택을 결정하지 말 것.** 위 7건 중 3건이
> 그 함정이다. 실기 회귀를 통과하기 전까지는 "후보" 로만 부른다.

### 3-1. 측정 아티팩트로 판명된 "발견" — **계측 자체를 먼저 의심할 것**

되돌린 것과 다른 범주다. 코드는 멀쩡했고 **측정이 틀렸다.** 둘 다 그 잘못된
숫자로 원인을 지목하고 처방까지 세웠다가 뒤집혔다.

| 주장했던 "발견" | 실제 | 출처 |
|---|---|---|
| **"트랙 미매칭 0프레임 → 검출은 멀쩡, 문제는 파이프라인 내부"** | `matched` 를 `_with_frozen()` 로 셌다. frozen 트랙이 포함돼 미매칭이 0 으로 보였다. 실제 검출 실패는 book 22%, gray notebook 22% | [publish_gap](publish_gap_2026-08-24.md) |
| **"672 는 큰 물체 두께를 18% 과소 추정"** | 46.8mm 는 두께가 아니라 **폭**이었다. 두 실행에 서로 다른 축 식별 규칙을 적용했다. 두께는 672/1008 이 57.8/57.7mm 로 동일 | [image_size](image_size_2026-08-21.md) ④ |

> 공통 교훈: **비교하기 전에 두 쪽이 같은 양을 재고 있는지 확인할 것.**
> 특히 OBB extent 는 `match_axes()` 가 축 순서를 **실행마다** 바꾼다 — 같은
> bag·같은 해상도에서 실행 길이만 달라도 `re1`/`re2` 배정이 뒤집힌다. 컬럼
> 인덱스로 대응시키면 안 되고, 축의 의미로 대응시켜야 한다: 두께 = `re3`
> (= 평면 법선 방향 `thk`, 이건 안 뒤집힌다), 길이·폭 = `max`/`min(re1, re2)`.
>
> 그리고 **정답이 있는 축과 없는 축을 구분할 것.** Isaac 정답은 벨트 z·상면
> z·두께·중심뿐이고 **폭·길이의 정답은 없다.** 정답 없는 축의 차이는 "오차" 가
> 아니라 "두 설정의 차이" 로만 말할 수 있다.

---

## 4. 열린 항목 (2026-08-24 기준)

### 로봇 요구사항이 있어야 정해지는 것 — `open_decisions_2026-09.md` 참고
| 항목 | 무엇이 걸려 있나 | 출처 |
|---|---|---|
| `PUB_POS_STD_MAX` (현 0.02) | σ↔오차 표는 있다. 커버리지 40~47% 를 잃고 오염을 1/3 로 줄이는 교환 | [publish_gap](publish_gap_2026-08-24.md) |
| 경계 물체 발행 정책 | 3갈래 — 치수만 불신 / `covariance` 로 알림 / 발행 안 함 | [truncation_unify](truncation_unify_2026-08-24.md), [border_margin](border_margin_2026-08-21.md) |
| `image_size` 672 vs 1008 | 속도(4.60 vs 2.70 Hz) ↔ **폭** 3.0mm 차이(정답 없음). 두께는 무관 | [image_size](image_size_2026-08-21.md) ③④ |

### 측정하면 답이 나오는 것
| 항목 | 출처 |
|---|---|
| test2 면내 크기 차이의 정오 — **실물 줄자 실측 필요**(책·노트북·보온병) | [belt_plane](belt_plane_2026-08-21.md) |
| 896~1008 사이 미측정. 896 이 진짜 소물체 하한인지 | [image_size](image_size_2026-08-21.md) |
| patch 정렬이 800 급락은 설명하나 전 구간은 아님 (812=0.67 < 784=0.81) | [image_size](image_size_2026-08-21.md) |
| isaac 블록(33px)이 672 에서 살아남는 이유 — 픽셀 크기만으로 예측 불가 | [image_size](image_size_2026-08-21.md) |
| `TAU`·`ALPHA` 는 test4 book 한 사례의 고원값. 라벨된 오염 구간이 그것뿐 | [footprint_gate](footprint_gate_2026-08-21.md) |

### 별건으로 남긴 것
| 항목 | 상태 | 출처 |
|---|---|---|
| **test4 book 가림 오염 27%** | 흐름 게이트로 못 고침. 다음 후보는 외형(색 히스토그램) 채널 | [belt_plane](belt_plane_2026-08-21.md) |
| **black bag 41%** | **센서 한계로 수용.** 개선하려면 조명·노출·재질 | [black_bag_limit](black_bag_limit_2026-08-21.md) |
| 필터 회복 국면 — 틀린 상태에서 나오는 데 2.5초 | 발산 감지는 보류 | [footprint_gate](footprint_gate_2026-08-21.md), [filter_divergence](filter_divergence_2026-08-21.md) |
| 규약 전이 재시드가 tilt 게이트 진동 시 영구 미승격 유발 가능 | 현행 데이터로는 안 드러남(폴백 2/459) | [belt_plane](belt_plane_2026-08-21.md) |
| 등속 씬에서 CV 모델이 벨트 가감속에 뒤처짐(`SIGMA_A`) | 지금 문제 아님 | [filter_divergence](filter_divergence_2026-08-21.md) |
| TensorRT | 엣지 타겟 확정 시 재개 | [tensorrt_deploy](tensorrt_deploy_2026-08-21.md) |

### 데이터 대기
[datasets.md](datasets.md) "Data still needed" — 전동 컨베이어 등속 bag,
이동 중 가림 bag, ArUco 씬. *(ArUco 는 Isaac 정답 좌표로 대체·종료됨)*

---

## 5. 읽기 전 주의

- **수치를 인용할 때 조건을 같이 적을 것.** 같은 양이 조건에 따라 크게 다르다:
  - 키프레임 시간 — `image_size` 없이는 무의미(672 vs 1008 이 2.25배)
  - 치수 정확도 — **어느 축인지 같이 적을 것.** 두께는 해상도와 무관(672/1008 모두
    정답 +0.5mm)이지만 **폭은 두 해상도가 3.0mm 벌어진다**(폭 정답은 없다). 그리고 raw CSV 의 `re1`/`re2`
    배정은 `match_axes` 가 실행마다 바꾼다 — 컬럼 인덱스로 비교하면 안 된다
    (이걸 놓쳐 "두께 −18%" 오측이 한 번 올라갔다: [image_size](image_size_2026-08-21.md) ④)
  - 재개 지연 — `datasets.md` 의 "최대 3.2초" 는 **완전 가림 후 재개**이고,
    [publish_gap](publish_gap_2026-08-24.md) 의 13.4초는 **부분 가림 지속 시간**이다. 다른 양이다
- **오프라인 러너 FPS 는 성능 지표가 아니다** — mp4 인코딩·bag 디코딩이 28~30%.
  파이프라인 내부는 CSV 의 `proc_ms` 로 볼 것
- [belt_plane](belt_plane_2026-08-21.md) 의 "켜기 전에 필요한 검증" 절은
  **오전 판단 시점의 기록**이다. 같은 문서 "재평가" 절이 그것을 뒤집었다
