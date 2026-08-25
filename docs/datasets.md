# Dataset inventory (rosbags)

Bags are **not** committed to this repo (see `.gitignore`) — they live on the
dev machine at `~/roboworld/bags/` with tar backups on the Windows desktop
(`bags.tar.gz`, `bags2.tar.gz`). **Do not delete** — besides their original
test purpose, all bags are reserved as seed data for the planned
SAM3-auto-label → lightweight-YOLO distillation pipeline.

| Bag | Recorded | Scene | Camera | Used for | Distillation value |
|---|---|---|---|---|---|
| `test2` | 2026-08-07 | static conveyor, 4+ objects (thermos, laptop, book, smartphone) | 1280×720 top-down | detection quality, size accuracy (±1 cm), prompt tuning (water bottle → thermos) | multi-object variety |
| `test3` | 2026-08-07 | hand-pushed rollers, 3 moving objects (book, glove, pink block) | 1280×720 top-down | tracking persistence, pose stability, hybrid-tracking validation. NOT usable for constant-velocity work (speed varies 40–137 mm/s) | motion blur / moving scenes |
| `test4` | 2026-08-11 | static conveyor, 3 objects (black bag, keyboard, book), hand+black folder occlusions | 640×480, closer mount | occlusion signal measurement (score collapse, depth intrusion up to 554 mm), occlusion-handling development, reappearance latency (median 200 ms) | occlusion-hard examples |
| `test5` | 2026-08-11 | same as test4 + gray notebook (4 objects), 68 s | 640×480 | held-out validation set for occlusion handling (params tuned on test4 only) | occlusion-hard examples, longer sequence |

## Measured facts derived from these bags

- Occlusion: full occlusion = detection loss (max 2.6 s); partial occlusion =
  score below 50 % of the track's own baseline; occluder depth intrusion
  554 mm at 918 mm nominal → 20 % depth gate
- Reappearance-to-pose-resume latency (27 occlusion events, test4+test5,
  well-detected objects): median ~0.3 s, 90 % within 1.7 s, max 3.2 s for
  gradually-clearing occlusions. Lower bound is the 5-frame SAM keyframe
  period (333 ms at 15 fps). Caveat: weak-prompt objects ("gray notebook",
  score ≈ 0.46 avg) show multi-second resume gaps that are a detection
  -threshold issue, not occlusion logic — fix the prompt word first
- EMA position lag at this belt speed (46 mm/s): 2.5–7 mm → Kalman deferred
- 560 px input loses smartphone-sized objects entirely (test2)

## 권장 프롬프트 (2026-08-24 실측)

test5 의 라벨 이름 두 개가 실물과 어긋나 있었다. **어휘를 실물에 맞추면 score 가
크게 오른다** — 이름이 정확한 물체(black bag 0.97, keyboard 0.98)와 부정확한
물체(book 0.72, gray notebook 0.28)의 차이가 그것이다.

| 현행 | **권장** | score 중앙 | p10 | `pub_score_min=0.6` 생존 |
|---|---|---|---|---|
| `gray notebook` | **`beige notebook`** | 0.28 → **0.85** | 0.14 → 0.59 | **6% → 89%** |
| `book` | **`manual`** | 0.72 → **0.91** | 0.63 → 0.89 | 94% → 97% |
| `smartphone`(test2) | **`cell phone`** | 0.82 → **0.93** | 0.71 → 0.92 | 98% → 99% |
| `keyboard` | (유지) | 0.97 | 0.95 | 99% |
| `black bag` | (유지) | 0.93 | 0.89 | 99% |

실물 확인 결과 이름이 틀렸던 것이다 — **`gray notebook` 은 회색이 아니라
베이지/크림 무지 공책**(랩톱 아님, 12.8×18.1×3.4cm)이고, **`book` 은 두꺼운
책이 아니라 흰 표지에 만화 일러스트가 인쇄된 얇은 소책자**다.

### 왜 중요한가 — 안전장치를 켤 수 있느냐의 문제
`gray notebook` 의 발행 score 중앙값이 **0.28 로 threshold(0.4)보다도 낮았다.**
그 트랙은 정상 검출이 아니라 **저점수 2차 매칭(`assoc_threshold=0.1`)에 얹혀서**
살아 있었다. 그래서 `pub_score_min` 을 켜면 조용히 사라진다:

```
pub_score_min      0.0     0.4     0.5     0.6
gray notebook      753     224     150      45   ← 94% 소멸
```

~~`isaac.launch.py` 는 이미 `pub_score_min=0.6` 을 쓴다.~~ **(2026-08-24 정정: 프리셋에서 제거했다. base 기본값 0.0 과 같아 다시 적으면 무동작이라서다 — Isaac 도 이제 0.0 으로 돈다.)** Isaac 씬에 약한 물체가
없어서 안 드러났을 뿐이고, 실기 프리셋에 그 값을 켜는 순간 약한 프롬프트 물체가
**에러도 경고도 없이** 날아간다. 어휘 강화의 목적은 커버리지가 아니라 **그
안전장치를 쓸 수 있게 만드는 것**이다.

### 부작용 없음 (실측)
- **기하 무회귀** — 중심 Δ 0.1~0.6mm, 크기 Δ 0.1~1.1mm. 2026-08-20 Isaac 실측
  ("어휘는 score 만 바꾸고 기하는 1mm 도 안 바뀐다")이 재현됐다
- **트랙 수 불변** — 라벨당 1개 유지. 조각·오검출 증가 없음
- **대조군 완전 무변화** — keyboard·black bag 은 발행 프레임·score 가 1도 안 변했다
- 커버리지는 거의 같다(notebook 753→731, book 568→572)

### test4 에서도 재현된다 (단일 bag 결론 아님)
`book -> manual` 을 test4 에서도 확인했다 — score 0.74 → **0.92**, p10 0.68 →
0.84, 발행 332 → 333. black bag·keyboard 는 발행 프레임·score·트랙 수가
**1도 안 변했다**. 어휘 이득이 bag 을 가리지 않는다.

(test4 의 `book` 은 원래 0.74/97% 라 `pub_score_min` 위험 구간이 아니었다 —
test5 의 `gray notebook` 처럼 극적인 개선이 필요한 상태는 아니지만, 같은 방향의
이득이 나온다는 것이 확인됐다. `gray notebook` 은 test4 에 없다.)

### ⚠ 어휘를 바꾸기 전에 — `publish_score_min` 과의 불변식 (2026-08-24)

**어휘를 고르면 게이트 문턱이 따라 움직인다.** 이 절만 보고 어휘를 바꾸면
`publish_score_min` 이 그 어휘의 최저 score 위로 올라가 **정상 물체가 에러 없이
잘린다.** 경고문은 `Track.publishable` docstring 에 이미 있었지만 그것만으로는
막지 못했다 — **경고는 비교할 숫자를 주지 않기 때문**이다. 숫자로 박아둔다.

> **불변식**: `publish_score_min` < (그 어휘의 정상 크기 트랙 per-track 최저 score)
>
> 채택 어휘 **`"blue bar with holes"`** 의 실측값 (게이트 off CSV, 2026-08-20):
>
> | 실행 | 실블록 전체(ext ≥ 0.15) | 경계 절단 T8 제외 |
> |---|---|---|
> | `output/belt_moving` | **0.102** | 0.135 |
> | `output/belt_merge` (현행 프리셋 = merge on) | **0.112** | 0.420 |
>
> **어휘를 바꾸면 게이트 off(`--publish-score-min 0.0`)로 재측정해 이 표를
> 갱신하고, `publish_score_min` 이 그 값보다 낮은지 확인할 것.**

**당시 `isaac.launch.py` 의 `0.6` 이 이 불변식을 위반했고, 그래서 껐다(2026-08-24). 아래 불변식은 게이트를 다시 켤 때를 위해 유지한다.** `belt_merge` 기준
0.112 여야 하는데 0.6 이다. 실제 손익(게이트를 0.6 으로 적용했을 때):

| | 잘리는 프레임 |
|---|---|
| **실블록** T6 25 + T8 60 + T12 잔여 5 | **90** |
| **조각**(T11, ext 0.073) | **0** |

즉 **조각을 하나도 못 막으면서 정상 블록 85프레임 + 중복 잔여 5프레임을 자른다.**

#### 왜 이렇게 됐나 — 근거 측정은 유효했고, 그 뒤 세상이 바뀌었다
`isaac.launch.py` 의 `0.6` 은 **2026-08-19 43사이클 측정에 근거했고 그 시점엔
옳았다** — 정상 0.840~0.945 대 조각 0.416~0.432 로 두 무리가 깨끗이 갈렸다.
그 뒤 **조각을 실제로 없앤 것은 게이트가 아니라 `enable_merge=true` 와 조각
수정**이었고, 그 결과 **남은 저크기 트랙이 고득점이 되었다**(T11 = 0.645).
동시에 정상 블록의 꼬리가 0.102 까지 내려왔다.

→ **score 는 더 이상 분리 변수가 아니다. 크기가 분리 변수다.** 같은 주석의
자기 데이터가 이미 그렇게 말한다 — 조각 extent 0.0447~0.0498 대 정상
0.152~0.198, **겹침 없음.** 지금 실측도 같다(T11 0.073 대 실블록 0.157~0.204).

**유효한 score 문턱이 존재하지 않는다**: 정상이 0.102 까지 내려가는데 조각은
0.645 아래로 안 내려간다. **0.102 위 어떤 값도 정상을 자르고, 0.645 아래 어떤
값도 조각을 통과시킨다.** 0.6 은 둘 다 하고 있다.

#### 0.6 은 조각이 아니라 **어휘 선택지**를 좁혀왔다
게이트에 닿거나 아래인 후보들 — 시험한 다섯 개가 전부 여기 걸린다:

| 후보 | 값 | 0.6 대비 |
|---|---|---|
| `blue plastic beam` | 최저 0.60 | 게이트에 **닿음** (`isaac.launch.py` 주석) |
| `blue plastic blocks` | 게이트 아래 | **발행 통째로 멈춤** (〃) |
| `booklet` | p10 0.28 | 아래 |
| `notepad` | 0.50 | 아래 |
| `textbook` | 0.24 | 아래 |
| **`beige notebook`**(채택 권장) | **p10 0.59** | **아래** ← 권장 어휘조차 걸린다 |

**권장 어휘로 바꿔도 p10 이 0.59 라 게이트 아래다.** 위 "권장 프롬프트" 표의
`pub_score_min=0.6` 생존 89% 가 그 뜻이다 — 100% 가 아니다. 즉
**어휘 강화로 이 게이트를 안전하게 만들 수 없다.** 선행조건이 아니라 상충이다.

#### 한계
CSV 증거는 **동일 강체 블록 9개짜리 Isaac 씬 하나**다. 실기 전반을 단정하면
안 된다. 다만 실기 권고는 **"현상 유지(기본값 0.0)"** 라 위험이 없다.

#### 측정할 때 — 게이트를 켜고 뽑은 CSV 로는 게이트를 정할 수 없다
`scripts/run_offline.py:249` 가 `if not o.publishable: continue` 로 거른다.
**게이트에 걸린 행이 CSV 에 아예 안 남으므로 질문이 사는 자리가 잘려 나간다.**
반드시 `--publish-score-min 0.0` 으로 뽑고 `score` 컬럼으로 오프라인 재적용할 것.

### 어휘가 해상도 하한도 낮춘다
`smartphone` → `cell phone` 은 점수만 올리는 게 아니다. **`image_size=672` 에서
`smartphone` 은 검출 0 인데 `cell phone` 은 0.77 로 살아남는다** — 8/21 에
"안전 하한 896" 이라 한 값이 사실 어휘에 딸려 있었다는 뜻이다. 상세는
`docs/image_size_2026-08-21.md` 의 2026-08-24 추가 실측.

### 위험 후보 — 그럴듯한 동의어가 죽는다
`tumbler` 0.00(2026-08-07) 과 같은 계보가 재현됐다. **어휘 실험에는 반드시
실패 후보를 넣어야 고원을 안다.**

```
diary      검출 0      magazine  검출 0
textbook   0.24        notepad   0.50
booklet    0.81(p10 0.28 — 중앙은 높지만 꼬리가 나쁘다)
backpack   0.77  ← black bag 0.96 보다 나쁘다. 어깨끈이 뚜렷해도 그렇다
```

## Data still needed (blockers for next work)

| Needed bag | Enables |
|---|---|
| Motorized conveyor, constant speed, objects passing | Kalman coasting + constant-velocity validation |
| Motorized conveyor + occlusion during motion | moving-occlusion handling (Kalman-gated re-matching)<br>**⚠ 속도 하한이 필수다** — 46mm/s(현행 벨트)로 찍으면 삭제 타임아웃(6.0s)이 위치 게이트(물체 폭×2.5)보다 항상 먼저 발화해 같은 결론이 반복될 뿐이다. **벨트 ≥140mm/s + 가림 2~3s**, 또는 60~140mm/s 면 **가림 4~6s**. 물체는 폭 ~130mm 이하 포함(게이트가 폭에 비례). 성립 조건: 가림 중 물체가 자기 폭의 2.5배 이상 이동 |
| Scene with printed ArUco markers | absolute pose error measurement |
