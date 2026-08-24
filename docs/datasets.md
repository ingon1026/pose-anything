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

**`isaac.launch.py` 는 이미 `pub_score_min=0.6` 을 쓴다.** Isaac 씬에 약한 물체가
없어서 안 드러났을 뿐이고, 실기 프리셋에 그 값을 켜는 순간 약한 프롬프트 물체가
**에러도 경고도 없이** 날아간다. 어휘 강화의 목적은 커버리지가 아니라 **그
안전장치를 쓸 수 있게 만드는 것**이다.

### 부작용 없음 (실측)
- **기하 무회귀** — 중심 Δ 0.1~0.6mm, 크기 Δ 0.1~1.1mm. 2026-08-20 Isaac 실측
  ("어휘는 score 만 바꾸고 기하는 1mm 도 안 바뀐다")이 재현됐다
- **트랙 수 불변** — 라벨당 1개 유지. 조각·오검출 증가 없음
- **대조군 완전 무변화** — keyboard·black bag 은 발행 프레임·score 가 1도 안 변했다
- 커버리지는 거의 같다(notebook 753→731, book 568→572)

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
| Motorized conveyor + occlusion during motion | moving-occlusion handling (Kalman-gated re-matching) |
| Scene with printed ArUco markers | absolute pose error measurement |
