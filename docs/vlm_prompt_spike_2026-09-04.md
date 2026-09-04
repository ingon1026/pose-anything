# VLM ↔ SAM3 프롬프트 스파이크 (2026-09-04)

**질문:** 로컬 VLM(Qwen3-VL)이 (a) SAM3 텍스트 프롬프트를 자동으로 만들 수 있는가,
(b) SAM3 검출을 검증할 수 있는가.

**답:** **(a) 열거는 채택 후보다** — 사람이 놓친 물체를 찾아냈고, 남은 문제는 이
저장소가 이미 아는 프롬프트 단어 문제(`PROMPT_ALIASES`)뿐이다. **(b) 검증은
보류다** — 의미(semantic) 오류는 잡지만 조각·절단은 못 잡고, 작은 모델(2B)은
검증기로 못 쓴다.

측정 환경: RTX 4070 Ti 12,282 MiB(다른 서비스 1.5 GB 점유), `transformers 5.5.0`,
`Qwen/Qwen3-VL-2B-Instruct` / `Qwen/Qwen3-VL-4B-Instruct` bf16, bag 당 프레임
12장(30프레임마다 1장 샘플, `frames.py --every 30 --max 12`), 긴 변 1024px로 축소
(확대는 안 함), SAM3 검출은 `assoc_threshold=min(0.1, threshold)`
(`sam3_detector.py:83`)라 **표시 threshold 0.4 미만~0.1 이상의 저점수 검출도
포함**된다. 스파이크 코드는 저장소 밖 `~/vlm/spike/`(`enumerate.py` · `verify.py` ·
`frames.py` · `report.py` · `vlm_common.py`).

## 1. (a) 자동 프롬프트 후보 — 열거

`enumerate.py` 의 질문(원문 그대로):

> *"List every distinct physical object visible on the conveyor belt or table
> in this image. For each object, give a short English noun phrase (1 to 3
> words) suitable as a text prompt for an open-vocabulary segmentation model
> -- color or material adjectives are allowed (e.g. "black bag", "pink foam
> block"). Do not include the belt, the table/desk surface, the background,
> or human hands/arms. Output ONLY a JSON array of strings, nothing else."*

프레임 12장 중 **50% 이상**에 등장한 명사구만 후보로 채택(`--min-frac 0.5`).

| bag | 모델 | 자동 후보 전체 | 수동 프롬프트 대비 | 지연 중앙값 | 피크 VRAM | 로드 |
|---|---|---|---|---|---|---|
| test4 | 4B | black bag, keyboard, pink foam block | exact: black bag, keyboard · miss: manual · extra: pink foam block | 557 ms | 8582 MB | 18.5 s |
| test2 | 4B | black chair, black phone, blue book, green water bottle, pink foam block, red box, silver laptop, white cloth | partial: laptop↔silver laptop · miss: thermos, manual, cell phone · extra: 나머지 7개 | 1529 ms | 8667 MB | 12.1 s |
| isaac_belt_moving | 4B | blue bar, metal roller | partial: blue bar with holes↔blue bar · extra: metal roller | 294 ms | 8561 MB | 10.5 s |
| test4 | 2B | black bag, book, keyboard, pink foam block | exact: black bag, keyboard · miss: manual · extra: book, pink foam block | 474 ms | 4149 MB | 13.7 s |

**핵심 관찰:**

① **사람이 놓친 물체를 찾았다** — test4 자동 후보의 "pink foam block" 은 손 프롬프트
목록에 없었다. `~/vlm/out/test4/frames/0000.png` 로 확인: 프레임 **오른쪽 끝**에
숫자가 적힌 분홍 폼블록이 실제로 있고(SAM3 검출 box `[556, 186, 640, 218]`,
score 0.94, x1=640=프레임 폭 — 정확히 우측 가장자리에서 잘림), 예시 문구가 샌
것이 아니라 실재 검출이다.

② **이름이 SAM3 최적이 아니다** — test2 에서 자동 후보는 "green water bottle"
(score 0.54)인데 수동 프롬프트 "thermos" 는 0.91 이다. 이건 이 저장소가 이미
아는 문제다: `sam3_detector.py:8` `PROMPT_ALIASES` 가 `"물통": "thermos"` 를
매핑하며 붙인 주석이 정확히 이 수치를 남겨뒀다 — *"초록 보온병: "water bottle"은
0.3~0.45로 불안정, thermos는 ~0.9"*.

③ **프레임마다 동의어가 갈려 50% 문턱에서 탈락한다** — test4 의 "manual" 을
VLM 은 프레임마다 다르게 불렀다: `instruction manual`(4/12), `manual`(3/12),
`white paper`(3/12), `instruction sheet`(1/12), `book`/`black book`(1/12씩).
합치면 거의 매 프레임 등장하지만 어느 표현도 6/12(50%)에 못 미쳐 후보에서
완전히 빠졌다.

④ **Isaac 은 벨트 부품을 물체로 포함한다** — "metal roller"(컨베이어 롤러 자체)가
후보에 들었다.

⑤ **2B 도 열거는 된다** — test4 에서 2B 는 4B 가 놓친 "book" 까지 후보에 넣었다
(4B: black bag/keyboard/pink foam block, 2B: +book). 피크 VRAM **4149 MB(4.1 GB)**
로 4B(8.6 GB)의 절반 이하다.

## 2. (b) 검증 — SAM3 검출을 VLM 이 확인

`verify.py` 의 질문(원문 그대로, 크롭마다):

> *"Is the object inside the red rectangle a {label}? Answer yes or no."*

크롭은 박스를 중심 기준 2배로 확장한 컨텍스트 이미지에 빨간 사각형을 그린 것.

| bag | 모델 | tag | score≥0.4 (yes%) | 0.2~0.4 (yes%) | score<0.2 (yes%) |
|---|---|---|---|---|---|
| test4 | 4B | hand | 94% (34/36) | 89% (8/9) | 48% (19/40) |
| test4 | 4B | auto | 97% (74/76) | 67% (4/6) | 37% (20/54) |
| test2 | 4B | hand | 93% (40/43) | 0% (0/11) | 13% (9/71) |
| test2 | 4B | auto | 89% (136/153) | 16% (15/92) | 46% (98/211) |
| isaac | 4B | hand | 75% (72/96) | 81% (13/16) | 64% (47/73) |
| isaac | 4B | auto | 97% (360/372) | 97% (63/65) | 92% (364/395) |
| test4 | 2B | hand | 100% (36/36) | 100% (9/9) | **92% (37/40)** |
| test4 | 2B | auto | 92% (85/92) | 33% (6/18) | 38% (38/101) |

**핵심 관찰:**

- **4B 는 실기(test2/test4)에서 저점수(<0.2)를 52~87% no 로 거른다** — test4 hand
  52%(21/40 no), test2 hand 87%(62/71 no). 진짜로 물체가 아닌 검출을 걸러내는
  신호로 쓸 수 있어 보인다.
- **2B 는 검증기로 부적합하다** — 같은 test4 hand `<0.2` 구간에서 4B 는 52% 를
  no 로 거르는데(yes 48%), **2B 는 같은 구간에서 92% yes 다.** 저점수 검출을
  걸러내지 못한다.
- **Isaac 은 조각도 yes 로 통과시킨다** — auto `<0.2` 구간이 **92% yes**(364/395).
  벨트 부품 조각(§1-④)까지 "blue bar" 로 확인해 버린다는 뜻이다.
- → **검증은 의미(semantic) 오류(다른 범주의 물체)는 잡지만, 조각·절단은 못
  잡는다.**

### "≥0.4 인데 no" 가 진짜 오검출인지 — 크롭을 직접 봤다

지시받은 3개 표본(test2 hand 전수 3개, test2 auto 17개 중 5개, test4 4B auto
2개)을 `crop_file` PNG 로 열어 판정했다.

| bag/tag | 크롭 | 라벨 | score | 육안 판정 |
|---|---|---|---|---|
| test2 hand | 0002_4.png | laptop | 0.490 | **진짜 오검출** — 빨간 박스 안은 카키색 마분지 상자, 노트북이 아니다 |
| test2 hand | 0004_5.png | laptop | 0.479 | **진짜 오검출** — 같은 마분지 상자 |
| test2 hand | 0007_6.png | laptop | 0.543 | **진짜 오검출** — 같은 마분지 상자 |
| test2 auto | 0000_28/32/33/37, 0002_34.png | red box | 0.41~0.50 | **색 오검출** — 5개 전부 빨간 박스 안이 분홍색 폼블록/상자다. 같은 물체가 다른 검출에서는 "pink foam block" 으로 정확히 잡힌다 |
| test4 4B auto | 0002_8.png, 0004_15.png | pink foam block | 0.648 / 0.563 | **판정 보류(아래 참조)** |

**test4 4B auto 2건은 파일이 덮어써져 있었다** — `crops_auto/*.png` 는 파일명이
`프레임_인덱스` 규칙이라, 같은 bag 을 2B 로 재실행한 `verify.py sam --tag auto`
호출이 **같은 파일명에 다른 프롬프트 조합의 크롭을 다시 썼다.** 지금 그
경로에 있는 이미지는 4B 검증 당시의 것이 아니다(`crops_auto.json` 현재 내용은
label `book`, score 0.115/0.125 — `verify_4b_auto.json` 이 기록한 label
`pink foam block`, score 0.648/0.563 과 다르다). `verify_4b_auto.json` 이 남긴
원본 `box`/`crop_box` 좌표로 원본 프레임에서 다시 잘라 확인한 결과: **0002_8 은
박스 좌표가 4px 이내로 거의 일치**(같은 분홍 폼블록 코너 마운트로 판단),
**0004_15 은 좌표가 완전히 다른 위치**(x 38~85 vs 원본 442~482)라 신뢰 불가.
0002_8 재구성본은 실재하는 분홍 폼블록이 노출 과다로 거의 흰색에 가깝게
찍혀 있다 — "물체가 없다" 는 오검출이 아니라 "이 조명에서는 색이 안 보인다"
에 가깝다.

**test4 프레임 0000 우측 가장자리 확인(§1-①):** box `[556,186,640,218]` score
0.9375, x1=640 이 프레임 폭과 정확히 같아 화면 밖으로 잘려 있다 — 예시가 샌
게 아니라 실측이다.

**test2 프레임 0000 자동 후보 3개 확인:** "black chair" — 하단 우측에 검은
바퀴의자가 실제로 있다. "white cloth" — 롤러 위에 흰 천이 실제로 걸쳐 있다.
"red box" — 컨베이어 위에는 없다. 화면 오른쪽 배경 선반의 **빨간 철제 캐비닛**을
가리키는 것으로 보인다(질문이 명시적으로 배경 제외를 지시했는데도 새어
들어옴). SAM3 가 이 프롬프트를 컨베이어 위에서 접지(ground)시키며 색이 다른
분홍 폼블록에 붙은 것 — 열거 단계와 검증 단계의 오류가 사슬로 이어진 사례다.

## 3. 대가

| 항목 | 실측 |
|---|---|
| 프레임당 지연(열거, 640×480) | **557 ms**(test4, 4B) |
| 프레임당 지연(열거, 1280×720) | **1529 ms**(test2, 4B) |
| 크롭당 지연(검증) | 중앙값 **~75~98 ms**(bag 별 72.4~98.2 ms) |
| VRAM(4B) | **8.5~8.7 GB** — SAM3(4 GB)와 12 GB 안에 동거 불가 |
| VRAM(2B) | **4.1 GB**(4149 MB) — SAM3 와 동거 가능 |
| 로드 시간 | **10.5~18.5 s**(모델·bag 별) |

## 4. 결론 — 열거는 채택 후보, 검증은 보류

**열거(자동 프롬프트 생성)는 채택 후보다.** 사람이 놓친 물체를 찾아내고
(§1-①), 남은 문제(이름이 SAM3 최적이 아님, §1-②)는 이 저장소가 이미
`PROMPT_ALIASES` 로 다루고 있는 문제와 같은 축이다.

**검증은 보류다.** 4B 는 실기에서 저점수 검출의 상당 부분(52~87%)을 걸러내
보이지만, 조각·절단은 못 걸러내고(Isaac §2), 저비용 2B 는 애초에 검증기로
못 쓴다(§2 — 같은 저점수 구간에서 92% yes).

**다음 단계 설계(제안):** VLM 이 물체당 이름 후보를 2~3개 내고, **SAM3 점수로
그중 최고 점수 문구를 고른다** — SAM3 Agent 의 `segment_phrase → examine` 루프와
같은 구조다. 이러면 "water bottle vs thermos" 같은 이름 문제가 SAM3 자신의
점수로 자동 해소된다. 검증은 트랙 생성 시 1회만 부르는 형태로 나중에 재검토
한다. **노드 통합은 아직 안 했다** — 스파이크 코드는 저장소 밖 `~/vlm/spike/`
에 있다.

## 5. 재현 명령

```bash
cd ~/vlm/spike
./run.sh /home/ingon/roboworld/bags/test4 "black bag,keyboard,manual" 4b
./run.sh /home/ingon/roboworld/bags/test2 "thermos,laptop,manual,cell phone" 4b
./run.sh /home/ingon/roboworld/bags/isaac_belt_moving "blue bar with holes" 4b
./run.sh /home/ingon/roboworld/bags/test4 "black bag,keyboard,manual" 2b
```

각 실행은 `frames.py → enumerate.py → verify.py sam(hand) → verify.py
sam(auto) → verify.py vlm(hand) → verify.py vlm(auto) → report.py` 순서로
`~/vlm/out/<bag>/` 아래 결과를 남긴다.

---

## 부록: 외형 re-ID(DINOv3)는 이번에 안 한다

가림 중 ID 가 실제로 바뀌는지부터 다시 쟀다(`output/reid_gate/log.txt`).
`=== distinct ids per label` 표:

| bag | 라벨별 distinct ID |
|---|---|
| test3 | pink foam block: [1] · book: [3] · glove: [2] — **라벨당 1개** |
| test4 | book: [3] · keyboard: [1] · black bag: [2] — **라벨당 1개** |
| test5 | keyboard: [1] · gray notebook: [4] · black bag: [2] · book: [3] — **라벨당 1개** |
| isaac_belt_moving | blue bar with holes: [1,2,3,4,5,6,7,8,10,11,17,20,21] — **13개, 전부 별개 블록** |

**실기 3 bag(test3/test4/test5) 는 라벨당 ID 가 정확히 1개다 — 실행 내내 ID 전환이
0건이었다.** Isaac 의 13개 ID 는 컨베이어 위 서로 다른 블록 각각에 대응하며,
외형은 전부 동일(같은 "blue bar with holes")하다. 그중 ID 7(444프레임,
x −0.91~−0.70)과 ID 17(1553프레임, x −0.87~−0.79)만 화면 왼쪽 가장자리의 같은
블록으로 보이는데, 둘 사이에 **3.87초의 공백**이 있고(CSV 타임스탬프 실측)
그 뒤 새 ID 로 이어졌다 — 위치가 아니라 트랙이 끊긴 뒤 재등장이다.

**이번에 DINOv3 같은 외형 re-ID 를 조사하지 않은 이유는 스위치가 0건이라
고칠 실패가 없기 때문이다.** `docs/README.md:253` 이 이미 같은 결론을 남겨
뒀다 — *"원인은 위치가 아니라 지속시간이었고 `occlusion_hold`(2026-08-11)로
이미 고쳐졌다. 그 이후 test4/test5 실행 전부에서 ID 변경 0건"*. 이번 측정은
그 결론과 일치한다.
