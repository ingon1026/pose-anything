# SAM3 비디오 세션 스파이크 (2026-08-28)

**질문:** `tracker.py` 437행이 하는 프레임 간 정체성 유지를 SAM3 가 이미 들고 있는가.
그렇다면 채택할 만한가.

**답:** 기술적으로 **전부 통과**했다. 채택 불가 사유로 꼽았던 두 가지(프롬프트 교체,
프레임 건너뛰기)가 둘 다 성립한다. **대가는 속도 2.6배와 비공식 내부 정리 코드**다.

측정 환경: RTX 4070 Ti 12,282 MiB, `transformers 5.5.0`, `facebook/sam3`, bf16,
입력 `bags/test4` 색상 프레임 90장(640×480 rgb8). Isaac 정지 상태.
스파이크 스크립트는 저장소에 안 넣었다 — 재현하려면 이 문서의 코드로 충분하다.

## 1. 결과

| 항목 | 실측 |
|---|---|
| 체크포인트 | `facebook/sam3` 그대로 로드됨 — 별도 가중치 불필요 |
| 로드 비용 | **1,916 MiB / 7.7초** |
| 스트리밍 | `add_new_frame(pixel_values, frame_idx)` 90프레임 정상 |
| **중간 프롬프트 추가** | **성공** — frame 44 에서 `add_text_prompt`, 객체 1 → 3 |
| **프레임 건너뛰기** | **성공** — stride 5(bag frame 0→85, 18키프레임) 추적 유지 |
| 마스크 안정성 | 화소수 **7,281~7,340 (±0.4%)** — 18키프레임 내내 |
| **처리량** | **376~623 ms/frame** (GPU 경합에 따라 변동) |

**프롬프트 교체가 되는 것이 결정적이다.** 이게 안 되면 제로샷 정체성을 파는 것이라
채택 불가였다.

## 2. 속도 — 이것이 대가다

| | 현행 이미지 경로 | 비디오 세션 |
|---|---|---|
| 키프레임 추론 | **217 ms** (1.2.0 A/B 중앙값) | **418~590 ms** (중앙값 범위) |
| 검출 발행률 | **4.587 Hz** (2026-08-28 실측) | **1.7~2.4 Hz** |

**2.6배 느리다.** 벨트 46mm/s 에서 검출 간 이동거리가 **10mm → 27mm** 가 된다.

⚠ **조건이 바이트 동일하지 않다** — 스파이크는 640×480 원본을 비디오 프로세서
자체 리사이즈에 맡겼고, 출하 경로는 `image_size=672` 다. 자릿수 비교로만 읽을 것.

## 3. 메모리 — 누수가 있고, 잡을 수 있다

### 기전
`Sam3VideoInferenceSession` 이 프레임마다 상태를 쌓는데 **부분 정리 경로가 없다.**
삭제는 `reset_tracking_data`(:404) · `reset_inference_session`(:416) ·
`reset_state`(:428) 의 `.clear()` 셋뿐이고 **전부 전체 삭제**다.

누수의 본체는 추적 상태가 아니라 **저장된 프레임 자체**다 —
`add_new_frame` 이 `self.processed_frames[frame_idx] = pixel_values`
(`modeling_sam3_video.py:396`)로 모든 프레임을 영구 보관한다.
상계가 있는 것은 `_vision_features`(`:87`, `max_vision_features_cache_size`) 하나뿐이다.

### 실측

| 조치 | RSS 증가 |
|---|---|
| 없음 | **10.94 MiB/frame** (평탄화 없음) |
| `non_cond_frame_outputs` 만 정리 | 9.15 (**16% 감소 — 본체가 아니다**) |
| + `processed_frames` **키를 pop** | 2.19 — **틀린 방법. 아래 참조** |
| + `processed_frames` **값만 비움** | **1.52** (n=20 이후 정상상태 **0.83**) |

`inference_state_device="cpu"` 는 메모리 뱅크를, `video_storage_device="cpu"` 는
프레임 픽셀을 각각 내린다(`processing_sam3_video.py:163`) — **둘은 별개 인자다.**
오프로드는 VRAM 은 잡지만 누수를 CPU RAM 으로 옮길 뿐이다(`alloc` 9.10 → 0.89,
처리량 손해 6%). 그것만으로는 누수가 안 잡힌다.

### ⚠ 정리는 키를 지우면 안 된다 — 이걸 틀렸다가 잡았다

**처음에 `processed_frames` 의 키를 `pop` 했다. RSS 는 잡히고 마스크도 안정적이라
되는 것처럼 보였다. 그런데 `num_frames = len(processed_frames)`
(`sam3_tracker_video/modeling:2610`)이고 `tracker:2345` 의
`ref_frame_idx >= num_total_frames` 가 살아 있어서, len 이 줄면 object pointer 가
0 개가 된다 — 예외 없이.** 즉 **채택 이유인 정체성 유지가 꺼진 채로 "성공" 을
보고할 뻔했다.**

실측으로 확인한 차이:

| | 키를 pop | 값만 비움 |
|---|---|---|
| `num_frames` (i 최종 89) | 21 로 축소 | **90** ✓ |
| object pointer | **0 개 (조용히)** | 살아 있음 |
| 실체 보관 프레임 | 21 | **16** (= HORIZON 15 + 1) |

```python
HORIZON = 15   # max_object_pointers_in_encoder(16) - 1. 7 이 아니다 —
               # object pointer 루프가 frame_idx-1 … frame_idx-15 를 읽는다
               # (sam3_tracker_video/modeling:2320, :2340-2352).

def prune(sess, cutoff):
    # 키는 남기고 값만 비운다. cond_frame_outputs 는 건드리지 않는다 —
    # max_cond_frame_num=4 로 이미 상계가 있고, 지우면 죽는다(실측):
    #   ValueError: maskmem_features in conditioning outputs cannot be empty
    for fi in [k for k in sess.processed_frames if k < cutoff]:
        t = sess.processed_frames[fi]
        if t.numel():
            sess.processed_frames[fi] = torch.empty(0, device=t.device, dtype=t.dtype)
    for d in sess.output_dict_per_obj.values():
        pf = d.get("non_cond_frame_outputs")
        if isinstance(pf, dict):
            for fi in [k for k in pf if isinstance(k, int) and k < cutoff]:
                pf.pop(fi, None)
```

**검증 방법** — 정리를 넣었으면 반드시 이 둘을 확인할 것:
```python
assert sess.num_frames == i + 1                      # 줄었으면 object pointer 가 죽었다
sorted(k for d in sess.output_dict_per_obj.values()
         for k in d["non_cond_frame_outputs"])       # 최근 인덱스가 연속이어야 한다
```

### ⚠ 되감김 함정
`add_new_frame` 의 `frame_idx` 기본값이 **`len(self.processed_frames)`**(`:391`)다.
키를 지우는 방식을 쓰면 이 길이가 줄어 인덱스가 되감기고 프레임이 서로
덮어써지는데 **에러가 안 난다.** 값만 비우는 방식에서는 len 이 유지되므로
자동 인덱스도 안전하다.

## 4. 채택 시 알아야 할 것 (소스 근거, 미실측)

`video-api` 조사분. **내가 실측으로 확인한 것과 구분해서 읽을 것.**

- **트래커 전용 모드가 없다.** `_det_track_one_frame` 이 넣는 프레임마다
  `get_vision_features` + `run_detection` 을 무조건 돌린다(`sam3_video/modeling:1613-1622`).
  즉 지금의 `detect_interval=5` + `tracker.py` 처럼 **"싼 중간 프레임" 을 얻는 경로가
  구조적으로 없다.** 이것이 속도 대가가 협상 불가인 이유다.
- **`add_new_frame` 을 직접 부르는 경로가 맞다.** `forward(frame=...)` 로 넘기면
  내부 `streaming=True` 가 되어 hotstart 중복 트랙 정리가 꺼진다
  (`:1714`, `:865`, `:909`). 인자 유무로 갈리는 분기라 주석이 필요하다.
- **프롬프트 제거 API 가 없다.** `prompt_id = len(self.prompts)`(`:224`)라 dict 을
  손으로 지우면 **다음 `add_prompt` 가 살아 있는 id 를 재사용**해 기존 트랙이
  조용히 다른 라벨로 붙는다. **교체는 세션 재생성이 정답이다**(세션은 가중치를
  안 들어 싸다. 단 트랙 ID 는 초기화된다).
- **리셋은 `reset_tracking_data()`** 가 맞다 — 프롬프트·캐시를 남기고 `max_obj_id` 가
  유지되어 ID 재사용이 없다. 단 어느 리셋도 `processed_frames` 를 안 비우므로
  따로 비워야 한다.
- **임계값이 그대로 이관되지 않는다.** 현행 `threshold=0.4` 대신
  `score_threshold_detection`(기본 0.5) · `new_det_thresh`(0.7)이고,
  `assoc_threshold=0.1` 저점수 2차 매칭에 해당하는 출력이 **없다**.
- **`facebook/sam3` 가 곧 `Sam3VideoModel` 이다** — config `architectures`가 그것이고,
  비디오 경로가 추가로 요구하는 가중치는 78 MB(bf16 39 MB)뿐이다.

## `kernels` — 실측 결과 **설치하지 말 것**

실행 로그에 매번 뜬다:
```
kernels library is not installed. NMS post-processing, hole filling,
and sprinkle removal will be skipped. Install it with `pip install kernels`
```
frame 70 부터 두 번째 객체가 56,876 → 70,586 화소(프레임의 23%)로 자라는 것의
원인인가 싶어 **실제로 깔아서 A/B 했다. 답은 아니오다.**

**① `pip install kernels` 는 저장소를 깨뜨린다.** 최신 `0.16.1` 이 깔리는데
`transformers/integrations/hub_kernels.py:89` 가 `LayerRepository(...)` 를
revision/version 없이 만들고 0.16 은 그걸 필수로 요구한다 →
`ValueError: Either a revision or a version must be specified.`
`activations.py` 를 타고 들어가므로 **비디오 경로만이 아니라 모든 모델 임포트가
죽는다.** `transformers 5.5.0` 이 요구하는 것은 **`kernels<0.13,>=0.12.0`** 이다.

**② 호환 버전(`0.12.3`)을 깔아도 결과가 안 바뀐다.** 같은 프레임 같은 코드로
A/B 했고 마스크 화소가 **17개 지점 전부 동일**하다:
```
없을 때  7309 7281 7288 7305 7293 7298 7303 7296 7296 7296 7298 7304 7292 7340 7293 7293 7295
있을 때  7309 7281 7288 7305 7293 7298 7303 7296 7296 7296 7298 7304 7292 7340 7293 7293 7295
```
폭주하던 객체 2 도 그대로다(70,586 → 70,586). **NMS·구멍 메우기·잡티 제거 부재는
그 현상의 원인이 아니다.**

**③ 런타임 허브 의존이 새로 생긴다.** 첫 실행에서 `Fetching 5 files` 로 허브에서
내려받는다. 로봇 파이프라인에 **네트워크 의존을 새로 만드는 것**이라 이득 0 으로는
정당화가 안 된다.

→ **설치하지 않았다. 환경은 원상복구했다**(`kernels`·`kernels-data` 모두 제거,
`transformers` 임포트 정상 확인). 저 경고는 **무시해도 되는 것**으로 결론.
객체 2 폭주의 원인은 **여전히 미확인**이다.

## 5. 남은 미확인


- **정상상태 1.06 MiB/frame 이 여전히 남는다.** 나머지 프레임별 딕셔너리
  (`frames_tracked_per_obj` · `obj_id_to_tracker_score_frame_wise` ·
  `unmatched_frame_inds` · `overlap_pair_to_frame_inds` · `suppressed_obj_ids` ·
  `output_buffer`)를 안 잘라서다. 작지만 **0 은 아니다.**
- **추적 품질을 현행 트래커와 비교하지 않았다.** 마스크 안정성(±0.4%)은 좋아 보이지만
  **가림 27% 문제를 실제로 고치는지는 미측정**이다. 그게 채택의 진짜 근거인데 아직 없다.
- frame 70 부터 두 번째 객체가 **56,876 → 70,586 화소**(프레임의 23%)로 자란다.
  `book` 프롬프트에 붙은 오검출로 보이나 **확인 안 했다.**
- `image_size=672` 조건에서 재측정 안 했다.

## 6. 판정 — **채택하지 않는다**

기술적으로는 전부 통과했다. **그런데 속도가 이 저장소가 이미 기각한 값보다 느리다.**

### 1. 저장소가 같은 축에서 이미 이 속도를 기각했다 — 결정타

`isaac.launch.py:38-43` 이 `image_size` 1008 을 기각하고 672 를 채택한 **유일한 근거**:

> *"672 에서 4.98 Hz, 1008 에서 2.68 Hz(−46.1%). Isaac 실시간은 이미 느려서
> (`detect_interval=1` 인 이유가 그것) **2.7 Hz 면 추적기가 설 자리를 더 잃는다.**"*

비디오 세션은 **1.7~2.4 Hz** 다(GPU 경합에 따라 변동. 중앙값 418~590 ms).
**가장 좋은 조건에서도 이미 기각된 2.68 Hz 에 못 미친다.**
그리고 §4 에서 봤듯 **싼 중간 프레임 경로가 구조적으로 없어** 협상이 안 된다.

### 2. 행수가 줄지 않는다 — 이 제안의 원래 근거가 무너진다

*"`tracker.py` 437행이 사라진다"* 는 **틀렸다.**

| | 행수 |
|---|---|
| 실제로 대체되는 것 | **181** (41%) — 그중 **79행(중복 병합)은 실기에서 이미 no-op** |
| 남는 것 | **349** — `fusion.py` 187행 **전부**(2D association 을 한 줄도 안 한다), `Track` 필드·발행 계약 90, `update_obb` 41 |
| 새로 지어야 하는 것 | **+175~270** — 세션 래퍼, obj_id↔Track 수명 결선, 정리 코드+회귀 테스트, 세션 재생성 경로, 두 시계 정합 |
| **순 차액 (현실)** | **+39 — 코드가 늘어난다** |

그리고 stride 를 쓰면 `pipeline.py` 의 LK 광학흐름 120행을 **못 지운다** — 세션
호출 사이 프레임에 세션이 아무것도 안 주는데 그 간격(588ms)이 `T_STALE`(0.5s)을
넘어 발행이 끊기기 때문이다.

### 3. 588ms 가 시간 상수 둘을 조용히 깬다

1. **`dt` 클립이 매 프레임 발화한다.** `fusion.py:87` · `pipeline.py:408` 둘 다
   `np.clip(dt, 1e-3, 0.5)`. 588ms > 0.5s 라 **필터가 실제 경과의 85%만 시간을
   전진**시킨다 → `Q` 과소 계상 → P 가 진실보다 작고 χ² 게이트가 진실보다 좁다.
   `fusion.py` 도스트링의 불변식 1·3(교착 불가능성)이 **P 가 실제 경과로 자란다는
   전제** 위에 서 있는데 그게 매 프레임 깨진다. **클립은 원래 "스탬프 이상 방어"
   라 정상 동작으로 설계돼 있어 로그가 조용하다.**
2. **`T_STALE` 여유가 0 이 된다.** `tracker.py:16` 의 주석이 이걸 **"그리퍼에 도달할
   수 있는 stale pose 최대 나이 (안전 속성)"** 이라고 못 박는다. 4.587 Hz(218ms)
   에서는 연속 기각 2회를 흡수하지만(2×218=436 ≤ 500), 1.7 Hz(588ms)에서는
   **0회 — 단 한 프레임 기각으로 즉시 발행이 끊긴다.** 실측 검출 실패율이
   **book 22% · gray notebook 22%**(§3-1 `publish_gap`)라 흡수 장치가 사라진다.
   부수로 `CONFIRM_N=3` 승격이 654ms → **1.76초**.

### 4. 그 외 비용

- 정리 코드가 **라이브러리 내부에 의존**한다 — 버전 올리면 조용히 깨질 수 있고
  회귀 테스트를 같이 지어야 한다.
- **노드의 계약은 "치환" 인데 세션 API 는 "추가" 다.** `parse_prompts` 가 매번
  전체 목록을 돌려주고 노드는 `self.prompts` 를 통째로 갈아끼운다
  (`perception_node.py:380`). 세션엔 제거 API 가 없으므로 `["book"] → ["cup"]`
  을 add 로 흉내내면 `book` 이 계속 검출된다. **"프롬프트 추가 가능" 을
  "교체 가능" 으로 읽으면 안 된다** — 교체는 세션 재생성이 유일한 경로다.
- **`new_det_thresh` 기본 0.7** (`configuration_sam3_video.py`). README §2 가
  `publish_score_min=0.6` 이 **정상 검출 90프레임을 잘랐다**고 기록한 그 축이다.
  0.7 이면 그 물체는 **트랙이 태어나지도 않는다.** ⚠ 같은 검출 헤드 출력으로
  보이나 **스케일 동일성 미확인** — 확인 전에 두 수치를 직접 비교하지 말 것.
- **`init_trk_keep_alive=30` 이 stride 와 결합한다.** 연속 15fps 면 2.0초로
  실측 최대 가림 2.6초(`tracker.py:247`)에 못 미치는데, stride 5 면 간격이
  333ms 라 같은 30프레임이 10초가 된다. **stride 가 가림 예산을 바꾸는 이 결합은
  어디에도 문서화돼 있지 않다.**
- **`camera_info` 변경 경로가 비싸진다.** 해상도가 세션 생성 시 고정이라 매
  변경이 **세션 재생성**이 된다. 냉시동 검증(§3-1)에서 Isaac 브리지와 bag 이
  서로 다른 해상도를 같은 토픽으로 쏘며 이 경로가 **반복 발화**한 전력이 있다.
- **`/perception/status` 에 트래커 유래 값이 0 개다**(실측 확인). 대체해도 status
  계약은 안 깨지는데, **그게 나쁜 소식이다** — 트래커가 조용히 망가져도
  heartbeat 는 계속 `OK` 를 낸다.

### 비용이 아닌 것 — 처음에 잘못 적었다
**~~프롬프트 교체 시 트랙 ID 가 초기화되는 것~~ 은 계약 위반이 아니다.**
현행 `tracker.reset()` 이 이미 `self._next_id = 1` 로 초기화하고
(`tracker.py:275`) `on_prompt` 가 `_reset_input_state` 를 통해 그 경로를 탄다
(`perception_node.py:380`). 세션 재생성은 **현행 동작과 동형**이다.
단 **`_reset_input_state` 경로를 그대로 타야 한다** — 세션만 갈아끼우고
`_withdraw_output()` 을 건너뛰면 소비자가 옛 pose 를 쥔 채 남는다.

### 결론

**얻는 것(가림 강건성)은 아직 측정도 안 됐고, 치르는 것은 전부 실측·소스로
확인됐다.** 이 순서로는 채택할 수 없다.

**재개 조건:** ① 가림 27% 를 실제로 고친다는 측정이 먼저 나오고, ② 속도가
4 Hz 대로 오를 길(더 빠른 GPU, 또는 오프라인 전용 경로)이 생길 것.
오프라인 경로(`run_offline.py`)는 `T_STALE`·`dt` 제약이 없으므로 **가림 A/B
측정 도구로는 지금도 쓸 수 있다** — 채택과 별개다.

### 이 스파이크가 산 것
하루가 아니라 **몇 시간에** 위 전부를 알아냈다. 특히 `processed_frames` 키를
pop 하면 object pointer 가 조용히 죽는 것(§3)은 구현에 들어갔으면 **"되는데
가림이 안 낫네" 로 며칠을 태울** 종류의 결함이었다.

---

## 7. 회귀망 — 33개가 깨지고 16개는 갈 곳이 없다

`tracker-delta` 분석분. 전체 145개 중:

| 분류 | 개수 |
|---|---|
| 상태 대입 (약한 테스트, 재작성 비용) | 9 |
| 행위 단언 — 이식 가능 | 8 |
| **행위 단언 — 단언 불가로 강등** | **16** |

**행위 24 : 상태 9 = 73% 가 행위 단언이다. 약한 테스트를 정리하는 작업이 아니다.**

**16개가 "강등" 인 이유:** `test_sam3_detector.py:1-13` 이 적어둔 대로
**CI 는 의도적으로 torch 를 안 깔고 `geometry`·`tracker`·`pipeline` 만 돌린다.**
association 이 모델 안으로 들어가면 이 16개(`test_tracker` 6 + `test_occlusion`
8 전부 + 2)는 **GPU 없이 한 줄도 못 돈다** — 재작성 비용이 아니라 **회귀망 상실**이다.
그중 2개(`test_rescue_rejected_on_depth_conflict`,
`test_depth_conflict_does_not_steal_match`)는 **능력 자체가 없어져 재작성조차
불가능하다** — 모델은 RGB+메모리로만 연관하므로 depth 채널이 association 에 없다.

LK 까지 걷어내면 6개가 더 폐기돼 **총 39개**.

## 8. ⚠ 이걸 "Re-ID 개선" 으로 팔면 §3-2 재발이다

README §3-2(`docs/README.md:238`)가 **"Re-ID 복귀 매칭이 위치 기반이라 가림 중
이동하면 ID 복구 실패"** 를 **데이터로 반박**했다 — 위치 게이트는 물체 폭 2.5배
(keyboard 328mm)를 허용하는데 기록된 모든 ID 분실은 29~153mm 에서 났다. 원인은
위치가 아니라 지속시간이었고 `occlusion_hold`(2026-08-11)로 이미 고쳐졌다.
**이후 test4/test5 실행 전부에서 ID 변경 0건.** 삭제 한계 6.0초 · 벨트 변위
276mm < 게이트 328mm 라 **재현이 원리적으로 불가능**하다.

**→ ID 분실은 현행 조건에서 0건이다. 세션이 팔 수 있는 명제는 하나뿐이다 —
가림 중 마스크 오염(test4 book 27%) 개선. 그리고 그 개선폭은 한 번도 측정된 적이
없다.** 이 둘을 뭉뚱그리면 §3-2 가 이미 기각한 서사를 되살리는 것이다.

## 9. 되살리려면 이 순서로

1. **이득을 먼저 재라** — 세션 마스크가 test4 book 가림 구간 오염 27% 를 얼마나
   줄이는지. 그 숫자 없이는 나머지가 전부 공중이다. 실기 회귀 전까지 "채택" 이
   아니라 **"후보"** 다(§3 공통 교훈: 되돌린 7건 중 3건이 시뮬·합성으로 결정한 함정).
2. **이득이 유의하면 속도를 먼저 해결하라.** 4 Hz 대를 회복 못 하면 §6-3 의 두
   무성 실패를 노브로 못 막는다. `dt` 클립의 두 용도(스탬프 방어 vs 필터 시간)
   분리가 선행돼야 한다.
3. **`Sam3Detector.detect()` 이음매를 유지**하고 그 뒤에 세션을 감싸
   `[{label, mask, box, score, obj_id}]` 를 돌려주는 **어댑터로 시작하라** —
   `StubDetector` 가 살아남아 `test_pipeline`·`test_perception_node` **53개가
   그대로 돌고**, 깨지는 것이 tracker/occlusion 계열로 국한된다.
4. **Isaac 프리셋(`detect_interval=1`)에서만 먼저 재라** — 이미 매 프레임 검출이라
   stride 교란이 없는 유일한 구성이다.

**그리고 "코드가 줄어든다" 로 팔지 말 것 — 안 줄어든다(§6-2).**
