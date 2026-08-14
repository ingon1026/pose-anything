# Isaac Sim 연동 안정화 (2026-08-14)

어제([isaac_sim_connection_2026-08-13.md](isaac_sim_connection_2026-08-13.md))
연결한 Isaac Sim ↔ `roboworld_perception` 파이프라인을 하루 돌려보며 나온
고장을 정리했다. 코드 수정 3 커밋(`715fd18`, `76014fc`, `c6e3874`)이 들어갔다.

이 문서는 **"증상 → 진짜 원인 → 조치"** 순으로 적었다. 오늘 증상만 보고
엉뚱한 곳을 여러 번 팠기 때문에, 같은 길로 다시 들어가지 않도록 **틀렸던
가설도 함께 남긴다.**

---

## 0. 한 줄 요약

이미지 토픽이 죽는 진짜 원인은 네트워크도 QoS 도 아니고 **VRAM 고갈로 인한
CUDA 핸들 무효화**다. Isaac Sim + SAM3 + RViz 를 12 GB GPU 하나에 올리면
넘친다.

### 오늘 시점의 실제 연결

![ROS 2 노드·토픽 그래프](images/isaac_ros2_graph_2026-08-14.png)

`rqt_graph` 는 미설치라(`sudo apt install ros-jazzy-rqt-graph`) 살아있는
그래프를 rclpy 로 조회해 graphviz 로 그렸다. Isaac 의 SDG 라이터 노드 이름은
`_Render_PostProcess_SDGPipeline_Replicator_01_NodeWriterWriter_02` 처럼
기계적이고 그래프를 다시 만들 때마다 접미사가 바뀌므로, **이름이 아니라
발행하는 토픽으로** 판별해 `Isaac Sim (color/depth/camera_info)` 로 표기했다.

---

## 1. 이미지 토픽이 통째로 죽는다 ← 오늘의 본체

### 증상

- Isaac 뷰포트는 멀쩡히 30~37 FPS 로 돌아간다.
- `ros2 topic list` 에 카메라 토픽 3 개가 다 보인다.
- `camera_info`(작은 메시지)는 오는데 **이미지만 0 장**.
- 재시작 말고는 복구가 안 된다. 타임라인 stop/play 도 안 통한다.

### 진짜 원인

Isaac 로그(`~/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/6.0/kit_*.log`)에:

```
[isaacsim.ros2.nodes] CUDA error 400: cudaErrorInvalidResourceHandle
  at isaacsim.ros2.nodes\nodes\OgnROS2PublishImage.cpp:466
```

VRAM 이 모자라면 렌더프로덕트 텍스처가 재할당되고, 그 순간 ROS2 발행 노드가
들고 있던 CUDA 핸들이 무효가 된다. 그 뒤로 **매 프레임 실패**한다.
`camera_info` 는 CPU 에서 만들어 보내므로 혼자 살아남는다 — 이 비대칭이
"토픽은 보이는데 이미지만 없다" 의 정체다.

실측: 최초 발생이 SAM3 기동 직후였다. 하루 누적 **23,410 회**.

### 조치

`image_size` 파라미터를 노출해 SAM3 입력을 **1008 → 672 px** 로 낮췄다.
카메라가 640x360 이라 1008 은 어차피 업스케일이었다.

```bash
ros2 launch roboworld_perception perception.launch.py image_size:=672
```

결과: 같은 워크로드에서 **23,410 회 → 0 회**, 4 분 연속 녹화 완주.

### ⚠️ 이건 완치가 아니다

여유를 만들어준 것뿐이다. 부하를 다시 올리면 재발한다. 실측:

| 구성 | CUDA 에러 |
|---|---|
| Isaac + SAM3(`detect_interval=8`), RViz 없음 | 0 회 |
| Isaac + SAM3(`detect_interval=1`) + RViz | **37,151 회 → GPU 크래시** |

마지막 조합에서는 `[omni.rtx] GPU crash is detected` 와 함께 Isaac 이
SIGSEGV(139)로 죽었다.

**근본 대책은 VRAM 확보다.** RTX 4070 Ti 12.3 GB 중 Chrome·Notion·PowerPoint
·Docker 가 이미 5.3 GB 를 잡고 있어 남는 7 GB 로 셋을 감당할 수 없다.
작업 전에 브라우저부터 닫는 것이 가장 효과가 크다.

---

## 2. 틀렸던 가설들 (같은 길로 다시 가지 말 것)

증상이 "이미지만 안 온다" 였기 때문에 전송 계층을 오래 팠다. **전부 아니었다.**

| 의심한 것 | 왜 아니었나 |
|---|---|
| WSL2 커널 소켓 버퍼 208 KB < 이미지 675 KB | `sysctl net.core.rmem_max=16777216` 올려도 전송률 그대로 |
| 웹 콘솔이 카메라를 경합 | `console_only.py` 로 바꿔도 그대로 |
| 뷰포트 캡처가 렌더를 굶김 | 캡처를 멈춰도 그대로 |
| FastDDS 버퍼 프로파일 미적용 | 프로파일 걸어도 차이 없음 |

**교훈: 이미지만 죽고 `camera_info` 는 살아 있으면 전송이 아니라 GPU 를 봐라.**
Isaac 로그의 `[Error]` 를 먼저 grep 하면 5 분 만에 끝난다.

```bash
grep -c cudaErrorInvalidResourceHandle "$(ls -t ~/.nvidia-omniverse/logs/Kit/Isaac-Sim\ Full/6.0/kit_*.log | head -1)"
```

---

## 3. TF 사슬이 끊겨 RViz 에 마커가 안 보인다

### 원인

검출은 `camera_color_optical_frame` 기준으로 발행되는데, `perception_node` 는
`world -> camera_link` 만 발행한다. 그 사이 링크
`camera_link -> camera_color_optical_frame` 은 **RealSense 드라이버가 채우는
것**이라, Isaac 이 이미지를 직접 쏘는 구성에는 아무도 안 보낸다.

### 조치 — `publish_optical_tf` 파라미터 (기본 `False`)

```bash
ros2 launch roboworld_perception perception.launch.py publish_optical_tf:=true
```

**Isaac 구성에서는 반드시 켜야 한다.** 기본값을 `False` 로 둔 이유는 실패
모드가 비대칭이기 때문이다:

- 잘못 **켜면**(bag 재생): 녹화된 RealSense 트리와 부모가 둘이 되어 멀쩡하던
  TF 가 깨진다. 증상이 엉뚱한 곳에 나타나 원인 찾기가 어렵다.
- 잘못 **끄면**(Isaac): 마커만 안 보이고, 해결은 인자 한 줄이다.

### 구현 주의 — 정적 TF 는 리스트로 한 번에

`StaticTransformBroadcaster` 의 발행자 QoS 는 `depth=1` + `TRANSIENT_LOCAL`
이라 **늦게 켠 구독자는 마지막 메시지 하나만** 받는다. `sendTransform` 을
나눠 부르면, 누적 재발행하지 않는 구현에서는 먼저 보낸 변환이 통째로 사라진다.
그래서 두 변환을 리스트로 묶어 한 번에 보낸다.

검증: `/tf_static` 에 `world -> camera_link -> camera_color_optical_frame`
두 줄이 한 메시지로 나오는 것을 실기동에서 확인했다.

### 남은 약점

`child_frame_id` 가 하드코딩이다. 정적 TF 는 `__init__` 에서 한 번 발행하는데
`camera_info` 는 나중에 오므로 이름을 기다릴 수 없다. Isaac 이 다른 optical
frame 이름을 실어 보내면 이 링크는 붙지 않는다.

---

## 4. launch 인자가 조용히 무시된다 ← 함정

노드가 `declare_parameter` 로 갖고 있어도, `perception.launch.py` 의
`parameters=[{...}]` 에 넣지 않으면 **launch 인자로 줘도 무시되고 노드
기본값이 남는다. 에러도 안 난다.**

실제로 이것 때문에 `max_per_prompt:=12` 를 줬는데 기본값 `1` 이 남아,
블록 9 개가 놓인 장면에서 **1 개만 검출되는 것**으로 보였다. 성능 문제로
오해하기 딱 좋다.

이제 넘어가는 인자 (전부 `ParameterValue` 로 형변환 필요 — 치환값은 문자열):

| 인자 | 기본 | 비고 |
|---|---|---|
| `max_per_prompt` | 1 | 같은 물체 여러 개면 반드시 올릴 것 |
| `detect_interval` | 5 | 입력이 느리면 낮춰야 트랙이 선다 (아래) |
| `stale_timeout` | 5.0 | 마커 잔상 정리 임계값 |
| `image_size` | 0 | 0 = SAM3 기본 1008px |
| `publish_optical_tf` | false | Isaac 구성은 true |

확인 방법:

```bash
ros2 param get /roboworld_perception max_per_prompt
```

---

## 5. RViz 마커 깜빡임

원인이 둘이었다.

1. **`_watchdog` 임계값 2.0 초 하드코딩.** bag 재생 기준이라 실시간에는 너무
   짧다. Isaac 실시간 입력은 3~4 초 공백이 예사라 **정상 동작 중에도** 매번
   `DELETEALL` 이 나가 마커가 통째로 사라졌다 나타났다.
   → `stale_timeout` 파라미터로 분리.
2. **`publish()` 가 매 사이클 `DELETEALL` 을 앞세웠다.** RViz 가 지움과 다시
   그림 사이를 렌더링해 번쩍인다.
   → 같은 `ns`/`id` 로 덮어쓰게 두고, 직전에 있다가 사라진 것만 골라 `DELETE`.

---

## 6. 라벨이 안 읽힌다 (2 건)

**가로 잘림** — 라벨은 박스 왼쪽 x 에서 시작해 그려지는데 경계 처리가 없어
`whd=(0.161,0.057,0.0` 처럼 중간에서 끊겼다. `_label_x` 로 가장 넓은 줄 기준
클램프. 폭은 `textlength` 가 아니라 **`textbbox(stroke_width=2)`** 로 잰다 —
실측 238.8 vs 241 로 stroke 번짐만큼 과소평가되어 2 px 씩 다시 잘린다.

**세로 겹침** — 컨베이어처럼 같은 높이에 물체가 늘어서면 박스 y 가 전부 같아
라벨이 한 자리에 쌓인다(블록 7 개 기준 통째로 뭉갬). 화면 왼→오른쪽 순으로
3 단 계단 배치. `track_id` 순으로 매기면 인접 물체가 같은 단에 걸린다.

> ⚠️ 가로 잘림 수정은 **화면으로 확인하지 못했다.** 코드 레벨 렌더 테스트만
> 통과한 상태에서 GPU 가 크래시했다. 다음 세션에서 눈으로 볼 것.

---

## 7. 운영 규칙 (밟으면 시간 날림)

### `stop → play` 를 쓰지 말 것

SDG 파이프라인 전체를 해제했다 재구성하느라 **메인 스레드가 1~13 분 멈춘다**
(실측: 20 회 폴링 만에 응답). 메모리가 6.9 GB → 2.0 GB 로 떨어졌다 다시 오르는
것이 그 과정이다.

**블록 위치를 초기화하려면 Isaac 재기동이 더 빠르다** — 16 초에 뜨고 씬을
새로 열면 배치도 원상복구된다.

### Isaac 그래프를 재생성하면 perception 을 재시작할 것

`stop→play` 나 `cellomni_ros2_rebuild.py` 는 ROS2 퍼블리셔를 **파괴하고 새로
만든다.** 그 전에 떠 있던 구독자는 재매칭되지 않는다. 증상은 "토픽은 보이는데
perception 은 입력 없음" 이다. 새로 만든 구독자는 잘 받으므로 진단이 헷갈린다.

### 해상도 변경은 세션당 한 번만

`cellomni_ros2_rebuild.py` 를 같은 세션에서 두 번 돌리면 `DeletePrims` 로
지워도 옛 SDG 라이터의 퍼블리셔가 살아남아 **옛 해상도의 K 를 계속 뿌린다.**
구독자가 먼저 온 것을 캐시하면 결과가 정확히 절반이 되는데 에러가 안 난다.

```bash
ros2 topic info /camera/camera/color/camera_info --verbose --no-daemon | grep "Publisher count"
# 반드시 1
```

### 입력이 느리면 `detect_interval` 을 낮출 것

입력 0.2 Hz 에 `detect_interval=8` 이면 SAM3 가 **40 초에 한 번** 돈다.
추적기를 그보다 자주 리셋하면 트랙이 설 틈이 없어 검출이 0 이 된다.
실측: `detect_interval=8` → 평균 0.1 개, `=1` → 평균 4.7 개.

### `ros2 topic hz` 는 `--no-daemon` 을 안 받는다

`list`/`echo` 는 받는다. `hz` 에 붙이면 usage 에러만 내고 조용히 죽어서
"토픽이 안 나온다" 로 보인다. 저속(0.2 Hz)에서는 창이 안 차 아무것도 못 내므로
직접 세는 편이 낫다.

---

## 8. 실측 성능 (참고)

640x360 기준, 카메라 → WSL 수신률:

| 구성 | color | depth | 5 초 초과 공백 |
|---|---|---|---|
| Isaac 단독 | 2.33 Hz | 1.37 Hz | 0 회 |
| + 웹 콘솔 | 1.53 Hz | 1.87 Hz | 1 회 |
| + SAM3 | 0.50 Hz | 0.50 Hz | 1 회 |

Isaac 자체는 내내 11~12 FPS 로 렌더링한다. 병목은 네트워크가 아니라
**같은 GPU 를 나눠 쓰는 것**이다.

검출 정확도(정답 대비, 가림 없는 블록):

```
x  0 ~ 4 mm      y  0 ~ 1 mm      z  0 ~ 1 mm
```

로봇 팔에 가린 블록은 마스크가 잘려 중심이 밀리고(-24 mm) depth 중앙값이
팔 표면을 물어 z 가 뜬다(-30 mm). 알고리즘 편향이 아니라 가림이다.

두께는 항상 0 에 가깝게 나온다 — 위에서만 보면 윗면만 보이므로 단일 시점
depth 의 구조적 한계다.

---

## 9. 다음 세션에서 할 것

- [ ] 라벨 가로 잘림 수정을 **화면으로** 확인 (코드만 통과한 상태)
- [ ] `perception.launch.py` 에 Isaac 프리셋 추가 검토
      (`publish_optical_tf:=true image_size:=672` 를 매번 치는 것은 잊기 쉽다)
- [ ] `child_frame_id` 하드코딩을 파라미터로 분리
- [ ] `overlay.py` 에 테스트 없음 — 회귀를 40 개 스위트가 못 잡는다
- [ ] VRAM 여유 확보 방안 (SAM3 분리 실행 또는 렌더프로덕트 축소)
