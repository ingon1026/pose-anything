# Isaac Sim 가상 카메라 연결 (2026-08-13)

Windows 의 Isaac Sim 6.0.1 디지털 트윈에서 RGB-D 를 뽑아 WSL2 의
`roboworld_perception` 에 D455 와 똑같은 토픽으로 먹인다. **연결 완료 상태**이며
인식 패키지 수정은 `img_to_np()` 11줄 한 건뿐이다.

이 문서는 Windows 쪽에서 작업한 사람이 WSL 쪽 작업자에게 넘기는 인수인계다.
"왜 그렇게 했는지" 와 "밟으면 하루 날리는 함정" 위주로 적었다.

---

## 1. 무엇이 연결됐나

```
Isaac Sim 6.0.1 (Windows)                        WSL2 Ubuntu 24.04 / ROS 2 Jazzy
  /Sensors/cam_belt                                  perception_node
      |                                                    ^
      +-- IsaacCreateRenderProduct 640x360                 |
             |                                             |
             +-- ROS2CameraHelper  type=rgb    ------> /camera/camera/color/image_raw
             +-- ROS2CameraHelper  type=depth  ------> /camera/camera/aligned_depth_to_color/image_raw
             +-- ROS2CameraInfoHelper          ------> /camera/camera/color/camera_info
```

토픽 이름을 D455 레이아웃 그대로 맞췄으므로 `perception.launch.py` 를 **아무 인자
변경 없이** 실행하면 붙는다. 런치 파일은 rviz2 와 perception_node 만 띄우고
RealSense 드라이버나 bag 재생을 시작하지 않으므로 그대로 재사용된다.

`run.sh` 는 못 쓴다 — `lsusb | grep RealSense` 로 카메라 유무를 검사하거나
bag 의 `metadata.yaml` 을 요구한다. "이미 토픽이 나오는 중" 인 경우를 상정하지 않았다.
Isaac 상대로는 런치를 직접 부른다:

```bash
cd /home/ingon/roboworld
source install/setup.bash
ros2 launch roboworld_perception perception.launch.py prompts:="blue plastic bar"
```

---

## 2. 인터페이스 계약 (실측값)

`ros2 topic echo` 로 직접 확인한 값이다. 추정치가 아니다.

| 항목 | 값 |
|---|---|
| RGB 인코딩 | `rgb8` |
| Depth 인코딩 | **`32FC1` (float32, 미터)** |
| 해상도 | 640x360 (조정 가능, §5 참조) |
| QoS | `RELIABLE` / `VOLATILE` — ROS 2 기본값, 기존 구독자와 그대로 매칭 |
| frame_id | `camera_color_optical_frame` |
| distortion | `plumb_bob`, d = [0,0,0,0,0] — 완전 무왜곡 |

내부 파라미터는 해상도에서 유도된다 (`fx = focalLength / hAperture x width`):

| 해상도 | fx = fy | cx | cy |
|---|---|---|---|
| 1280x720 | 645.7112944985439 | 640.0 | 360.0 |
| **640x360 (현재)** | **322.86** | **320.0** | **180.0** |

### 씬 기하 (자세 검증용 기준값)

| 항목 | 값 |
|---|---|
| 카메라 | `/Sensors/cam_belt`, z = 1.408, xy = (-0.315, -0.682) |
| 벨트 상면 | z = 0.408 → **카메라 높이 정확히 1.000 m** |
| 벨트 범위 | x = -1.565 ~ 0.935 (길이 2.5 m), 중심선 y = -0.682, 폭 0.554 |
| 작업물 | 200 x 55 x 55 mm, 9개. 초기 x = -1.450 ~ 0.550 (0.25 간격), z = 0.435 |
| 로봇 베이스 | (0, 0, 0.554), 도달 반경 0.85 |
| 벨트 진행방향 | 월드 +X (실측 확인: 속도 양수 → +X) |

Depth 실측 범위는 0.862 ~ 1.515 m 였다. 최솟값 0.862 는
`1.408 - 0.862 = 0.546` 으로 로봇 베이스 높이 0.554 와 일치한다 — 씬을 제대로
보고 있다는 교차검증이다.

---

## 3. 코드 변경 (1건)

`roboworld_perception/pipeline.py` 의 `img_to_np()` 에 `32FC1` 분기 추가.
기존 테스트 40개 전부 통과.

```python
elif msg.encoding == "32FC1":
    ch, dtype = 1, np.float32
...
if dtype is np.float32:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
```

**왜 여기서 mm 로 바꾸는가.** `geometry.py` 의 `mask_depth_to_points(...,
depth_scale=0.001)` 은 RealSense 규약(16UC1 밀리미터)을 가정한다. 변환하지 않고
연결하면 모든 물체가 카메라 1 mm 앞에 뭉친다. `img_to_np()` 한 곳에서 흡수하면
하위 단계가 전부 무수정이고, **bag(16UC1)과 Isaac(32FC1)을 같은 코드로** 쓸 수 있다.
bag 재생 시에는 이 분기를 아예 타지 않는다.

inf/NaN 을 0 으로 두는 것도 RealSense 규약("0 = 값 없음")에 맞춘 것이다.

---

## 4. 밟으면 하루 날리는 함정

실제로 다 밟았다. 근거를 같이 적는다.

### 4.1 WSL2 미러 네트워킹은 DDS 를 깨뜨린다 ← 가장 크게 날린 것

`.wslconfig` 에 `networkingMode=mirrored` 를 넣으면 **안 된다.** 기본 NAT 를 유지할 것.

미러 모드는 Windows 와 WSL 이 **같은 IP(10.0.4.7)를 공유**한다. DDS 는 "나한테
오려면 이 주소로 와" 하고 로케이터를 교환하는 프로토콜이라, 양쪽이 같은 주소를
주장하면 discovery 가 성립하지 않는다.

측정한 것:

| 검사 | 미러 모드 | NAT |
|---|---|---|
| 유니캐스트 UDP 양방향 | 통함 (실측) | 통함 |
| 멀티캐스트 239.255.0.1 | 안 넘어감 | — |
| FastDDS 초기피어 유니캐스트 우회 | **실패** | — |
| Hyper-V 방화벽 UDP 7400-7700 개방 | 효과 없음 | — |
| **기본 설정 그대로** | 실패 | **즉시 성공** |

NAT 로 되돌리면 FastDDS 프로파일도, 초기피어도, 방화벽 규칙도 **전혀 필요 없다.**
기본 멀티캐스트로 그냥 붙는다.

### 4.2 Windows 쪽 ROS 2 환경변수는 Isaac 실행 "전에" 넣어야 한다

```bat
set ROS_DISTRO=jazzy
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set PATH=%PATH%;<release>\exts\isaacsim.ros2.core\jazzy\lib
```

세 번째 줄이 없으면 Isaac 이 ROS 2 가용성을 판정하는
`isaacsim.ros2.core.check.exe` 가 **크래시**한다:

| PATH 에 `jazzy\lib` | 결과 |
|---|---|
| 없음 | `-1073741819` = 0xC0000005 ACCESS_VIOLATION |
| 있음 | exit 0 |

크래시하면 확장이 스스로 꺼지면서 `os.environ["ROS_DISTRO"]` 를 프로세스에
심어버린다. 그 뒤로는 재시작 전까지 **어떤 런타임 수리도 안 먹는다** — 환경변수가
있으면 Isaac 은 "사용자가 시스템 ROS 를 소싱했다" 고 판단해 내장 라이브러리를
쓰지 않기 때문이다 (`isaacsim.ros2.core/python/impl/extension.py:53-77`).

추가로 Windows 기본 배포판은 **humble** 이다 (같은 파일 :70). Jazzy 로 명시하지
않으면 WSL2 와 서로 못 본다.

`RoboWorld.usdc` 에 ROS 2 OmniGraph 노드(`ROS2Context`, `ROS2PublishJointState`,
`ROS2PublishTransformTree` 등)가 **이미 들어있다.** 즉 씬을 여는 순간 브리지가
스스로 뜨려 하므로, 위 환경변수가 없으면 사용자가 뭘 하기도 전에 이미 망가진다.

### 4.3 CameraInfo 의 K 는 해상도 변경을 따라오지 않는다

이미 만들어진 렌더프로덕트의 `inputs:width/height` 만 바꾸면 **이미지 크기는
바뀌는데 K 는 옛 값에 머문다.** 실측:

```
RGB 크기 : 640x360
K        : fx=645.71  cx=640.0  cy=360.0    <- 1280x720 기준값
```

`cx` 는 폭의 절반이어야 하므로 640 폭에는 320 이 맞다. 이 상태로 6D 자세를
계산하면 통째로 틀어진다. **해상도를 바꾸려면 그래프를 지우고 다시 만들어야 한다**
(Windows 쪽 `cellomni_ros2_rebuild.py` 의 `RES` 수정 후 재실행).

### 4.4 Depth 는 Z-depth 다 (RealSense 와 같은 규약)

`ROS2CameraHelper` 의 `type="depth"` 는 `DistanceToImagePlaneSD` AOV 로 간다
(`OgnROS2CameraHelper.py:52`). `distance_to_image_plane` 과 동일한 AOV 이므로
**광축 방향 Z 거리**이며 RealSense aligned depth 와 같은 의미다. 그대로 쓰면 된다.

`distance_to_camera` 는 유클리드 방사 거리라 다르다. 이걸 잘못 쓰면 화면
중앙은 맞고 **가장자리만 조용히 틀어진다** — 찾기 매우 어려운 버그.

### 4.5 정렬(alignment)은 공짜다

RGB 와 Depth 가 **같은 카메라 프림의 같은 렌더프로덕트에서 같은 틱에** 나온다.
따라서 픽셀 단위로 완벽 정렬되고 타임스탬프도 동일하다.
실장비에서 제일 골치 아픈 부분이 사라지므로 `ApproximateTimeSynchronizer` 가
항상 즉시 매칭된다. `aligned_depth_to_color` 라는 토픽 이름이 거짓말이 아니다.

### 4.6 (Isaac 쪽) 컨베이어 확장

벨트를 돌리려면 `isaacsim.asset.gen.conveyor` 확장이 켜져 있어야 한다.
꺼진 채로 씬을 열면 OmniGraph 가 `ConveyorNode` 를 인스턴스화하지 못해 벨트가
영원히 안 돈다. 게다가 브리지는 USD 쓰기가 성공했다고 판단해 **경고조차 남기지
않는다.** 컨베이어를 움직여 데이터를 뽑을 계획이면 알아둘 것.

---

## 5. 재현 절차

### Windows 쪽

1. 바탕화면 `cellomni 실행.bat` — 컨베이어 확장 + ROS 2 환경변수를 셋업해 Isaac 기동
2. `File > Open` → `Desktop\Collected_RoboWorld_cellomni\Collected_RoboWorld_cellomni\RoboWorld_cellomni.usdc`
3. Script Editor 에서 `exec(open(r"...\cellomni_ros2_rebuild.py").read())`
4. `Play (>)` — 재생 중에만 토픽이 나간다

해상도를 바꾸려면 `cellomni_ros2_rebuild.py` 맨 위 `RES` 를 고치고 3번 재실행.

### WSL 쪽

```bash
cd /home/ingon/roboworld
source install/setup.bash
ros2 launch roboworld_perception perception.launch.py prompts:="blue plastic bar"
```

확인용:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list --no-daemon
ros2 topic echo /camera/camera/color/camera_info --once --no-daemon
```

**`ros2` CLI 는 `--no-daemon` 을 붙여라.** 이 환경의 데몬이 자주 죽어서 붙이지
않으면 토픽 목록이 빈 채로 나온다 (실제로 이것 때문에 한참 헤맸다).
`ros2 topic hz` 는 `--no-daemon` 을 받지 않으므로 아예 못 쓴다 —
`/home/ingon/cellomni_tools/rate.py` 로 대체했다.

---

## 6. 현재 성능 한계

| 토픽 | 실측 속도 |
|---|---|
| CameraInfo (픽셀 없음) | 30 ~ 63 Hz |
| RGB (0.69 MB/장) | **1.5 ~ 2.0 Hz** |
| Depth | 1.0 ~ 1.9 Hz |

프레임 간격이 매우 불규칙하다 (최소 0.00초, 최대 2.7~4.6초).

**앱과 그래프는 멀쩡하다.** CameraInfo 가 30 Hz 이상으로 나오는 것이 증거다 —
`OnPlaybackTick` 이 정상 속도로 돌고 있다. 병목은 오직 **이미지 픽셀을 GPU 에서
꺼내 DDS 로 흘리는 구간**이다. Isaac 로그에 근거가 있다:

```
OgnSdPostRenderVarToHost : rendervar copy from texture directly to host buffer
is counter-performant. Please use copy from texture to device buffer first.
```

시도했지만 효과가 미미했던 것:

- `net.core.rmem_max` 212992 → 16777216 : 1.5 → 2.0 Hz (거의 차이 없음)
- FastDDS 소켓 버퍼 8 MB 프로파일 : 유의미한 개선 없음
- 해상도 1280x720 → 640x360 : 0.2 → 1.5 Hz (**이건 효과 있었음, 7배**)

즉 네트워크가 아니라 Isaac 내부 경로가 병목이다. 아직 안 해본 것은 §7 에.

**파이프라인 검증에는 2 Hz 로도 충분하다.** `detect_interval=5` 로 SAM 을 띄엄띄엄
돌리는 구조이므로 정적 검출·자세 계산은 지금 바로 평가할 수 있다.
다만 **추적 안정성·시간적 평활 평가는 이 속도로는 의미가 없다.**

---

## 7. 미해결 / 다음에 할 것

### 7.1 TF 트리 (확인 필요)

`publish_world_tf=True` 는 "카메라가 1 m 위에서 아래를 본다" 는 world TF 를
**RealSense TF 트리의 뿌리인 `camera_link` 위에** 붙이도록 돼 있다
(`perception_node.py:55-59` 주석). 그런데 Isaac 은 `camera_color_optical_frame`
하나만 내보내고 `camera_link` 가 없다. 그대로 두면 TF 가 끊기거나 부모가
어긋날 수 있다. **Isaac 상대로 실행할 때 이 파라미터 동작을 확인할 것.**

높이는 우연히 정확히 맞는다 (1.408 - 0.408 = 1.000). 다만 카메라 XY 가
(-0.315, -0.682) 이고 로봇 베이스가 (0, 0, 0.554) 이므로, 로봇 좌표계와
정합하려면 world TF 에 이 오프셋을 반영해야 한다.

### 7.2 프레임률 개선 (안 해본 것)

- `--/app/settings/fabricDefaultStageFrameHistoryCount=3`
  (`isaacsim.ros2.core/config/extension.toml` 의 테스트 인자에 등장 — ROS 2
  브리지 테스트가 이걸 켜고 돈다. 관련 있어 보이지만 미검증)
- `exts."isaacsim.ros2.bridge".publish_multithreading_disabled`
- `exts."isaacsim.ros2.bridge".publish_queue_thread_sleep_us` (기본 1000)
- `type="rgb_h264"` 압축 발행 (구독 측 수정 필요)
- SDG 를 device-buffer 경로로 (`SdPostRenderVarTextureToBuffer`)

### 7.3 프롬프트 튜닝

씬의 작업물은 **파란 플라스틱 블록 3종**(Guide / Spacer / End_Stopper)이다.
치수는 셋 다 200x55x55 mm 로 동일하고 **구멍 패턴만 다르다** — 형상으로는
구분이 안 되고 텍스처로만 구분된다. 텍스트 프롬프트 검출의 난이도가 bag 데이터와
전혀 다르므로 재튜닝이 필요하다.

`PROMPT_ALIASES` 의 `"블록" → "pink foam block"` 은 색이 안 맞는다.
`"blue plastic bar"`, `"blue block"` 등 영어로 직접 주는 편이 낫다.
문턱값도 `score_threshold:=0.25` 정도로 낮춰 시작할 것.

### 7.4 정답값 활용 ← 이 연결의 진짜 가치

`PROJECT_CONTEXT.md` 에 이렇게 적혀 있다:

> test3만으로는 실제 정답 자세와의 절대 오차보다 반복성과 시간적 안정성을
> 주로 평가할 수 있다

**Isaac 은 정답을 알고 있다.** rosbag 으로는 불가능한 것들:

| 얻을 수 있는 것 | 현재 | Isaac 연결 후 |
|---|---|---|
| 블록 9개의 정확한 6D 포즈 | 없음 | 위치·자세 절대 오차 (mm, deg) |
| `semantic_segmentation` 마스크 | 없음 | SAM 마스크 IoU 정량화 |
| 가려짐 비율 | 추정 | 정답 occlusion % |
| 조명·재질·배치 변화 | 촬영해야 함 | 도메인 랜덤화로 무한 생성 |

`output/` 에 `occ_fixed2`, `occ_measure5`, `depth_occ5`, `escape5` 등 가려짐
실험이 잔뜩 쌓여 있는데, **가려짐 정답값을 공짜로 얻는 것**만으로도 연결 가치가
충분하다.

정답 포즈를 꺼내는 경로는 두 가지다:

1. Windows 의 HTTP 브리지 `GET /api/state` — 블록 9개의 월드 좌표를 JSON 으로
   준다. 현재 회전은 안 주므로 추가가 필요하다. (지금은 카메라 자원 경쟁 때문에
   꺼둔 상태. `cellomni_start.py` 로 되살릴 수 있다)
2. `ROS2CameraHelper` 의 `type="semantic_segmentation"` / `bbox_3d` 를 추가
   발행 — 같은 렌더프로덕트를 공유하므로 렌더 비용이 두 배가 되지 않는다.
   ROS 2 로 바로 받을 수 있어 이쪽이 더 깔끔하다.

---

## 8. 파일

### WSL (이 저장소)

| 경로 | 내용 |
|---|---|
| `roboworld_perception/pipeline.py` | `img_to_np()` 에 32FC1 분기 (§3) |
| `/home/ingon/cellomni_tools/rate.py` | 토픽 수신율·K·depth 범위 측정기 |

`/home/ingon/fastdds_unicast.xml`, `fastdds_bigbuf.xml` 은 미러 네트워킹
삽질의 잔재다. NAT 에서는 불필요하므로 지워도 된다.

### Windows (참고용, 이 저장소 밖)

| 경로 | 내용 |
|---|---|
| `Desktop\cellomni 실행.bat` | 컨베이어 확장 + ROS 2 환경변수 셋업 후 Isaac 기동 |
| `...\cellomni_ros2_rebuild.py` | ROS 2 카메라 그래프 생성 (해상도는 `RES`) |
| `...\cellomni_start.py` | HTTP 브리지 (웹 콘솔 + 정답 좌표 API) |
| `...\cellomni_ros2_probe.py` | Isaac 내부에서 자기 토픽 확인 |

`.bat` 파일은 **ASCII 로만** 써야 한다. cmd.exe 가 시스템 ANSI 코드페이지(한국어
Windows 는 CP949)로 읽기 때문에 UTF-8 한글 주석은 깨지고, 깨진 바이트가 명령으로
실행된다.
