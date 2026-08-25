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

> **⚠ 미해결 — 주점이 반픽셀 어긋난 것으로 보인다 (2026-08-25, 브리지 확인 요청 중)**
>
> OpenCV/`image_geometry` 규약에서 정수 픽셀 좌표 = **픽셀 중심**이므로 대칭
> 절두체의 주점은 `(W−1)/2` 다 — 640 폭이면 **319.5 / 179.5**. 근거는
> `getDefaultNewCameraMatrix(centerPrincipalPoint=true)` 가 쓰는
> `cx = (imgsize.width-1)*0.5` (OpenCV 소스 확인).
>
> **Isaac 은 물리 주점이 아니라 해상도마다 `W/2` 를 재계산하는 것으로 보인다.**
> 픽셀 중심 규약이면 1280 으로 올릴 때 `2×(320+0.5) − 0.5 = 640.5` 여야 하는데
> 위 표는 **640.0** 이다. fx 도 정확히 2 배(645.7112944985439 / 322.85564724927195
> = 2.0). 즉 캘리브레이션 값이 아니라 **공식**이다. `D` 도 전부 0 이다.
>
> **영향**: 중심 추정이 x·y 각각 **~1.46mm** 밀린다(0.5px, z=0.945m, fx=322.86).
> 부호는 −x, −y 방향. **extent 에는 사실상 영향이 없다** — cx 오차는 평행이동이
> 아니라 전단(shear)인데, 풋프린트 점들이 전부 상면 한 장(거의 같은 z)이라
> 균일 이동이 되어 `minAreaRect` 의 `(su, sv)` 가 안 변한다. 잔여 상한은
> Δz=55mm 점군에서 **85µm** 다. 자세는 축당 0.089° 기운다.
>
> **RealSense bag(test2~5)은 무관하다** — 적합값이고 `D` 도 0 이 아니다(공장
> 캘리브레이션). 이 건은 Isaac 전용이다.
>
> **우리 코드에 −0.5 보정을 넣지 말 것.** 브리지가 고쳐지면 곧바로 반대로
> 틀어지고, RealSense 에는 적용하면 안 되는 보정이라 조건 분기가 생긴다.
> `geometry._backproject` 는 `camera_info` 의 K 를 규약대로 소비하고 있어
> **우리 쪽에 자기모순은 없다** — K 만 맞으면 맞는다.
>
> **실측으로는 못 가른다**: 씬 표가 작업물의 y 를 명시하지 않아, 관측된
> `y_optical` 평균 −0.5mm 는 `cy=179.5`(예측 −1.47mm)로도 `cy=180.0`(예측 0mm)
> 로도 평범한 배치 오프셋으로 설명된다. **판별력 없는 대조다.**
>
> **우선순위 낮음.** 다만 **절대 위치를 로봇 파지에 쓰기 시작하면 지워지지 않는
> 2mm 오프셋**이라 그 전에는 정리해야 한다.
>
> **브리지에 물을 것**: Isaac 의 `camera_info` 발행기가 주점을 `W/2, H/2` 로
> 내는지 `(W−1)/2, (H−1)/2` 로 내는지, 그리고 렌더러 투영행렬이 대칭 절두체 +
> 픽셀 중심 반정수인지.

### 씬 기하 (자세 검증용 기준값)

> **이 표가 씬 정답이다.** 광학 좌표 환산은 아래 "카메라 → 월드 변환" 절의
> `z_optical = 1.408 − z_world` 를 쓴다. 자주 쓰는 값:
> **벨트면 1000.0mm · 블록 상면 945.5mm · 물체 중심 973.0mm · 벨트 위 가시
> 높이 54.5mm**(블록이 0.5mm 잠겨 있다: 하면 0.4075 vs 벨트 상면 0.408).
> 2026-08-24 까지 이 표를 못 보고 **depth 판독에서 "정답" 을 파생해 쓴 문서가
> 있었다** — 경위는 [`belt_plane_2026-08-21.md`](belt_plane_2026-08-21.md)
> 맨 위 정정 블록.

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

### 카메라 → 월드 변환 (정답 대조용)

USD 에서 읽은 `/Sensors/cam_belt` 의 월드 자세는 **회전이 전혀 없다**
(회전행렬 = 단위행렬). 정확히 수직 하방을 본다.

```
위치      (-0.3150, -0.6820, 1.4080)
보는 방향  월드 (0, 0, -1)
```

검출 결과는 `camera_color_optical_frame` 으로 나온다. ROS optical 규약
(x 오른쪽, y 아래, z 전방) 이므로 월드 변환이 이렇게 단순해진다:

```
world_x = -0.315 + x_optical
world_y = -0.682 - y_optical
world_z =  1.408 - z_optical
```

검증: 블록 윗면을 검출했을 때 `z_optical = 0.945` → `world_z = 0.463`.
실제 윗면은 `0.435 + 0.055/2 = 0.4625` 이므로 **오차 0.5 mm**. 깊이 경로는
단위·규약 모두 올바르다.

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

### 4.3b 그래프를 지워도 ROS 퍼블리셔는 안 죽는다 ← 조용히 절반으로 틀어짐

§4.3 때문에 그래프를 재생성하면, **옛 퍼블리셔가 살아남아 옛 해상도의 K 를
계속 뿌린다.** `DeletePrims` 는 OmniGraph 프림만 지우고 SDG 라이터가 만든
ROS 2 퍼블리셔는 정리하지 않는다. 실측:

```
/camera/camera/color/camera_info    Publisher count: 2
   _Render_PostProcess_SDGPipeline_Replicator_NodeWriterWriter_01   <- 옛 그래프 (1280x720)
   _Render_PostProcess_SDGPipeline_Replicator_NodeWriterWriter_04   <- 새 그래프 (640x360)
```

구독자는 둘 중 **먼저 온 것을 캐시**한다. 잘못 걸리면 이미지는 640 폭인데
K 는 `fx=645.71 / cx=640` 이라 계산된 위치·크기가 **정확히 절반**이 된다.
실제로 이 상태에서 200x55 mm 블록이 95x24 mm 로 나왔다.

에러도 경고도 없다. 반드시 확인할 것:

```bash
ros2 topic info /camera/camera/color/camera_info --verbose --no-daemon
# Publisher count 가 1 이어야 한다
```

**해상도를 바꾸려면 Isaac Sim 을 껐다 켜고 그래프를 한 번만 만들어야 한다.**
같은 세션에서 재생성하면 안 된다.

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

### 6a. GPU 경쟁 — 인식 노드 자신이 프레임률을 3.6배 깎는다

RTX 4070 Ti 12 GB 한 장에서 Isaac 의 RTX 렌더링과 SAM3 추론이 경쟁한다.
A/B 실측 (다른 조건 동일):

| | perception_node 켜짐 | 꺼짐 |
|---|---|---|
| GPU 메모리 | 10.3 GB / 12.3 | **7.9 GB** |
| RGB | 0.70 Hz | **2.55 Hz** |
| Depth | 0.25 Hz | 1.10 Hz |

즉 **"라이브로 붙여서 보는" 구성 자체가 프레임률의 주된 손실원**이다.
브리지(HTTP 콘솔)도 앱 틱을 30 → 18 Hz 로 떨어뜨리므로 같이 끄는 게 좋다.

**권장 작업 방식 — 녹화 후 오프라인 처리.**
정확도·가려짐 평가가 목적이라면 라이브로 붙일 이유가 없다:

1. 인식 노드를 끈 상태로 Isaac 토픽을 `ros2 bag record` 로 녹화 (최대 속도로)
2. 녹화된 bag 에 SAM 을 오프라인으로 돌린다 (`scripts/run_offline.py` 가 이미 있다)

GPU 경쟁이 사라지므로 Isaac 은 최대 속도로 렌더하고 SAM 은 전력으로 추론한다.
게다가 같은 데이터를 반복 재현할 수 있어 파라미터 비교에 유리하다.
라이브 연결은 "눈으로 확인" 용도로만 쓰는 것을 권한다.

시도했지만 효과가 미미했던 것:

- `net.core.rmem_max` 212992 → 16777216 : 1.5 → 2.0 Hz (거의 차이 없음)
- FastDDS 소켓 버퍼 8 MB 프로파일 : 유의미한 개선 없음
- 해상도 1280x720 → 640x360 : 0.2 → 1.5 Hz (**이건 효과 있었음, 7배**)

즉 네트워크가 아니라 Isaac 내부 경로가 병목이다. 아직 안 해본 것은 §7 에.

**파이프라인 검증에는 2 Hz 로도 충분하다.** `detect_interval=5` 로 SAM 을 띄엄띄엄
돌리는 구조이므로 정적 검출·자세 계산은 지금 바로 평가할 수 있다.
다만 **추적 안정성·시간적 평활 평가는 이 속도로는 의미가 없다.**

---

## 6b. 첫 검증 결과 (2026-08-13 실측)

`prompts:="blue plastic bar"`, `score_threshold:=0.25`, `max_per_prompt:=12`,
640x360. 벨트 정지, 블록 9개 초기 배치.

**텍스트 프롬프트 검출은 합성 영상에서 잘 먹는다** — score 0.93~0.94.
도메인 갭 걱정은 기우였다.

정답 대비 절대 오차 (월드 좌표, 한 프레임):

| 월드 x | 정답 x | dx | dy | dz | 마스크 크기 L x W | score |
|---|---|---|---|---|---|---|
| -0.950 | -0.950 | **-0.1 mm** | 0.0 | 0.5 | 193.2 x 47.7 | 0.93 |
| -0.700 | -0.700 | **+0.2 mm** | 0.0 | 0.5 | 196.3 x 49.3 | 0.93 |
| -0.450 | -0.450 | **+0.3 mm** | 0.1 | 0.5 | 193.8 x 49.6 | 0.94 |
| -0.204 | -0.200 | -3.8 mm | 0.1 | 0.5 | 194.1 x 50.5 | 0.93 |
| +0.026 | +0.050 | **-24.2 mm** | 0.1 | **-29.5** | **151.4** x 50.9 | 0.93 |
| +0.275 | +0.300 | **-24.6 mm** | 0.1 | **-29.5** | **155.1** x 52.3 | 0.93 |

- **가려지지 않은 블록은 밀리미터 이하**로 맞는다 (dx <= 0.3 mm, dz 0.5 mm).
  ⚠ **이 dz 는 물체 중심이 아니라 상면 기준이다** — 아래 주 참고.
- **y 오차는 전부 0.1 mm 이내** — 카메라가 벨트 중심선 바로 위라 관측이 가장 좋은 축.
- 오른쪽 두 블록(x = +0.05, +0.30)만 dx -24 mm, dz -30 mm 로 크게 튄다.
  **로봇 팔이 그 위를 덮고 있어** 마스크가 194 mm → 151/155 mm 로 잘렸다.
  즉 **가려짐이 자세 추정을 어떻게 망가뜨리는지가 정량적으로 찍힌다.**
  `output/occ_*` 실험에 그대로 이어붙일 수 있는 데이터다.
- 크기는 정답 200 x 55 mm 대비 193~196 x 48~52 mm. 마스크가 변당 2~3 mm 안쪽으로
  들어온다 (SAM 마스크의 체계적 수축).
- **높이는 항상 0.000 m** — 위에서 내려다보므로 윗면만 보이고 점군이 평면이다
  (`qhull: initial hull is narrow` 경고의 정체). top-down 단일 시점의 구조적 한계.

> **주(2026-08-24) — "밀리미터 이하" 를 자세 정확도로 인용하지 말 것.**
> 그 `dz 0.5 mm` 의 정답은 §2 가 밝히듯 **상면**(`0.435 + 0.055/2 = 0.4625`)이다.
> 그리고 바로 위 항목이 적은 대로 이 경로는 **높이가 항상 0.000 m** — 점군이
> 평면이라 **OBB 중심이 상면에 얹혀 있었다.** 따라서 **물체 중심(z = 0.435)
> 기준으로는 +27.5mm(= 55/2) 틀린다.**
>
> 이 줄이 틀린 것은 아니다 — 상면 대비로는 실제로 0.5mm 다. 틀리는 것은
> **그것을 자세 정확도로 읽는 것**이다. **로봇이 파지에 쓰는 것은 상면이 아니라
> 물체 중심이다.**
>
> 이것이 정확히 [`belt_plane_2026-08-21.md`](belt_plane_2026-08-21.md) 가 평면
> 구속을 도입한 이유다 — 거기 첫 줄이 *"무구속 OBB 는 두께가 붕괴하고 중심이
> 상면에 얹힌다"* 라고 적는다. 현재 파이프라인은 `use_belt_plane` **기본 켜짐**
> 이라 중심이 제자리로 온다: 씬 정답 973.0mm 대비 **+0.8~+1.2mm**.

한 프레임에서 6개가 잡혔고 오버레이의 tracker 는 9개를 유지했다. 프레임이
1.5~2 Hz 로 띄엄띄엄 오는 탓에 프레임별 검출 수가 흔들린다 (§6).

> `max_per_prompt` 기본값이 **1** 이라 그냥 실행하면 한 개만 나온다.
> `perception.launch.py` 는 이 파라미터를 노출하지 않으므로, 여러 개를 보려면
> 설치된 실행파일을 직접 부른다 (이 환경에는 `ros2 run` 이 없다):
>
> ```bash
> ./install/roboworld_perception/lib/roboworld_perception/perception_node \
>   --ros-args -p prompts:="blue plastic bar" -p max_per_prompt:=12 \
>   -p score_threshold:=0.25 -p publish_world_tf:=false
> ```

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
