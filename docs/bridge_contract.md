# Isaac Sim → ROS 2 브리지 계약 (2026-08-19)

이 문서는 **비전 시스템(WSL2 ROS 2 Jazzy)이 Isaac Sim 쪽에 무엇을 기대해도
되는지**와 **그 대가로 무엇을 지켜야 하는지**를 적는다. 브리지 내부 사정과
왜 이렇게 됐는지는 `isaac_sim_stability_2026-08-14.md` §9 에 있다.

---

## 1. 비전 쪽이 반드시 해야 할 것 (안 하면 아무것도 안 온다)

### 1.1 전송 설정 — 노드를 띄우는 모든 경로에 넣을 것

```bash
export FASTDDS_BUILTIN_TRANSPORTS='LARGE_DATA?max_msg_size=190KB&sockets_size=200KB&non_blocking=true&tcp_negotiation_timeout=50'
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

`~/.bashrc` 에 이미 있지만 **대화형 셸에서만 읽힌다.** 스크립트,
`wsl -e bash -lc`, launch 파일, systemd 로 뜬 프로세스는 못 받는다.

한쪽만 설정되면 Isaac=TCP / 비전=UDP 가 되어 **아예 안 붙는다.**
증상은 "토픽이 `/rosout` 만 보인다" 이고, Isaac 은 멀쩡히 돌고 있다.

확인:
```bash
tr '\0' '\n' < /proc/$(pgrep -f perception_node)/environ | grep FASTDDS
```
비어 있으면 그 프로세스는 UDP 다.

**launch 파일에 직접 박아 넣는 것을 권한다.** env 를 잊는 사고가 이미 두 번 났다.

### 1.2 `use_sim_time`

Isaac 이 `/clock` 을 발행한다(약 85 Hz, `resetOnStop=True`). 시간을 쓰는
**모든 노드**에 `use_sim_time=true` 를 줄 것. rviz2 포함.

```python
Node(..., parameters=[{'use_sim_time': True}])
```

**일부만 켜면 안 된다** — 켠 노드와 안 켠 노드가 다른 시간축을 쓰게 되어
TF 와 message_filters 가 조용히 깨진다.

### 1.3 Isaac 을 재시작하면 비전 노드도 재시작할 것

Isaac 재기동 시 sim time 이 0 으로 되감기고 ROS2 퍼블리셔가 새로 만들어진다.
그 전에 떠 있던 구독자는 재매칭되지 않는다. 증상은 "토픽은 보이는데 입력 없음".

---

## 2. 브리지가 보장하는 것 (2026-08-19 실측)

### 2.1 토픽

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | rgb8, 640x360 |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | **32FC1, 단위 미터** |
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` | K 아래 참고 |
| `/clock` | `rosgraph_msgs/Clock` | 약 85 Hz |

`frame_id` 는 전부 `camera_color_optical_frame`.

### 2.2 성능

| 항목 | 값 | 조건 |
|---|---|---|
| color / depth 발행률 | **23 ~ 27 Hz** | perception 함께 구동 시 (실사용) |
| | 39 Hz | 브리지만 단독 |
| 최대 공백 | **0.03 초** | 6 라운드 전부 |
| color-depth 스탬프 | **완전히 동일** | 시간차 최소=중앙=최대=0.0000 초 |
| `slop=0.05` 로 붙는 비율 | **100%** | 6 라운드 x 14,000 장 이상 |

**color 와 depth 는 같은 렌더 틱에서 나오므로 스탬프가 비트 단위로 같다.**
따라서 `ApproximateTimeSynchronizer` 의 `slop` 은 기본값 0.05 로 충분하다.
`isaac.launch.py` 프리셋도 0.05 로 되돌려 놨다.

**slop 을 크게 두지 말 것** — 서로 다른 순간의 프레임을 잘못 묶는다.
다시 "입력 없음" 이 나오면 slop 을 올리기 전에 **1.1 의 전송 설정부터 확인**할 것.

### 2.3 카메라 내부 파라미터

```
width x height  640 x 360
fx = fy         322.85564724927195
cx              320.0
cy              180.0
distortion      plumb_bob, d = [0,0,0,0,0]  (왜곡 없음)
```

**해상도를 바꾸면 K 도 바뀐다.** 브리지 쪽에 요청할 것 — 비전 쪽에서
렌더프로덕트만 고치면 K 가 옛 값에 남아 6D 자세가 통째로 틀어진다.

### 2.4 TF

```
world → camera_link → camera_color_optical_frame
```

`publish_optical_tf=true` 로 노드가 직접 발행한다(RealSense 드라이버가 없으므로).
`rviz/perception.rviz` 의 `Fixed Frame` 은 `world` 다.

---

## 3. 진단할 때 헷갈리는 것

### 3.1 `ros2 topic list` 가 거짓으로 빈 목록을 준다

TCP 전환 후 `--no-daemon` 목록이 `/rosout` 만 보여주는 일이 있다. 새 참가자가
디스커버리를 끝내기 전에 타임아웃되는 것으로 보인다. **데이터는 흐른다.**

같은 시점 실측:
```
ros2 topic list --no-daemon           -> /rosout 뿐
ros2 topic hz /perception/detections  -> 4.0 Hz
직접 구독                              -> 89 사이클 622 검출
```

**"토픽이 없다" 로 보이면 `topic hz` 나 직접 구독으로 다시 확인할 것.**

### 3.2 `ros2 topic hz` 는 `--no-daemon` 을 안 받는다

`list` / `echo` 는 받는다. `hz` 에 붙이면 usage 에러만 내고 조용히 죽는다.

### 3.3 "입력 없음" 이 뜰 때 확인 순서

1. 전송 설정이 그 프로세스에 있는가 (1.1)
2. Isaac 이 재시작됐는가 (1.3)
3. `topic hz` 로 실제 흐름 확인 (3.1)
4. 그래도면 브리지 쪽에 문의

**VRAM 을 먼저 의심하지 말 것.** 2026-08-19 에 그러다 반나절을 썼다.

---

## 4. 브리지 쪽에 요청해야 하는 것

비전 쪽에서 바꿀 수 없고 Isaac 쪽 작업이 필요한 것:

- **해상도 변경** (K 가 함께 바뀌므로 그래프 재생성 필요)
- 카메라 위치/각도 변경
- 발행률 상향 (현재 23~27 Hz 가 상한, 그 이상은 Isaac 내부 프로파일링 필요)
- 새 카메라 추가

---

## 5. 알려진 한계

- **발행률 23~27 Hz** 가 현재 상한이다. 더 올리려면 Isaac 의 이미지 반출
  경로(`OgnROS2PublishImage`)를 프로파일링해야 한다. 미착수.
- 크래시가 2026-08-19 에 한 번 났다(depth 텍스처 회수 경합). 원인인 발행 큐를
  제거해서 구조적으로는 해소됐으나 **장시간 검증은 8 분뿐**이다.
- `/clock` 간격이 가끔 음수로 찍힌다(스탬프 역행). 영향은 확인하지 않았다.
