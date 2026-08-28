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

`isaac.launch.py`는 이 값을 기본 `true`로 선언하고, base launch가 동일한
bool 값을 perception 노드와 RViz에 함께 전달한다. 일반
`perception.launch.py`의 기본값은 `false`다.

수동 실행 또는 다른 launch를 쓸 때도 둘을 함께 맞춘다:

```bash
ros2 launch roboworld_perception perception.launch.py use_sim_time:=true
```

**일부만 켜면 안 된다** — 켠 노드와 안 켠 노드가 다른 시간축을 쓰게 되어
TF 와 message_filters 가 조용히 깨진다.

### 1.3 Isaac 을 재시작하면 비전 노드 상태를 확인할 것

Isaac 재기동 시 sim time 이 0 으로 되감기고 ROS2 퍼블리셔가 새로 만들어진다.
비전 파이프라인은 0.5초를 넘는 시간 역행을 감지하면 track·filter·광학흐름
상태를 자동으로 비워, 이전 run의 pose를 새 시간축에서 발행하지 않는다.

다만 그 전에 떠 있던 구독자가 재매칭되지 않는 경우는 별개다. 증상은
"토픽은 보이는데 입력 없음"이며, 이때는 비전 노드도 재시작한다.

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

비전 노드는 `input_qos_depth=1`로 각 RGB/depth 스트림에서 최신 메시지 하나만
보관한다. SAM 추론이 입력보다 느릴 때 오래된 프레임을 순서대로 처리하는 대신
지연을 버리는 정책이다. 정확한 end-to-end 지연은 실제 로봇 투입 전에 별도로
측정해야 한다.

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

브리지가 보장하는 프레임은 `camera_color_optical_frame` 뿐이다. 비전 노드는
기본값으로 `world` TF를 만들지 않는다. `world → camera_link`는 카메라 설치와
hand-eye 보정으로 얻은 변환을 로봇/셀 TF 트리에서 발행해야 한다.

Isaac에서 RealSense 드라이버가 없다면 `publish_optical_tf=true`로
`camera_link → camera_color_optical_frame`만 보완할 수 있다. 이때도
`world → camera_link`는 Isaac의 실제 카메라 pose 또는 캘리브레이션 값으로
별도 발행한다. `publish_world_tf=true`는 과거 RViz 편의용 공칭 1m 변환이며,
경고와 함께만 발행된다. 로봇 좌표 변환에 사용하면 안 된다.

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
3. **퍼블리셔 수를 먼저 본다** — `ros2 topic info <토픽>` 의 `Publisher count`
4. `topic hz` 로 실제 흐름 확인 (3.1)
5. 그래도면 브리지 쪽에 문의

**`topic list` 에 이름이 보인다고 발행자가 있는 것이 아니다.** 우리 쪽 노드가
**구독만 해도** 그 토픽은 목록에 뜬다. 2026-08-24 에 이걸 몰라서 "토픽은 보이는데
데이터가 안 온다"를 전송 설정 문제로 오인하고 시간을 썼다 — 실제로는 Isaac 의
ROS 2 브리지가 죽어 **퍼블리셔가 0개**였다.

**원인을 "방화벽"으로 적었던 것은 틀렸다(같은 날 정정).** 방화벽·포트·네트워크와
무관하고, Windows Smart App Control 이 **커널 레벨에서 DLL 로딩을 차단**한
것이다 — 상세와 진단 절차는 바로 아래 §3.4. 네트워크 쪽으로 접근하면 아무것도
안 나온다.

```bash
ros2 topic info /camera/camera/color/image_raw
# Publisher count: 0  ← 브리지가 죽었다. 비전 쪽에서 고칠 수 있는 것이 없다
# Publisher count: 1  ← 발행자는 있다. 그때부터 전송 설정(1.1)을 의심한다
```

`Publisher count: 0` 이면 **위 1~2 번을 확인할 필요가 없다.** 바로 브리지 쪽 문의다.

**VRAM 을 먼저 의심하지 말 것.** 2026-08-19 에 그러다 반나절을 썼다.

---

### 3.4 퍼블리셔가 0개면 Windows 가 브리지를 차단한 것이다 (2026-08-24)

토픽 목록에 이름이 보이는 것만으로는 아무것도 증명되지 않는다. 이름은
**구독자만 있어도** 뜬다. 반드시 발행자 수를 볼 것:

```bash
ros2 topic info -v /camera/camera/color/image_raw | head -5
#   Publisher count: 0     <- Isaac 이 발행하지 않고 있다
```

0 이면 Play 여부나 전송 설정 문제가 아니다. **Windows Smart App Control** 이
Isaac 번들 ROS 2 바이너리를 차단했는지부터 볼 것. NVIDIA 가 넣은 rmw/rcl DLL 에
Microsoft 가 신뢰하는 서명이 없어서 커널이 로딩을 막는다.

Windows 쪽 확인 (관리자 권한 없이도 읽힌다):

```powershell
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy').VerifiedAndReputablePolicyState
#   0=꺼짐   1=적용중(차단)   2=평가모드

Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-CodeIntegrity/Operational'; Id=3077} |
  Where-Object { $_.Message -match 'isaacsim' }
```

Isaac 로그에는 이렇게 남는다:

```
[ext: isaacsim.ros2.core-1.9.4] Failed to startup python extension.
OSError: [WinError 4551] 애플리케이션 제어 정책에서 이 파일을 차단했습니다
/ROS2_Camera/Info: Assertion raised in compute - No module named 'sensor_msgs'
/ROS2_Clock/PublishClock: Unable to create ROS2 node, please check that namespace is valid
```

주의할 점 — 이 경우 §1.1 전송 설정도 §4.2 환경변수도 **전부 정상인데** 안 된다.
2026-08-24 에 ROS_DISTRO=jazzy, RMW_IMPLEMENTATION=rmw_fastrtps_cpp, PATH 에
jazzy\lib 까지 모두 들어간 상태에서 막혔다. 환경변수를 의심하는 데 시간을 쓰지 말 것.

Smart App Control 은 예외 목록이 없다. 끄는 방법뿐이고 한 번 끄면 Windows 를
재설치하기 전까지 다시 켤 수 없다.

**SAC 는 예고 없이 평가 모드에서 차단 모드로 승격한다.** 평가 모드에서는
로그만 남기고 통과시키므로, 승격 전까지는 아무 문제가 없다. 승격 시점에
코드도 설정도 그대로인 채로 갑자기 막힌다. 실제 기록(이 PC):

    08-13  빌드 직후         이벤트 있음, 그러나 통과 (평가 모드)
    08-19 ~ 08-21           통과 — 39 Hz 로 정상 동작
    08-24                   차단 — 승격됨

승격 여부는 레지스트리에 남는다:

```powershell
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' |
  Select-Object VerifiedAndReputablePolicyState, SAC_PreviousState, SAC_EnforcementReason
#   VerifiedAndReputablePolicyState : 1   <- 지금 차단 중
#   SAC_PreviousState               : 2   <- 원래 평가 모드였다  = 스스로 승격했다
```

`SAC_PreviousState` 가 2 인데 현재 상태가 1 이면 이 경우다. 정책 갱신
이벤트(3099)를 찾아봐야 안 나온다 — 정책이 바뀐 게 아니라 모드가 바뀐 것이다.

끄고 나면 `VerifiedAndReputablePolicyState` 가 0 이 되고 `SAC_PreviousState`
가 1 로 바뀐다. **값이 0 이어도 재부팅 전에는 적용되지 않는다.**

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

---

## 6. 비전이 내보내는 것 — 소비자 계약 (2026-08-25)

§1~5 는 **비전이 받는 것**의 계약이다. 이 절은 **비전이 내보내는 것** 중
소비자가 오해하기 쉬운 것들이다.

발행 토픽 전부:

| 토픽 | 타입 | 소비자 |
|---|---|---|
| `/perception/detections` | `vision_msgs/Detection3DArray` | **로봇이 읽는 것** |
| `/perception/markers` | `visualization_msgs/MarkerArray` | RViz |
| `/perception/debug_image` | `sensor_msgs/Image` | 사람 |
| `/perception/points` | `sensor_msgs/PointCloud2` | **기본 꺼짐** — §6.3 |
| `/perception/status` | `diagnostic_msgs/DiagnosticArray` | §6.1 |

**앞 넷은 회수 계약을 함께 진다** — `_withdraw_output()` 이 단일 정의이고,
리셋·워치독에서 같은 stamp 로 같이 나간다. 하나라도 빠지면 소비자가 옛 것을
붙잡은 채 남는다(2026-08-25 에 워치독·프롬프트 교체 두 경로가 정확히
그렇게 빠져 있었다).

### 6.1 `/perception/status` — 입력 건강 하트비트 (신규)

`diagnostic_msgs/DiagnosticArray`, **1 Hz**(`perception_node.py` 의
`_watchdog` → `_publish_status`, `create_timer(1.0, ...)`). 항상 status 하나만
담는다:

```
status[0].name        = "roboworld_perception/input"
status[0].hardware_id = "rgbd_camera"
```

**레벨 규칙** — `input_health.classify_input_health()` 가 정한다. **위에서부터
먼저 걸리는 것이 이긴다**:

| 조건 | level | message |
|---|---|---|
| 입력 계약 위반이 잡혀 있음 | **ERROR**(2) | `input contract invalid` |
| `camera_info` 아직 무효(K 없음) | **ERROR**(2) | `waiting for valid camera_info` |
| 프레임을 **한 번도** 못 받음 | **WARN**(1) | `waiting for RGB-D frames` |
| 마지막 프레임이 `stale_timeout` 초과 | **WARN**(1) | `RGB-D input stale` |
| 그 외 | **OK**(0) | `RGB-D input healthy` |

**KeyValue 키 (10개, 항상 이 순서로 전부 실린다)**:

| 키 | 값 |
|---|---|
| `camera_info_valid` | `true` / `false` |
| `input_contract_valid` | `true` / `false` |
| `last_frame_age_s` | 초, 소수 3자리. 프레임을 한 번도 못 받았으면 **`never`** |
| `last_processing_duration_ms` | ms, 소수 1자리. 아직 없으면 **`never`** |
| `out_of_order_frame_drops` | 정수 누적 카운터 |
| `stale_timeout_s` | 초, 소수 1자리 |
| `camera_frame` | `camera_info` 가 보고한 광학 프레임 이름 |
| `prompts` | 현재 프롬프트를 `,` 로 이은 것(`",".join(prompts)`). **비어 있으면 빈 문자열** |
| `camera_image_size` | `"{w}x{h}"`. 아직 모르면 **`unknown`** |
| `input_error` | 계약 위반 문자열. 정상이면 **빈 문자열** |

> **⚠ 값이 전부 문자열이다.** `last_frame_age_s` / `last_processing_duration_ms`
> 는 숫자로 파싱하기 전에 **`never` 를 먼저 걸러야 한다.** `camera_image_size`
> 의 `unknown` 도 같다.

> **⚠ 이건 파이프라인 건강이 아니라 *입력* 건강이다.** `OK` 는 RGB-D 프레임이
> 최근에 들어왔다는 뜻이지 검출이 나가고 있다는 뜻이 아니다. 발행 여부는
> `/perception/detections` 로 볼 것.

> **⚠ `level` 만 보면 안 되는 구체적인 경우 — 빈 프롬프트.**
> 프롬프트가 비면 `on_frames` 가 **모든 프레임을 버린다** — 콘솔에는
> throttled warn 이 초당 한 번 뜬다(`1b1e793` 이후. 그 전에는 로그도 에러도
> 없는 완전 무성이었고, 이 절의 옛 문구 *"조용히 버린다"* 는 그때 것이다).
> **그런데 원격에서는 여전히 안 보인다** — 그래서 아래 키가 필요하다. 그리고
> `classify_input_health()` 는 **프롬프트를 인자로 받지도 않아서** `level` 은
> 카메라 상태만 반영한다. 즉 **`level` 이 `OK` / `WARN` 이어도 검출이 안 나갈 수
> 있고, 프롬프트가 비었는지는 `prompts` 값으로만 알 수 있다.** 그래서 이 키가
> 있는 것이다 — **소비자는 `level` 과 `prompts` 를 같이 볼 것.**
>
> 게다가 **문구가 카메라를 지목한다.** `on_prompt` 가 `_reset_input_state` 를
> 타므로 `_last_frame_time` 이 비워지고, **프레임이 흐르던 중 프롬프트가 비면**
> status 가 `RGB-D input stale` 이 아니라 **`waiting for RGB-D frames`** 로
> 나간다. 둘 다 카메라 탓처럼 읽히므로 **진짜 원인 지목은 `prompts` 값이 한다.**

### 6.2 `Detection3DArray` 의 **회전 covariance 는 sentinel 이다**

`pose.covariance` 는 `x, y, z, roll, pitch, yaw` 순 6×6 이다.

- **위치 대각**(`[0] [7] [14]`) — 필터가 낸 **실제 x/y/z 분산**(m²)
- **회전 대각**(`[21] [28] [35]`) — **`pi² = 9.8696 rad²` 고정**

> **회전 대각은 측정된 불확실성이 아니라 "미추정" sentinel 이다.**
> 이 파이프라인은 **위치만** 필터링한다. OBB 쿼터니언은 시각화에는 쓸 만하지만
> **교정된 의미론적 물체 자세가 아니고 자세 불확실성 모형이 없다.**
> **9.87 을 실제 측정값으로 읽지 말 것.**

**왜 `pi²` 인가** — 근거 없는 수가 아니라 **보수적** sentinel 이다. 원 위
균등분포의 실제 분산은 `pi²/3 ≈ 3.29 rad²` 이므로 **9.87 은 그보다도 크다.**
1σ = 180° 라 자세로 게이팅하는 파지에는 **명시적으로 쓸 수 없게** 만들면서도,
일반 ROS covariance 소비자에게 **유효한 양의 준정부호 행렬**로 남는다.
`0.0` 을 쓰면 ROS 소비자에게 **"정확히 확실함"** 을 뜻하므로 안 된다
(`pose_covariance.py` docstring).

> **⚠ 이건 협상 중인 채널이다.** `open_decisions_2026-09.md` **Q5**
> (*"로봇 쪽이 `pose.covariance` 를 읽고 판단할 수 있는가"*)가 아직 열려 있고,
> `docs/README.md` §4 의 **"경계 물체 발행 정책"** 3갈래 중 *"`covariance` 로
> 알림"* 갈래도 미선택이다. **Q5 답이 이 규약을 바꿀 수 있다** — 소비자 쪽에
> 파서를 굳히기 전에 Q5 를 먼저 볼 것.

### 6.3 `/perception/points` — 객체별 점군 (2026-08-28, 기본 꺼짐)

`sensor_msgs/PointCloud2`, 조밀·비정형 1×N, `point_step` **16**
(x·y·z·rgb 전부 float32). `frame_id` 는 다른 출력과 같은 카메라 광학 프레임.

**켜는 법** — `publish_points:=true`. 꺼져 있으면 발행자를 아예 안 만든다.

- **발행 게이트가 `/perception/detections` 와 문자 그대로 같다** — 같은 루프,
  `publishable` 기각 뒤에서 만든다. 두 토픽은 항상 같은 물체를 말한다.
- **색은 `track_id` 로 결정**되어 프레임 간 안정이고, 마커·디버그 영상과
  구성상 같은 색이다.
- **발행할 것이 없어도 `width=0` 으로 보낸다** — 빈 점군이 "지금은 아무것도
  없다" 다. 안 보내면 검출은 비었는데 RViz 엔 옛 점군이 남는다.

⚠ **점군과 상자는 같은 물체지만 같은 기하가 아니다.** 상자의 center·extent 는
**필터 상태**에서 오고(`Track.update_obb`), 이번 프레임 관측은 χ²·풋프린트
게이트에 기각됐을 수 있다. 점군은 그 뒤에 있는 **이번 프레임 원관측**이다.
**어긋남이 보이는 것이 이 토픽의 목적이다** — 상자 검증에 쓰지 말 것.

⚠ **대역폭·속도 비용.** 마스크 화소당 약 8 바이트(화소 2개당 점 1개 × 16 B —
`stride=2` 는 1차원으로 솎으므로 감축이 1/4 이 아니라 1/2 다). 640×480 물체
3개면 **프레임당 수백 KB**. 그리고 publishable 트랙마다
`mask_depth_to_points` 를 프레임당 **한 번 더** 부른다(≈10ms/트랙, 3개면
**+15~30ms**). **관측 도구지 상시 기능이 아니다.**

`rviz/perception.rviz` 에 PointCloud2 디스플레이를 넣어 뒀다(2026-08-28).
`publish_points` 가 꺼져 있으면 빈 디스플레이일 뿐이다. **Color Transformer 는
`RGB8` 로 고정해 뒀다** — RViz 기본값(Intensity)이면 `track_id` 색이 안 나온다.
