# Roboworld 비전 시스템 점검·수정 인수인계

## 프로젝트 개요

- ROS2 Jazzy Python 기반 RGB-D 비전 시스템
- CameraInfo + color/depth 동기화
- SAM3 텍스트 프롬프트 검출
- 트래킹, depth geometry, OBB pose 생성
- `Detection3DArray`, `MarkerArray`, `/perception/status` 발행
- Isaac Sim 실시간 입력 및 rosbag 재생 대상

## 발견한 문제와 수정 내용

### 1. timestamp reset 및 늦은 프레임

기존에는 `/clock`이 새 bag 시작점으로 돌아가거나 오래된 RGB-D pair가 늦게 도착할 때 tracker/filter 상태가 과거 시간으로 되돌아갈 수 있었다.

수정 내용:

- 새 run reset 조건을 명확히 분리했다.
- 이전 stamp가 1초보다 크고 새 stamp가 1초 이하로 돌아가며 0.5초 이상 역행할 때만 clock reset으로 처리한다.
- 일반적인 늦은 non-zero frame은 `late_frame_drop_required()`로 판정하고 폐기한다.
- 늦은 frame은 tracker/filter/`last_accept` 상태를 되감지 않는다.
- 실제 reset 발생 시 SAM3 추론 전에 이전 검출과 마커를 먼저 삭제한다.
- 늦은 frame 경고는 1초 단위로 제한한다.

관련 파일:

- `src/roboworld_perception/roboworld_perception/pipeline.py`
- `src/roboworld_perception/roboworld_perception/perception_node.py`
- `src/roboworld_perception/test/test_pipeline.py`

### 2. CameraInfo/RGB-D 입력 계약

기존에는 CameraInfo가 바뀌거나 color/depth 크기·frame이 달라도 기존 K와 tracker 상태를 사용할 위험이 있었다.

수정 내용:

- CameraInfo width/height 검증
- K matrix 9개 및 finite 값 검증
- positive `fx/fy` 검증
- `frame_id` 존재 검증
- color/depth 해상도와 CameraInfo 해상도 비교
- color/depth `frame_id`와 CameraInfo frame 비교
- CameraInfo signature가 바뀌면 pipeline/tracker reset
- 계약 위반 프레임은 버리고 기존 출력도 삭제

### 3. QoS와 동기화 큐

SAM3 추론이 카메라보다 느리면 오래된 이미지가 큐에 쌓여 실제보다 늦은 pose를 만들 수 있었다.

수정 내용:

- color/depth subscriber를 `BEST_EFFORT`, `KEEP_LAST`, depth 1로 설정
- `ApproximateTimeSynchronizer` queue 기본값도 1로 설정
- Isaac preset에서 `sync_queue_size=1` 명시
- `sync_slop` 기본값은 0.05초 유지

관련 파일:

- `perception_node.py`
- `launch/perception.launch.py`
- `launch/isaac.launch.py`

### 4. TF 안전성

기존 nominal world-to-camera transform이 실제 로봇 좌표로 오인될 위험이 있었다.

수정 내용:

- `publish_world_tf` 기본값을 `false`로 변경
- true로 켤 경우 보정되지 않은 RViz 전용 transform이라는 경고 출력
- 실제 로봇 좌표는 외부 calibrated TF/URDF가 담당하도록 명확화
- Isaac preset에서는 `publish_optical_tf=true`
- 일반 RealSense/bag에서는 optical TF 기본값 `false`

### 5. `use_sim_time`

- base launch 기본값: `use_sim_time=false`
- Isaac launch preset: `use_sim_time=true`
- perception node와 RViz 양쪽에 동일하게 전달
- Isaac `/clock`과 perception/RViz 시간축 일치

### 6. Pose covariance

OBB 회전은 실제로 추정하지 않는데 covariance가 낙관적으로 설정될 수 있었다.

수정 내용:

- `pose_covariance.py` 추가
- xyz에는 depth/fusion filter variance 반영
- 미추정 roll/pitch/yaw에는 보수적인 `pi²` variance 사용
- 6x6 covariance의 xyz index는 0, 7, 14
- rotation index는 21, 28, 35

관련 파일:

- `pose_covariance.py`
- `test_pose_covariance.py`

### 7. Optical flow fail-safe

`propagate_mask()`가 실패해도 depth geometry/fusion/last_accept가 갱신될 가능성이 있었다.

수정 내용:

- flow propagation 실패 시 display용 track만 유지
- depth geometry 계산, fuse, `last_accept` 갱신은 수행하지 않음
- 잘못된 flow 결과가 정상 pose로 승격되지 않도록 차단

### 8. Diagnostics

`/perception/status`를 추가했다.

타입:

```text
diagnostic_msgs/DiagnosticArray
```

포함 정보:

- CameraInfo 유효성
- RGB-D contract 유효성
- 마지막 frame age
- 마지막 processing duration
- stale timeout
- camera frame 및 image size
- input error
- out-of-order frame drop count

추가 변경:

- `diagnostic_msgs`를 `package.xml` dependency에 추가
- Docker에 `ros-jazzy-diagnostic-msgs` 추가
- 일부 WSL ROS generated message ABI에서 `DiagnosticStatus.level`이 bytes인 문제를 `set_diagnostic_level()`로 호환 처리

## 검증 결과

- Python 테스트: `111 passed`
- `compileall` 통과
- `git diff --check` 통과
- `colcon build --packages-select roboworld_perception --symlink-install` 통과
- Isaac rosbag 실제 재생 성공
- SAM3 정상 기동
- CameraInfo 정상 수신
- `/perception/status`에서 RGB-D input healthy 확인
- 실제 처리시간 약 148~301ms
- ROS Assertion/Traceback/프로세스 크래시 없음
- 늦은 RGB-D frame을 실제로 발견했고 안전하게 폐기됨
- offline smoke에서 12 frames 처리 및 MP4/CSV 출력 성공

## 남은 경고

~~현재 환경은 `torch 2.10.0+cu128`이고 SAM3 C++ extension은 `torch >= 2.11.0`을 요구한다.~~

~~현재는 fallback 모드로 동작하며 테스트는 통과했다. 운영 전 PyTorch/CUDA/Isaac 조합을 공식 호환 버전으로 맞추는 작업이 필요하다. torch를 즉시 업그레이드하지 말고 GPU/Isaac 호환성을 먼저 검증한다.~~

> **정정 (2026-08-28).** 위 두 문단은 부정확하다. 실행 로그 첫 줄의
> `Skipping import of cpp extensions ... (found 2.10.0+cu128)` 는 **`torchao 0.17.0`**
> 이 내는 경고이고, 빠지는 것은 그 **양자화 커널**이지 SAM3 자체가 아니다.
> SAM3 는 `transformers` 로 bf16 로 돌고 `quantization_config` 도 없다
> (`transformers/models/sam3/` 에 `torchao` 참조 0건). 따라서 **"SAM3 가 fallback
> 모드로 동작 중" 이라는 서술은 성립하지 않는다.**
>
> 이 문장을 그대로 읽으면 **torch 업그레이드가 급한 일로 오해되는데, 급하지 않다.**
> 속도와 관계가 있는지는 **미확인**이므로 "확장이 없어서 느린 값" 이라고 적지 말 것.
> 근거가 속도 하나뿐이라 torch 를 올리면 한 번은 다시 재야 한다.
>
> 원문: `docs/README.md` §5

## 중요한 빌드 주의사항

기존 workspace의 `install/roboworld_perception`가 오래된 Python build directory를 가리키는 현상이 확인됐다. 소스를 수정해도 오래된 모듈이 실행될 수 있다.

배포 전 반드시 다음을 확인한다.

```bash
python3 -c "import roboworld_perception.perception_node as m; print(m.__file__)"
```

실제 import 경로가 최신 소스/최신 build인지 확인한다. 필요하면 해당 패키지만 clean rebuild하거나 isolated overlay를 사용한다. workspace 전체를 무조건 삭제하지 않는다.

## 변경 파일

수정:

- `Dockerfile`
- `README.md`
- `docs/bridge_contract.md`
- `src/roboworld_perception/launch/isaac.launch.py`
- `src/roboworld_perception/launch/perception.launch.py`
- `src/roboworld_perception/package.xml`
- `src/roboworld_perception/roboworld_perception/perception_node.py`
- `src/roboworld_perception/roboworld_perception/pipeline.py`
- `src/roboworld_perception/test/test_pipeline.py`

추가:

- `src/roboworld_perception/roboworld_perception/input_health.py`
- `src/roboworld_perception/roboworld_perception/pose_covariance.py`
- `src/roboworld_perception/test/test_input_health.py`
- `src/roboworld_perception/test/test_pose_covariance.py`

아직 commit하지 않은 작업 트리 상태다.

## 다음 작업 우선순위

1. 최신 소스가 실제 ROS 실행 프로세스에 import되는지 확인
2. 일반 workspace에서 package clean rebuild
3. Isaac rosbag에서 `/perception/status` drop count 확인
4. 실제 Isaac Sim live publisher 연결 시 CameraInfo/frame/TF 계약 검증
5. torch 업그레이드는 별도 호환성 작업으로 진행
6. 최종 검증 후 commit 생성
