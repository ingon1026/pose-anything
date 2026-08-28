#!/bin/bash
# 사용법:
#   ./run.sh                                  # 실시간 카메라 (D455)
#   ./run.sh bags/test3                       # bag 재생
#   ./run.sh bags/test3 --prompts "책"        # 무인 실행
# 플래그: --prompts "a,b" --threshold 0.4
# (mp4/CSV만 뽑는 offline 처리: scripts/run_offline.py 직접 실행)
cd "$(dirname "$0")"

SOURCE="" PROMPTS="" THRESHOLD=0.4 HEADLESS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prompts)   PROMPTS="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --headless)  HEADLESS=1; shift ;;      # RViz·디버그 창 없이 (서버/도커)
    --source)    SOURCE="$2"; shift 2 ;;   # 하위 호환
    -*) echo "알 수 없는 인자: $1"; exit 1 ;;
    *) SOURCE="$1"; shift ;;               # 위치 인자 = bag 경로
  esac
done

if [ -z "$SOURCE" ] || [ "$SOURCE" = "live" ]; then
  SOURCE=live
  if ! lsusb 2>/dev/null | grep -qi "RealSense"; then
    echo "RealSense 카메라가 연결되어 있지 않습니다."
    echo "카메라를 연결하거나, bag으로 실행하세요:  ./run.sh <bag경로>"
    exit 1
  fi
elif [ ! -f "$SOURCE/metadata.yaml" ]; then
  echo "bag을 찾을 수 없음: $SOURCE (metadata.yaml 없음)"; exit 1
fi

if [ -z "$PROMPTS" ]; then
  read -rp "감지할 물체 (쉼표 구분): " PROMPTS
fi
[ -z "$PROMPTS" ] && { echo "물체를 최소 1개 입력하세요"; exit 1; }

echo ">> 소스: $SOURCE | 물체: $PROMPTS | threshold: $THRESHOLD"
[ "${DRY_RUN:-0}" = "1" ] && { echo "[dry-run] source=$SOURCE prompts=$PROMPTS"; exit 0; }
mkdir -p output

source install/setup.bash
# 전송 설정. .bashrc 는 대화형 셸에서만 읽혀서, 손으로 치면 상속으로 통과하고
# cron·systemd·wsl -e bash -lc 에서만 조용히 죽는다 — scripts/ros_env.sh 주석 참고
source scripts/ros_env.sh   # 8행에서 이미 저장소 루트로 cd 했다

# 이전 실행이 남긴 perception 노드 정리 — 노드가 쌓이면 GPU에 SAM3가
# 중복 상주해 전체가 심하게 느려진다 (실측: 노드 4개 → 12GB 포화)
if pgrep -f "roboworld_perception/perception_node" > /dev/null; then
  echo ">> 이전 실행의 perception 노드가 남아 있어 정리합니다..."
  pkill -f "roboworld_perception/perception_node" 2>/dev/null
  sleep 2
fi

# 각 구성요소는 setsid로 자기 프로세스그룹을 만들어 띄우고, 종료 시 그룹
# 전체(-PID)를 죽인다 — ros2 run 래퍼만 죽고 실제 노드(디버그 창)가
# 고아로 살아남는 문제 방지. Ctrl+C·Enter·오류 어느 경로든 cleanup 하나로.
PIDS=()
cleanup() {
  rc=$?
  trap - INT TERM HUP EXIT
  for pid in "${PIDS[@]}"; do kill -TERM -- -"$pid" 2>/dev/null; done
  pkill -f "roboworld_perception/perception_node" 2>/dev/null  # 래퍼 미전달 대비
  exit "$rc"
}
trap cleanup INT TERM HUP EXIT  # HUP: 터미널 창을 그냥 닫아도 노드 정리

if [ "$SOURCE" = "live" ]; then
  if ros2 topic list 2>/dev/null | grep -q "^/camera/camera/color/image_raw$"; then
    echo ">> 이미 실행 중인 카메라 노드를 사용합니다 (중복 실행 방지)"
  else
    # 아래 launch 는 패키지가 없으면 곧바로 죽는다 — 미리 확인한다
    if ! ros2 pkg prefix realsense2_camera > /dev/null 2>&1; then
      echo "!! realsense2_camera 패키지가 없습니다 — 카메라 노드가 뜨지 않습니다."
      echo "   sudo apt install ros-jazzy-realsense2-camera"
    fi
    CAM_LOG=$(mktemp)
    echo ">> RealSense 카메라 시작... (로그: $CAM_LOG)"
    setsid ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true > "$CAM_LOG" 2>&1 &
    CAM_PID=$!
    PIDS+=($CAM_PID)
    sleep 2
    if ! kill -0 $CAM_PID 2>/dev/null; then
      echo "!! RealSense 카메라 기동 실패 — 영상이 들어오지 않아 검출이 진행되지 않습니다."
      echo "   흔한 원인: realsense2_camera 패키지 부재, USB 연결·권한, 다른 프로세스가 카메라 점유."
      tail -20 "$CAM_LOG"
      exit 1
    fi
  fi
fi

# GPU 가 없으면 sam3_detector.py 가 경고 없이 CPU 로 떨어진다
# (`"cuda" if torch.cuda.is_available() else "cpu"`). 모델 로드 자체는 CPU
# 에서도 성공해 "SAM3 ready" 가 정상적으로 찍히므로 아래 900초 상계에는
# 걸리지 않는다 — 증상은 추론에서만 나온다: 848M 파라미터를 fp32 CPU 로
# 돌려 프레임당 수십 초, bag 은 끝까지 재생되고 CSV 는 거의 빈다.
# 그래서 이 확인은 여기서, 기동 전에 한 번 해야 한다.
# nvidia-smi 유무가 아니라 같은 식을 그대로 물어본다 — torch 가 CPU 전용
# 빌드면 nvidia-smi 가 있어도 CUDA 는 못 쓴다. torch import 자체가 실패하면
# 빈 문자열이 되어 여기서는 조용히 넘어간다(노드가 곧바로 죽고 아래
# `kill -0` + `tail -20` 경로가 그 이유를 보여준다).
GPU_OK=$(python3 -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null)
if [ "$GPU_OK" = "0" ]; then
  echo "!! CUDA GPU 를 쓸 수 없습니다 — SAM3 가 CPU 로 돕니다 (프레임당 수십 초)."
  echo "   기동은 정상으로 끝나지만 검출이 사실상 진행되지 않습니다."
  echo "   결과 CSV 가 거의 비어 있으면 다른 원인을 찾기 전에 이 줄을 보세요."
fi
LOG=$(mktemp)
CSV="output/ros_$(date +%Y%m%d_%H%M%S).csv"
# 노드 stdout/stderr 는 전부 이 파일로 간다. pipeline.py 의 진단([stamp] 지연
# 프레임 드롭, [belt_plane] dist_thresh/추정실패)은 get_logger() 가 아니라 맨
# print() 라 /rosout 에도 ~/.ros/log 에도 안 남는다 — 여기가 유일한 사본이다.
echo ">> 노드 로그: $LOG"
echo ">> 노드 시작 (SAM3 로딩 ~40초, 체크포인트 첫 다운로드면 훨씬 길다)..."
DISPLAY_PARAM=true
[ "$HEADLESS" = "1" ] && DISPLAY_PARAM=false
setsid ros2 run roboworld_perception perception_node --ros-args \
  -p prompts:="$PROMPTS" -p score_threshold:="$THRESHOLD" \
  -p display:=$DISPLAY_PARAM -p csv_path:="$CSV" > "$LOG" 2>&1 &
NODE_PID=$!
PIDS+=($NODE_PID)
# "SAM3 ready"는 perception_node가 찍는 로그 — 노드 쪽 문구 변경 시 함께 수정
#
# 상계가 필요한 이유: kill -0 은 *죽은* 프로세스만 잡는다. HF 체크포인트
# (facebook/sam3, 3.44GB) 첫 다운로드가 느리거나 멈추면 노드는 살아 있는 채로
# 진행이 없고, 이 루프는 영원히 돈다. (토큰은 이 분기에 오지 않는다 —
# from_pretrained 는 gated repo 에서 대화형 프롬프트를 띄우지 않고 예외를
# 던진다. HF_TOKEN 미설정·무권한은 즉시 사망해 위 `kill -0` 이 잡는다.)
# 무한 대기를 상계로 바꾸는 것은 pipeline.py 의 LATE_DROP_STREAK_MAX 와 같은 이유다.
# 900초 근거: 캐시가 있으면 ~40초. 캐시가 없으면 3.44GB 를 받아야 하는데
# 900초는 실효 3.9MB/s(≈31Mbps) 이상이면 통과한다. 그보다 느린 회선이면
# STARTUP_TIMEOUT 을 올리거나, 미리 캐시를 채워 두는 쪽이 낫다.
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}
SECONDS=0
until grep -q "SAM3 ready" "$LOG"; do
  kill -0 $NODE_PID 2>/dev/null || { echo "노드 실행 실패:"; tail -20 "$LOG"; exit 1; }
  if [ "$SECONDS" -ge "$STARTUP_TIMEOUT" ]; then
    echo "노드가 ${STARTUP_TIMEOUT}초 안에 'SAM3 ready' 를 찍지 못했습니다 (프로세스는 살아 있음)."
    echo "흔한 원인: 체크포인트 첫 다운로드가 느리거나 멈춤."
    echo "대처: 캐시를 미리 채우거나(hf download facebook/sam3), STARTUP_TIMEOUT=1800 ./run.sh ..."
    echo "로그 전문: $LOG"
    tail -20 "$LOG"
    exit 1
  fi
  sleep 2
done
if [ "$HEADLESS" = "1" ]; then
  echo ">> 준비 완료 (headless) — 토픽: /perception/detections"
else
  # world TF는 perception_node가 camera_info의 실제 frame 이름으로 직접
  # 발행한다 (frame 이름이 bag·카메라 설정마다 달라 여기서 하드코딩 불가)
  if ! command -v rviz2 > /dev/null 2>&1; then
    echo "!! rviz2 가 없습니다 — 3D 마커 창은 뜨지 않습니다 (ros-base 설치에는 없다)."
    echo "   sudo apt install ros-jazzy-rviz2   ·  또는 --headless 로 실행"
  fi
  RVIZ_LOG=$(mktemp)
  setsid rviz2 -d src/roboworld_perception/rviz/perception.rviz > "$RVIZ_LOG" 2>&1 &
  RVIZ_PID=$!
  PIDS+=($RVIZ_PID)
  sleep 2
  if ! kill -0 $RVIZ_PID 2>/dev/null; then
    # 위 `command -v` 를 통과하고도 죽는 경우 — WSL 에서 DISPLAY 가 없을 때가 흔하다
    echo "!! rviz2 가 떴다가 죽었습니다 — 3D 마커 창만 없고 인식은 계속됩니다 (로그: $RVIZ_LOG)."
    tail -20 "$RVIZ_LOG"
  fi
  echo ">> 준비 완료 — 디버그 창(전체 크기) + RViz 3D 박스(MarkerArray)"
fi

if [ "$SOURCE" = "live" ]; then
  echo ">> 실시간 실행 중 (종료: Ctrl+C)"
  wait $NODE_PID
else
  # 여기를 /dev/null 로 버리면 mcap 플러그인 부재·bag 손상·storage id 불일치
  # 어느 것이 터져도 화면에는 ">> 준비 완료" 다음 즉시 ">> bag 재생 끝" 과
  # 헤더뿐인 CSV 만 남는다. $LOG 가 아니라 별도 파일인 이유: 노드는 재생 중
  # 프레임마다 맨 print() 를 쏟으므로(노드 로그 파일 주석) 재생 도중 난 실패는 tail -20
  # 밖으로 밀려난다.
  BAG_LOG=$(mktemp)
  echo ">> bag 재생 중 (로그: $BAG_LOG)..."
  ros2 bag play "$SOURCE" > "$BAG_LOG" 2>&1
  BAG_RC=$?   # `if ! ...` 로 감싸면 ! 가 종료코드를 삼킨다
  if [ "$BAG_RC" -ne 0 ]; then
    echo "!! bag 재생 실패 (ros2 bag play 종료코드 $BAG_RC) — CSV 는 비어 있습니다: $CSV"
    echo "   흔한 원인: mcap 스토리지 플러그인 부재(sudo apt install ros-jazzy-rosbag2-storage-mcap),"
    echo "   bag 손상, metadata.yaml 의 storage id 불일치."
    tail -20 "$BAG_LOG"
    exit 1
  fi
  echo ">> bag 재생 끝. 결과 CSV: $CSV (종료: Enter)"
  read -r
fi
