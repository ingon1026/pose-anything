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
source "$(dirname "$0")/scripts/ros_env.sh"

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
    echo ">> RealSense 카메라 시작..."
    setsid ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true > /dev/null 2>&1 &
    PIDS+=($!)
  fi
fi

LOG=$(mktemp)
CSV="output/ros_$(date +%Y%m%d_%H%M%S).csv"
echo ">> 노드 시작 (SAM3 로딩 ~40초)..."
DISPLAY_PARAM=true
[ "$HEADLESS" = "1" ] && DISPLAY_PARAM=false
setsid ros2 run roboworld_perception perception_node --ros-args \
  -p prompts:="$PROMPTS" -p score_threshold:="$THRESHOLD" \
  -p display:=$DISPLAY_PARAM -p csv_path:="$CSV" > "$LOG" 2>&1 &
NODE_PID=$!
PIDS+=($NODE_PID)
# "SAM3 ready"는 perception_node가 찍는 로그 — 노드 쪽 문구 변경 시 함께 수정
until grep -q "SAM3 ready" "$LOG"; do
  kill -0 $NODE_PID 2>/dev/null || { echo "노드 실행 실패:"; tail -5 "$LOG"; exit 1; }
  sleep 2
done
if [ "$HEADLESS" = "1" ]; then
  echo ">> 준비 완료 (headless) — 토픽: /perception/detections"
else
  # world TF는 perception_node가 camera_info의 실제 frame 이름으로 직접
  # 발행한다 (frame 이름이 bag·카메라 설정마다 달라 여기서 하드코딩 불가)
  setsid rviz2 -d src/roboworld_perception/rviz/perception.rviz > /dev/null 2>&1 &
  PIDS+=($!)
  echo ">> 준비 완료 — 디버그 창(전체 크기) + RViz 3D 박스(MarkerArray)"
fi

if [ "$SOURCE" = "live" ]; then
  echo ">> 실시간 실행 중 (종료: Ctrl+C)"
  wait $NODE_PID
else
  ros2 bag play "$SOURCE" > /dev/null 2>&1
  echo ">> bag 재생 끝. 결과 CSV: $CSV (종료: Enter)"
  read -r
fi
