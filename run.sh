#!/bin/bash
# 사용법:
#   ./run.sh                                  # 실시간 카메라 (D455)
#   ./run.sh bags/test3                       # bag 재생
#   ./run.sh bags/test3 --prompts "책" --mode offline   # 무인 실행
# 플래그: --prompts "a,b" --mode ros|offline --threshold 0.4
cd "$(dirname "$0")"

SOURCE="" PROMPTS="" MODE="" THRESHOLD=0.4
while [ $# -gt 0 ]; do
  case "$1" in
    --prompts)   PROMPTS="$2"; shift 2 ;;
    --mode)      MODE="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
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

[ "$SOURCE" = "live" ] && MODE=ros  # 실시간은 ROS 모드 고정
if [ -z "$MODE" ]; then
  echo "실행 방식:  1) 간단 — mp4+CSV 생성, 화면 표시   2) ROS — 노드+RViz"
  read -rp "번호 선택 [1-2]: " m
  if [ "$m" = "2" ]; then MODE=ros; else MODE=offline; fi
fi

echo ">> 소스: $SOURCE | 물체: $PROMPTS | 모드: $MODE | threshold: $THRESHOLD"
[ "${DRY_RUN:-0}" = "1" ] && { echo "[dry-run] source=$SOURCE prompts=$PROMPTS mode=$MODE"; exit 0; }
mkdir -p output

if [ "$MODE" = "offline" ]; then
  python3 scripts/run_offline.py --bag "$SOURCE" --prompts "$PROMPTS" \
    --threshold "$THRESHOLD" --show
  exit
fi

source install/setup.bash
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null' EXIT

if [ "$SOURCE" = "live" ]; then
  echo ">> RealSense 카메라 시작..."
  ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true > /dev/null 2>&1 &
  PIDS+=($!)
fi

LOG=$(mktemp)
CSV="output/ros_$(date +%Y%m%d_%H%M%S).csv"
echo ">> 노드 시작 (SAM3 로딩 ~40초)..."
ros2 run roboworld_perception perception_node --ros-args \
  -p prompts:="$PROMPTS" -p score_threshold:="$THRESHOLD" \
  -p display:=true -p csv_path:="$CSV" > "$LOG" 2>&1 &
NODE_PID=$!
PIDS+=($NODE_PID)
until grep -q "SAM3 ready" "$LOG"; do
  kill -0 $NODE_PID 2>/dev/null || { echo "노드 실행 실패:"; tail -5 "$LOG"; exit 1; }
  sleep 2
done
rviz2 -d src/roboworld_perception/rviz/perception.rviz > /dev/null 2>&1 &
PIDS+=($!)
echo ">> 준비 완료 — 디버그 창(전체 크기) + RViz 3D 박스(MarkerArray)"

if [ "$SOURCE" = "live" ]; then
  echo ">> 실시간 실행 중 (종료: Ctrl+C)"
  wait $NODE_PID
else
  ros2 bag play "$SOURCE" > /dev/null 2>&1
  echo ">> bag 재생 끝. 결과 CSV: $CSV (종료: Enter)"
  read -r
fi
