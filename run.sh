#!/bin/bash
# 대화형:   ./run.sh
# 비대화식: ./run.sh --source live --prompts "thermos" [--mode ros]
#           ./run.sh --source bags/test3 --prompts "책,장갑" --mode offline [--threshold 0.4]
cd "$(dirname "$0")"

SOURCE="" PROMPTS="" MODE="" THRESHOLD=0.4
while [ $# -gt 0 ]; do
  case "$1" in
    --source)    SOURCE="$2"; shift 2 ;;
    --prompts)   PROMPTS="$2"; shift 2 ;;
    --mode)      MODE="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    *) echo "알 수 없는 인자: $1  (--source --prompts --mode --threshold)"; exit 1 ;;
  esac
done

# --- 입력 소스 ---
if [ -z "$SOURCE" ]; then
  echo "=== 컨베이어 물체 인식 실행 ==="
  echo
  echo "입력 소스를 고르세요:"
  mapfile -t BAGS < <(find . -maxdepth 3 -name metadata.yaml -printf '%h\n' 2>/dev/null | sed 's|^\./||' | sort)
  i=1
  for b in "${BAGS[@]}"; do echo "  $i) bag: $b"; i=$((i+1)); done
  echo "  L) 실시간 카메라 (RealSense D455)"
  echo "  P) bag 경로 직접 입력"
  read -rp "선택: " sel
  case "$sel" in
    [Ll]) SOURCE=live ;;
    [Pp]) read -rp "bag 경로: " SOURCE ;;
    *) SOURCE="${BAGS[$((sel-1))]:-}"
       [ -z "$SOURCE" ] && { echo "잘못된 선택"; exit 1; } ;;
  esac
fi
if [ "$SOURCE" != "live" ] && [ ! -f "$SOURCE/metadata.yaml" ]; then
  echo "bag을 찾을 수 없음: $SOURCE (metadata.yaml 없음)"; exit 1
fi

# --- 물체 (자유 텍스트) ---
if [ -z "$PROMPTS" ]; then
  KNOWN=$(awk '/^PROMPT_ALIASES/,/^}/' \
    src/roboworld_perception/roboworld_perception/sam3_detector.py \
    | sed -n 's/^\s*"\([^"]*\)".*/\1/p' | paste -sd, -)
  echo
  echo "감지할 물체를 입력하세요 (쉼표 구분)"
  echo "  한글 자동 변환: $KNOWN"
  echo "  그 외 물체는 영어로 (예: red box, screwdriver)"
  read -rp "물체: " PROMPTS
fi
[ -z "$PROMPTS" ] && { echo "물체를 최소 1개 입력하세요"; exit 1; }

# --- 모드 ---
[ "$SOURCE" = "live" ] && MODE=ros  # 실시간은 ROS 모드 고정
if [ -z "$MODE" ]; then
  echo
  echo "실행 방식을 고르세요:"
  echo "  1) 간단 — 결과 영상(mp4)과 CSV 생성, 처리 중 화면 표시 (추천)"
  echo "  2) ROS — 노드 실행 + RViz, 토픽으로 발행"
  read -rp "번호 선택 [1-2]: " m
  if [ "$m" = "2" ]; then MODE=ros; else MODE=offline; fi
fi

echo
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
