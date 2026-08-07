#!/bin/bash
# 대화형 실행: 데이터/물체/모드를 번호로 고르면 알아서 실행한다.
cd "$(dirname "$0")"

echo "=== 컨베이어 물체 인식 실행 ==="
echo
echo "어떤 영상으로 하시겠습니까?"
echo "  1) test2 — 컨베이어 정지, 물체 4종 (검출 품질 확인)"
echo "  2) test3 — 컨베이어 이동, 물체 3종 (추적 확인)"
read -rp "번호 선택 [1-2]: " bag_sel
case "$bag_sel" in
  1) BAG=bags/test2 ;;
  2) BAG=bags/test3 ;;
  *) echo "1 또는 2를 입력하세요."; exit 1 ;;
esac

echo
echo "감지할 물체를 고르세요 (여러 개면 공백으로 구분, 예: 1 3 4)"
NAMES=(물통 노트북 책 스마트폰 장갑 블록)
for i in "${!NAMES[@]}"; do echo "  $((i+1))) ${NAMES[$i]}"; done
echo "  7) 직접 입력"
read -rp "번호 선택: " -a obj_sel
PROMPTS=()
for s in "${obj_sel[@]}"; do
  if [ "$s" = "7" ]; then
    read -rp "물체 이름 입력 (쉼표 구분): " custom
    PROMPTS+=("$custom")
  elif [ "$s" -ge 1 ] 2>/dev/null && [ "$s" -le 6 ]; then
    PROMPTS+=("${NAMES[$((s-1))]}")
  else
    echo "잘못된 번호: $s"; exit 1
  fi
done
PROMPTS_STR=$(IFS=,; echo "${PROMPTS[*]}")
[ -z "$PROMPTS_STR" ] && { echo "물체를 최소 1개 선택하세요."; exit 1; }

echo
echo "실행 방식을 고르세요:"
echo "  1) 간단 — 결과 영상(mp4)과 CSV 생성, 처리 중 화면 표시 (추천)"
echo "  2) ROS — 노드 실행 + bag 재생, 토픽으로 발행 (RViz 연동용)"
read -rp "번호 선택 [1-2]: " mode_sel

echo
echo ">> 데이터: $BAG | 물체: $PROMPTS_STR"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[dry-run] mode=$mode_sel bag=$BAG prompts=$PROMPTS_STR"; exit 0
fi

if [ "$mode_sel" = "1" ]; then
  python3 scripts/run_offline.py --bag "$BAG" --prompts "$PROMPTS_STR" --show
else
  source install/setup.bash
  LOG=$(mktemp)
  echo ">> 노드 시작 (SAM3 로딩 ~40초)..."
  ros2 run roboworld_perception perception_node --ros-args \
    -p prompts:="$PROMPTS_STR" -p csv_path:=output/ros_result.csv > "$LOG" 2>&1 &
  NODE_PID=$!
  trap 'kill $NODE_PID 2>/dev/null' EXIT
  until grep -q "SAM3 ready" "$LOG"; do
    kill -0 $NODE_PID 2>/dev/null || { echo "노드 실행 실패:"; tail -5 "$LOG"; exit 1; }
    sleep 2
  done
  echo ">> 준비 완료, bag 재생 시작"
  echo ">> 다른 터미널에서 확인: ros2 topic echo /perception/detections  또는 rviz2"
  ros2 bag play "$BAG" > /dev/null 2>&1
  echo ">> bag 재생 끝. 결과 CSV: output/ros_result.csv (종료: Enter)"
  read -r
fi
