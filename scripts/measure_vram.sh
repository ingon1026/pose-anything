#!/bin/bash
# Isaac + 인식 VRAM 측정 (② Isaac 단독 / ④ 동시 피크).
#
# 계약서(docs/bridge_contract.md)의 함정 두 개를 전제 검사로 박아뒀다:
#   §1.1 FASTDDS_BUILTIN_TRANSPORTS 는 비대화형 셸에 안 실린다
#   §3.3 topic list 에 이름이 보여도 퍼블리셔가 0 일 수 있다
# 그리고 검출이 실제로 도는지 확인한다 — 2026-08-24 에 SAM3 가 로드만 되고
# 추론을 한 번도 안 한 상태의 VRAM 을 재서 '동시 피크' 로 잘못 보고했다.
#
# 사용: bash scripts/measure_vram.sh [출력디렉토리]
set -u
source "$(dirname "$0")/ros_env.sh"
# ROS setup.bash 는 미정의 변수를 참조한다 — 이 구간만 set -u 를 푼다
set +u
source /opt/ros/jazzy/setup.bash
source "$(dirname "$0")/../install/setup.bash" 2>/dev/null || true
set -u
OUT=${1:-/tmp/vram_$(date +%H%M%S)}; mkdir -p "$OUT"
TOPIC=/camera/camera/color/image_raw

sample() {  # sample <라벨> <초>
  local f="$OUT/$1.csv"; echo "used_MiB" > "$f"
  for _ in $(seq 1 $(( $2 * 2 ))); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$f"; sleep 0.5
  done
  python3 -c "
v=[int(x) for x in open('$f').read().split()[1:]]
print(f'[$1] n={len(v)} 최소 {min(v)} 평균 {sum(v)//len(v)} **최대 {max(v)}** MiB')"
}

echo "=== 0. 전제 — 발행자가 있는가 ==="
PUB=$(timeout 15 ros2 topic info $TOPIC 2>/dev/null | grep -oP 'Publisher count: \K\d+')
echo "Publisher count: ${PUB:-조회실패}"
[ "${PUB:-0}" -eq 0 ] && {
  echo "→ 브리지가 죽었다. 비전 쪽에서 고칠 것이 없다 (docs/bridge_contract.md §3.4)."; exit 1; }
timeout 15 ros2 topic hz $TOPIC 2>&1 | grep -m1 "average rate" || {
  echo "→ 발행자는 있는데 흐름이 없다. 전송 설정(§1.1) 확인."; exit 1; }

echo; echo "=== ② Isaac 단독 ==="; sample isaac_only 30

echo; echo "=== 인식 노드 기동 ==="
setsid ros2 launch roboworld_perception isaac.launch.py > "$OUT/node.log" 2>&1 &
until grep -q "SAM3 ready" "$OUT/node.log" 2>/dev/null; do sleep 5; done
echo "SAM3 ready — 추론이 실제로 도는지 확인:"
# 경고로 두면 안 된다 — 아침(2026-08-24) 측정을 무효로 만든 조건이 정확히
# 이것이다. SAM3 가 로드만 되고 추론을 한 번도 안 한 상태의 VRAM 을 재서
# '동시 피크 9,497 MiB' 로 보고했다(실제 8,691). 값이 나와버리면 사람은 쓴다.
timeout 25 ros2 topic hz /perception/detections 2>&1 | grep -m1 "average rate" || {
  echo "→ 검출 미발행. 이 상태의 VRAM 은 활성화 몫이 빠져 무효다 — 측정을 중단한다."
  pkill -f perception_node; exit 1; }

echo; echo "=== ④ 동시 구동 피크 ==="; sample isaac_plus_sam3 60

echo; echo "=== 정리 ==="
pkill -f perception_node; sleep 3
echo "회수 후: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
