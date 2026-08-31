# ROS 2 전송 설정 — 노드를 띄우거나 토픽을 읽는 모든 셸 경로에서 source 할 것.
# 계약서 docs/bridge_contract.md §1.1 의 값이다.
#
# 왜 .bashrc 로 충분하지 않은가:
#   .bashrc 는 대화형 셸에서만 읽힌다(Ubuntu 기본 조기 반환 `case $- in *i*)`).
#   실측 — env -i bash -lc  -> 빈 값
#          env -i bash -ic  -> 설정됨
#   그런데 터미널에서 손으로 스크립트를 치면 부모 셸에서 상속받아 통과한다.
#   **그래서 손으로 테스트하면 항상 깨끗해 보이고, cron·systemd·
#   wsl -e bash -lc·docker 에서만 조용히 실패한다.** env 를 잊는 사고가
#   이미 두 번 났다(계약서 §1.1). 증상은 에러가 아니라 "토픽은 보이는데
#   흐름이 0" 이라 원인을 전송 설정에서 찾지 않게 된다.
#
#   install/setup.bash 는 대안이 아니다 — colcon 생성 파일은 AMENT_*/PATH/
#   PYTHONPATH 만 건드리고 전송 변수는 전혀 설정하지 않는다(확인함).
#
# 실측 효과: LARGE_DATA 적용 시 color/depth 1.3 Hz -> 39.0 Hz,
# 최대 공백 7~9초 -> 0.03초, 스탬프 일치 11~16% -> 100%.
#
# 제약: max_msg_size <= sockets_size <= net.core.rmem_max(기본 212992).
# 1MB 를 요구하면 트랜스포트 등록이 실패해 통신이 통째로 끊긴다.
#
# 이 값의 사본이 여기 말고 더 있다. 하나를 바꾸면 전부 바꿀 것:
#   docs/bridge_contract.md §1.1 / docker-compose.yml /
#   src/roboworld_perception/launch/perception.launch.py (SetEnvironmentVariable) /
#   (Windows) cellomni 실행.bat
# ⚠ `docker run` 으로 직접 띄우면 이 파일이 **안 읽힌다** — `docker-entrypoint.sh`
#   는 `/opt/ros` 와 `/ws/install` 만 source 한다. 그 경로에서는 위 두 값을 `-e` 로
#   손으로 넘겨야 하고, 안 넘기면 `camera_info` 만 오고 이미지가 조용히 안 온다
#   (2026-08-31 라이브 재생에서 실제로 걸렸다. docs/shared_server_2026-08-31.md §9-3)
# Windows 쪽 사본은 이 저장소가 관리할 수 없다 — 한쪽만 켜면 안 붙는다.
export FASTDDS_BUILTIN_TRANSPORTS='LARGE_DATA?max_msg_size=190KB&sockets_size=200KB&non_blocking=true&tcp_negotiation_timeout=50'
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
