#!/bin/bash
# 임의 uid 로 띄워도 견딘다.
#
# `--user $(id -u):$(id -g)` 로 non-root 를 걸면 그 uid 가 이미지 `/etc/passwd`
# 에 없어서 `getpass.getuser()` 가 `KeyError: getpwuid()` 로 죽는다 —
# `transformers` 의 SAM3 가 `torchvision` 을 import 하며 그 경로를 탄다.
# 이미지에는 uid 1000(`ubuntu`)만 있으므로 **자기 uid 를 넣는 순간 걸린다.**
# `getpass` 는 pwd 조회 **전에** LOGNAME·USER·LNAME·USERNAME 을 보므로 값을
# 채워주면 통과한다. `HOME` 은 uid 미등록 시 `/` 가 되어 `~/.ros/log` 가 막힌다.
# 근거·실측: docs/shared_server_2026-08-31.md §4
if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  : "${USER:=appuser}"; export USER
  case "${HOME:-/}" in /|"") HOME=/tmp ;; esac; export HOME
fi

source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash

# DDS 전송 설정. 없으면 이미지처럼 **큰 메시지가 에러 없이 사라진다**
# (`camera_info` 만 오고 RGB-D 가 안 온다 — 2026-08-31 실측, §9-3).
# `docker-compose.yml` 은 `environment:` 로 같은 값을 넣지만 `docker run` 은
# 안 넣는다. 여기서 채우면 두 경로가 같이 덮인다. 이미 걸려 있으면 그대로 둔다.
[ -n "${FASTDDS_BUILTIN_TRANSPORTS:-}" ] || source /ws/scripts/ros_env.sh

exec "$@"
