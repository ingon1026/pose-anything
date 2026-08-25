#!/bin/bash
# CI 환경(torch 도 rclpy 도 없음) 재현 — 푸시 전에 돌릴 것.
#
# .github/workflows 는 의도적으로 torch/SAM3 를 설치하지 않는다
# ("geometry·tracker·pipeline 단위 테스트만 실행"). 로컬에는 torch 가 있어서
# top-level import torch 가 들어간 테스트를 추가해도 로컬은 통과하고 CI 만
# 깨진다 — 2026-08-24 에 실제로 그래서 CI 가 3번 연속 실패했다.
#
# ⚠ 막아야 하는 것은 torch 만이 아니다. CI 러너에는 rclpy 도 없다. 아래
# _Block 은 **둘 다 명시적으로** 막는다 — 예전에는 torch 만 적혀 있었고
# rclpy 는 `PYTHONPATH=d` 가 기존 PYTHONPATH 를 통째로 덮어써서 **우연히**
# 같이 사라지고 있었다(이 머신의 rclpy 는 PYTHONPATH 로만 도달 가능하다:
# /home/ingon/ros2_jazzy/install/rclpy/...). 그 우연에 기대면, PYTHONPATH 를
# `d + os.pathsep + 기존` 으로 이어붙이는 sitecustomize 의 표준 관용구로
# 고치는 순간 rclpy 가 되살아나 재현이 조용히 깨진다.
set -eu
cd "$(dirname "$0")/../src/roboworld_perception"
python3 - <<'PY'
import sys, subprocess, tempfile, pathlib, os
d = tempfile.mkdtemp()
pathlib.Path(d, "sitecustomize.py").write_text('''
import sys
from importlib.abc import MetaPathFinder
class _Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "rclpy"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _Block())
''')
env = dict(os.environ, PYTHONPATH=d)
r = subprocess.run([sys.executable, "-m", "pytest", "test", "-q"],
                   capture_output=True, text=True, env=env)
print("\n".join(r.stdout.strip().splitlines()[-2:]))
sys.exit(r.returncode)
PY
