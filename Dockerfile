# ⚠ 이 태그는 고정이 아니다 — 오늘 빌드하면 오늘의 ros-base 를 받는다.
# 아래 pip 핀은 그 위에 얹히는 층만 묶으므로, "측정 조건 고정" 은 OS·ROS
# 층에는 미치지 않는다. 다이제스트로 묶으려면
#   FROM ros:jazzy-ros-base@sha256:<...>
# 인데 그러면 보안 갱신이 안 오므로 손으로 올려야 한다 - 아직 안 골랐다.
FROM ros:jazzy-ros-base

# ──────────────────────────────────────────
# 1. apt: RealSense + RViz + 메시지·bag 플러그인 + 런타임 라이브러리
# ──────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-jazzy-librealsense2* \
    ros-jazzy-realsense2-camera \
    ros-jazzy-rviz2 \
    ros-jazzy-vision-msgs \
    ros-jazzy-diagnostic-msgs \
    ros-jazzy-rosbag2-storage-mcap \
    fonts-noto-cjk \
    libgl1 libgomp1 usbutils \
  && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────
# 2. pip: PyTorch CUDA (가장 무거운 레이어 — 소스 복사보다 먼저 캐시)
# ──────────────────────────────────────────
# ⚠ torch / transformers / open3d 는 **고정한다.** 이 저장소의 성능·정확도
# 수치는 전부 특정 조합에서 잰 값이고(docs/README.md §5 "측정 조건을 같이
# 적을 것"), 재빌드가 이 셋 중 하나를 올리면 측정 조건 전체가 라벨을 잃는다.
# 아래 값은 그 측정이 실제로 돌던 호스트 조합이다 — docs/README.md §5 가
# 인용한 실행 로그의 `found 2.10.0+cu128` 이 같은 torch 다.
# torchvision 은 torch 와 짝이라 같이 고정한다(따로 두면 cu128 인덱스에서
# 짝이 안 맞는 버전이 잡힌다).
# 올릴 때는 버전만 바꾸지 말고 scripts/check_accuracy.py 로 기준 대비
# 회귀부터 판정할 것.
RUN pip3 install --break-system-packages \
    torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# ⚠ opencv 는 pip 패키지 버전과 라이브러리 버전이 다르다 — cv2.__version__ 은
# 4.13.0 인데 pip 패키지는 4.13.0.92 다(4자리). __version__ 을 그대로 핀으로
# 쓰면 "No matching distribution" 으로 빌드가 죽는다 (2026-08-26 에 실제로 그랬다).
# 호스트가 임포트하는 것은 opencv-contrib-python 4.13.0.92 인데, 이 코드가 쓰는
# cv2 함수는 전부 코어라(minAreaRect/erode/findContours/calcOpticalFlowPyrLK 등)
# contrib 가 필요 없다. 라이브러리 버전만 맞춘다.
#
# scipy 는 geometry.py:7 이 Rotation/Slerp 로 자세를 직접 계산하는 데 쓰고,
# numpy/opencv 는 마스크·OBB 경로 전체가 탄다. 측정 조건을 고정한다면서
# 이들을 띄워 두면 고정이 절반만 성립한다 — 호스트(측정이 돌던 곳) 값으로
# 맞춘다. 위 네 개와 같은 이유다.
RUN pip3 install --break-system-packages --ignore-installed \
    transformers==5.5.0 open3d==0.19.0 \
    numpy==2.5.0 scipy==1.17.1 opencv-python==4.13.0.92 pillow==12.1.1 \
    rosbags==0.11.3 \
    pytest==9.0.2
# pytest 는 이미지 안에서 `pytest src/roboworld_perception/test -q` 로 설치를
# 자기검증하기 위한 것이다. 이게 없으면 Docker 로 시작한 사람은 데이터 없이
# 설치가 정상인지 확인할 수단이 아예 없다.
#
# ⚠ 핀이 필요하다 — 측정 조건이라서가 아니라 **ROS 플러그인 호환** 때문이다.
# 9.1.1 을 깔면 ROS 를 소싱한 셸에서 pytest 가 아예 안 뜬다:
#   PluginValidationError: Plugin 'launch_testing' for hook
#   'pytest_pycollect_makemodule' ... {'path'} ... can not be found in the hookspec
# jazzy 의 launch_testing / launch_testing_ros 가 옛 훅 시그니처를 쓴다.
# 9.0.2(호스트와 같은 버전)는 정상이다. 2026-08-26 에 이미지에서 실측했다.

# ──────────────────────────────────────────
# 3. 소스 복사 + colcon 빌드
# ──────────────────────────────────────────
WORKDIR /ws
COPY src ./src
COPY scripts ./scripts
COPY run.sh docker-entrypoint.sh ./
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

# ──────────────────────────────────────────
# 4. 모델 가중치는 이미지에 포함하지 않는다 (facebook/sam3는 Meta gated
#    라이선스 — 재배포 불가). /cache/hf 볼륨 + HF_TOKEN으로 첫 실행 시
#    자동 다운로드되고 이후 재사용된다.
# ──────────────────────────────────────────
ENV HF_HOME=/cache/hf

ENTRYPOINT ["/ws/docker-entrypoint.sh"]
CMD ["./run.sh"]
