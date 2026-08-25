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

RUN pip3 install --break-system-packages --ignore-installed \
    transformers==5.5.0 open3d==0.19.0 rosbags scipy opencv-python pillow

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
