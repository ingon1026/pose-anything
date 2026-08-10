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
    ros-jazzy-rosbag2-storage-mcap \
    fonts-noto-cjk \
    libgl1 libgomp1 usbutils \
  && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────
# 2. pip: PyTorch CUDA (가장 무거운 레이어 — 소스 복사보다 먼저 캐시)
# ──────────────────────────────────────────
RUN pip3 install --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --break-system-packages --ignore-installed \
    "transformers>=5.5" open3d rosbags scipy opencv-python pillow

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
