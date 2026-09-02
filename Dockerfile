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
# 이 블록은 아키텍처로 갈리지 않는다 — 위 여섯 패키지가 packages.ros.org 의
# binary-arm64 에 amd64 와 **같은 이름으로 전부** 있다(글로브 `librealsense2*`
# 도 매치된다. 매치가 0건이면 apt 가 실패하므로 확인했다 - 2026-08-31 인덱스 조회).

# ──────────────────────────────────────────
# 2. pip: PyTorch CUDA (가장 무거운 레이어 — 소스 복사보다 먼저 캐시)
# ──────────────────────────────────────────
# ⚠ **여기부터 아키텍처가 갈린다** (amd64 워크스테이션 / arm64 DGX Spark).
# TARGETARCH 는 BuildKit 이 자동으로 채우는 예약 ARG 다 — 선언만 하면 값이 온다.
# **legacy builder(`DOCKER_BUILDKIT=0`)에서는 빈 문자열**이라 아래 분기가 전부
# else(amd64) 로 떨어진다. amd64 보존이 최우선이라 fallback 을 그쪽에 뒀다.
# 갈리는 곳은 아래 두 RUN(torch, open3d)뿐이고, 이 선언을 apt 블록 뒤에 둔 것도
# apt 레이어 캐시를 건드리지 않기 위해서다.
ARG TARGETARCH

# ⚠ **`TARGETARCH` 하나만 믿으면 arm64 에서 조용히 틀린 게 깔린다.** 2026-08-31 확인:
# cu128 인덱스에 `torch-2.10.0+cu128-cp312-cp312-manylinux_2_28_aarch64.whl` 이
# **있다.** 그래서 else 가지가 aarch64 에서 실패하지 않고 **CUDA 12.8 빌드를 성공적으로
# 깐다** — CUDA 13.0 기계(GB10)에 맞지 않는 물건이다. 빌드는 그 뒤 open3d 에서야
# 죽는데(PyPI 0.19.0 에 aarch64 휠이 없다), 그때 뜨는 오류는 open3d 를 지목하므로
# **원인이 TARGETARCH 라는 것을 아무도 못 읽는다.** 핀을 의심하며 시간을 쓰게 되고,
# 이 저장소에서 핀을 건드리는 것이 가장 위험한 행동이다(바로 아래 블록).
# (그 실패 양식 자체를 없앴다 — 아래 fallback 이 들어오기 전 얘기다.)
# 그래서 아래 두 분기는 `TARGETARCH` 하나에 기대지 않고 **비었으면 기계에 직접
# 묻는다** — `dpkg --print-architecture` 가 `TARGETARCH` 와 **같은 어휘**
# (`amd64`/`arm64`)를 낸다(2026-09-01 확인). BuildKit 이면 `TARGETARCH` 가 그대로
# 이기고, legacy builder 면 실제 기계로 떨어져 **arm64 legacy 빌드가 죽는 대신
# 성공한다.** buildx 크로스빌드도 `TARGETARCH` 를 채우므로 fallback 을 안 탄다.

# ⚠ torch / transformers / open3d 는 **고정한다.** 이 저장소의 성능·정확도
# 수치는 전부 특정 조합에서 잰 값이고(docs/README.md §5 "측정 조건을 같이
# 적을 것"), 재빌드가 이 셋 중 하나를 올리면 측정 조건 전체가 라벨을 잃는다.
# 아래 값은 그 측정이 실제로 돌던 호스트 조합이다 — docs/README.md §5 가
# 인용한 실행 로그의 `found 2.10.0+cu128` 이 같은 torch 다.
# torchvision 은 torch 와 짝이라 같이 고정한다(따로 두면 cu128 인덱스에서
# 짝이 안 맞는 버전이 잡힌다).
# 올릴 때는 버전만 바꾸지 말고 scripts/check_accuracy.py 로 기준 대비
# 회귀부터 판정할 것.
#
# ⚠ **CUDA 빌드 변종만 아키텍처로 갈린다 — 버전(2.10.0 / 0.25.0)은 양쪽이 같다.**
# arm64(DGX Spark, GB10 Blackwell)는 cu130 이어야 한다 — 그 기계의 툴킷이 CUDA 13.0 이다.
# ⚠ **cu128 에도 aarch64 휠은 있다**(`torch-2.10.0+cu128-cp312-cp312-manylinux_2_28_aarch64.whl`
# - 2026-08-31 인덱스 조회). 즉 **잘못 갈려도 설치는 성공한다** — 그래서 위 가드가 있다.
# 골라야 하는 근거는 "설치되느냐" 가 아니라 **기계의 CUDA 층과 맞느냐** 다
# (cu130 쪽: `torch-2.10.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl` ·
# `torchvision-0.25.0+cu130-...-aarch64.whl` - 같은 날 조회).
# amd64 는 위 측정이 돌던 cu128 조합 그대로다 — else 가지는 분기 전 줄과 동일하다.
# **올릴 때는 두 가지를 같이 올릴 것.** 한쪽만 올리면 조용히 갈라진다.
# 그리고 **arm64 로 잰 값은 amd64 표와 같은 표에 넣지 말 것** — CUDA 층도 GPU 도
# 다르므로 측정 조건이 다른 수치다(docs/README.md §5).
RUN ARCH="${TARGETARCH:-$(dpkg --print-architecture)}" ; \
    if [ "$ARCH" = "arm64" ]; then \
      pip3 install --break-system-packages \
        torch==2.10.0+cu130 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130 ; \
    else \
      pip3 install --break-system-packages \
        torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 ; \
    fi

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
#
# open3d 만 arm64 에서 출처가 다르다 — **PyPI 0.19.0 에 aarch64 휠이 없다.**
# 상류 릴리스 자산에서 직접 받되, **한 pip 호출 안에서 변수로만 갈랐다.** 따로
# 떼어 두 번 install 하면 resolver 가 두 번 돌아 아래 `numpy==2.5.0` 을 open3d 가
# 요구하는 범위로 조용히 갈아치울 수 있고, 그게 이 저장소가 제일 무서워하는
# "측정 조건이 라벨을 잃는" 경로다. 핀을 두 벌 적지 않는 이유도 같다 — 한쪽만
# 올라가 갈라지는 패턴은 `isaac.launch.py` 주석이 이미 금지한 것이다.
# manylinux_2_35 는 노블(glibc 2.39)이 충족한다. `opencv-python` 은 휠 이름이
# `cp37-abi3` 라 헷갈리지만 aarch64 빌드가 있어 그대로 간다.
#
# ⚠ **`main-devel` 은 움직이는 태그다** — 맨 위 `FROM ros:jazzy-ros-base` 와 같은
# 종류의 문제로, **같은 URL 이 예고 없이 다른 파일을 가리킬 수 있다.** 그러면
# 재빌드가 open3d 를 조용히 올리고 측정 조건이 라벨을 잃는다. 오늘(2026-08-31)
# 조회한 앵커: `Content-Length 48,247,379` · `Last-Modified 2026-07-09`.
# 제대로 묶으려면 `--hash=sha256:<...>` 인데 그러려면 46MB 를 받아 해시를 떠야 한다.
# **만료 조건: arm64 수치를 문서에 한 줄이라도 기록하기 전에 해시를 뜰 것.** 그 순간
# 이 휠은 측정 조건의 일부가 되고, 앵커 없이는 재빌드가 조건을 조용히 바꾼다. 그때는
# 이미 빌드가 46MB 를 받은 뒤라 비용도 사실상 0 이다(`pip hash` 로 뜬다).
RUN ARCH="${TARGETARCH:-$(dpkg --print-architecture)}" ; \
    if [ "$ARCH" = "arm64" ]; then \
      OPEN3D=https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-0.19.0-cp312-cp312-manylinux_2_35_aarch64.whl ; \
    else \
      OPEN3D=open3d==0.19.0 ; \
    fi ; \
    pip3 install --break-system-packages --ignore-installed \
    transformers==5.5.0 "$OPEN3D" \
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
