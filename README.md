# roboworld — 텍스트로 지정하는 컨베이어 물체 인식 + 3D 자세 추정

물체 이름을 텍스트로 입력하면(`"물통"`, `"book"`) RGB-D 카메라 영상에서 해당 물체를
**학습 없이(zero-shot)** 찾아 분할·추적하고, 3D 중심 좌표·크기·자세(OBB)를
ROS 2 토픽으로 출력하는 인식 파이프라인입니다.

| 이동 추적 (컨베이어 동작 중) | 다중 물체 검출 (정지 장면) |
|---|---|
| ![tracking](docs/images/demo_tracking.png) | ![static](docs/images/demo_static.png) |

## 특징

- **Open-vocabulary 검출** — Meta SAM3로 임의 텍스트 프롬프트 물체를 클래스 정의·학습 없이 검출·분할
- **하이브리드 검출·추적** — SAM3는 N프레임마다(키프레임), 사이는 Lucas-Kanade 광학흐름으로 마스크 추적 → 단독 SAM 대비 약 3배 처리율 (RTX 4070 Ti 기준 ~9 FPS)
- **기하학적 6D 자세** — 마스크 + aligned depth → 포인트 클라우드 → Open3D OBB. CAD 모델 불필요
- **시간 안정화** — 축 순열·부호 연속성 매칭(뒤집힘 방지), 회전 데드밴드 + slerp, 중심·크기 EMA → yaw 떨림 0.94°/frame
- **rosbag / 실시간 겸용** — 같은 노드가 `ros2 bag play`와 RealSense D455 라이브 입력을 동일하게 처리

## 검증 결과

자체 촬영 rosbag(정지 장면 13초 / 컨베이어 동작 20초, 저장소 미포함) 기준:

| 항목 | 결과 |
|---|---|
| 이동 물체 추적 (20초, 3물체) | 전 구간 단일 ID 유지, 축 뒤집힘 0회 |
| 3D 크기 추정 | 실측 대비 ±1cm 이내 (책 0.27×0.22m 등) |
| 중심 안정성 (정지 물체) | 표준편차 ≤ 1.3mm |
| 자세 안정성 | yaw 프레임간 변화 평균 0.94°, 5° 이상 점프 0% |
| 처리 속도 | ~9 FPS (bf16, 하이브리드, 3프롬프트) |

## 요구사항

- Ubuntu 24.04 (WSL2 검증됨) + ROS 2 Jazzy
- NVIDIA GPU (VRAM 6GB+, bf16 지원 권장) + PyTorch CUDA 빌드
- Python 3.12: `transformers>=5.5` `open3d` `rosbags` `scipy` `opencv-python`
- Intel RealSense D455 (실시간 입력 시) + `realsense2_camera`
- **Hugging Face 계정** — [facebook/sam3](https://huggingface.co/facebook/sam3)은 gated 모델이라 페이지에서 접근 동의 후 `hf auth login` 필요

## 설치

```bash
git clone https://github.com/ingon1026/roboworld.git
cd roboworld
pip install --user transformers open3d rosbags scipy opencv-python
hf auth login                      # facebook/sam3 접근 승인된 계정
colcon build --symlink-install
```

## 사용법

```bash
./run.sh                          # 실시간 카메라 (D455)
./run.sh bags/test3               # rosbag 재생
./run.sh bags/test3 --prompts "책,장갑"   # 무인 실행
```

실행하면 물체 이름 하나만 물어보고, 노드 + RViz(3D 박스·축) + 전체 크기
디버그 창이 자동으로 뜹니다. 결과는 `output/ros_<시각>.csv`에 프레임별로 기록됩니다.

mp4·CSV만 필요한 오프라인 처리:

```bash
python3 scripts/run_offline.py --bag bags/test3 --prompts "책" [--show]
```

### 프롬프트

SAM3 텍스트 인코더는 영어 기반입니다. 아래 한국어 단어는 자동 변환되며
(`sam3_detector.py`의 `PROMPT_ALIASES`), 그 외 물체는 영어로 입력하세요.

> 물통, 마우스, 필통, 노트북, 책, 스마트폰, 장갑, 천, 블록

검출이 불안정하면 threshold보다 **단어를 먼저 바꿔보세요** — 같은 물체도
"water bottle"(score 0.45)과 "thermos"(0.91)처럼 단어에 따라 크게 달라집니다.

## ROS 2 인터페이스

**구독**: `/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`,
`.../camera_info`, `/perception/prompt`(std_msgs/String — 런타임 물체 교체)

**발행**:

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/perception/detections` | `vision_msgs/Detection3DArray` | 물체별 라벨·score·추적 ID·pose(중심+quaternion)·크기 |
| `/perception/markers` | `visualization_msgs/MarkerArray` | RViz용 OBB 큐브·XYZ축·라벨 |
| `/perception/debug_image` | `sensor_msgs/Image` | 마스크·3D박스·상태줄 오버레이 영상 |

**주요 파라미터**: `prompts`, `score_threshold`(0.4), `detect_interval`(5, SAM 키프레임 주기),
`max_per_prompt`(1, 물체당 트랙 수), `csv_path`, `display`

## 파이프라인

```
텍스트 프롬프트
      │
색상+depth ──► [키프레임] SAM3 검출·분할 ──► IoU 매칭 (ID 유지)
      │        [중간 프레임] 광학흐름 마스크 추적
      │
      └─► 마스크+depth+K ─► 포인트 역투영(MAD 필터) ─► Open3D OBB
                                                        │
           축 연속성·데드밴드·slerp·EMA 안정화 ◄────────┘
                    │
      Detection3DArray · RViz Marker · 디버그 영상 · CSV
```

## 프로젝트 구조

```
src/roboworld_perception/       ROS 2 패키지 (ament_python)
  roboworld_perception/
    sam3_detector.py            SAM3 래퍼, 프롬프트 별칭
    pipeline.py                 하이브리드 검출·추적 오케스트레이션
    tracker.py                  IoU 트래커, 자세 안정화
    geometry.py                 역투영, OBB, 축 매칭
    overlay.py                  디버그 렌더링
    perception_node.py          ROS 2 노드
  rviz/perception.rviz          RViz 프리셋
  test/                         단위 테스트 (pytest)
scripts/run_offline.py          ROS 없이 bag → mp4/CSV
run.sh                          단일 진입점
```

## 로드맵

- [ ] Docker / docker-compose 배포 환경
- [ ] 가림(occlusion) 대응 — score 급락 시 트랙 동결
- [ ] 칼만 필터 기반 이동 지연 보정 (로봇 그리핑 연동용)
- [ ] 카메라→로봇 베이스 외부 캘리브레이션

## 라이선스

MIT
