# 컨베이어 텍스트 물체 인식 + 3D OBB 자세 추정 — 설계

날짜: 2026-08-07 · 상태: 사용자 승인된 방향(권장안 3건 채택) 기반

## 목표

텍스트(예: `물통`)로 지정한 물체를 RGB-D 스트림에서 검출·분할·추적하고,
마스크+Depth로 3D 중심·크기·OBB 자세(RPY/quaternion)를 계산해
디버그 영상·RViz Marker·`vision_msgs/Detection3DArray`로 출력한다.
rosbag(test2/test3)과 D455 실시간 입력 모두 지원. CSV로 프레임별 결과 저장.

## 확정된 기술 결정

- **SAM**: transformers 5.5.0 내장 `Sam3Model`/`Sam3Processor` + gated `facebook/sam3`
  (SAM 3.1 체크포인트는 transformers 미통합이라 제외. 사용자 HF 접근 승인 필요)
- **추적**: 프레임별 SAM3 텍스트 검출 + 2D box IoU greedy 매칭 트래커 (경량, ROS 스트리밍 친화)
- **Docker**: 이번 범위에서 제외 (네이티브 WSL2 검증 후 별도 단계)
- **Python 환경**: 시스템 python3.12 (torch 2.10+cu128 기설치, rclpy와 동일 인터프리터)
- **한국어 프롬프트**: SAM3 텍스트 인코더는 영어 기반 → `물통→thermos` 등 별칭 테이블로 변환
  (프롬프트 단어에 따라 score가 2배 이상 차이남: water bottle 0.45 vs thermos 0.9 — 불안정하면 threshold보다 단어를 먼저 교체)

## 데이터 (bag 실측)

- 토픽: `/camera/camera/color/image_raw`(~18fps), `/camera/camera/aligned_depth_to_color/image_raw`(16UC1 mm), 각 `camera_info`
- test2: 13.5s 정지 장면(검출 품질 확인), test3: 20.4s 이동+3물체(추적·안정성 확인)
- frame_id는 camera_info 헤더 값을 그대로 사용 (TF 변환 없음, 카메라 광학 프레임 기준 출력)

## 구조

```
src/roboworld_perception/          # ament_python 패키지
  roboworld_perception/
    sam3_detector.py   # 텍스트 프롬프트 → [Detection(mask, box, score, label)]
                       #   vision embedding 1회 계산 후 다중 프롬프트 재사용
    geometry.py        # mask+depth+K → 포인트 → Open3D OBB(robust) → center/extent/R/RPY
                       #   depth median±MAD 클리핑, 마스크 침식으로 경계 bleed 제거
                       #   stabilize_R: 이전 R과 축 순열·부호 매칭(뒤집힘 방지) + quaternion nlerp
    tracker.py         # IoU greedy 매칭, ID 유지, missed>N 제거, center/extent EMA
    pipeline.py        # 위 3개 결합: (rgb, depth, K, prompts) → [TrackedObject] (offline/ROS 공용)
    overlay.py         # 마스크·3D박스 투영·XYZ축·라벨 OpenCV 렌더링
    perception_node.py # ROS2 노드: 동기 구독(color+depth), /perception/{detections,markers,debug_image}
                       #   /perception/prompt(String)로 런타임 프롬프트 교체
  launch/perception.launch.py
  test/test_geometry.py, test/test_tracker.py   # 합성 데이터 단위 검증
scripts/run_offline.py # mcap 직접 읽기(rosbags) → mp4 + CSV + 안정성 지표 요약
```

## 검증

1. 단위 테스트: 합성 박스 포인트로 OBB 복원, 축 뒤집힘 교정, ID 유지 (pytest)
2. test2 offline: `물통/마우스/필통` 프롬프트별 검출·마스크·오버레이 확인
3. test3 offline: ID 유지율, 중심/크기/RPY 프레임별 표준편차, 뒤집힘 횟수, FPS → CSV+요약
4. ROS: `ros2 bag play` + 노드 → 토픽 echo, RViz Marker, 디버그 영상

## 알려진 제약

- `facebook/sam3` 접근 승인 전에는 모델 추론 불가 (승인 후 자동 다운로드)
- 회전 대칭 물체(물통)의 yaw는 기하학적으로 비결정 — 축 연속성 안정화만 보장
- 절대 자세 정답이 없어 test3는 반복성/시간 안정성 지표로 평가
