# TensorRT 도입의 배포 영향과 철수 기준 (2026-08-21)

## 결정: **보류(접음).** 다시 열 조건은 아래 "재개 조건"

관문(ONNX export)은 통과했는데도 접는다 — **태울 수 있는 범위가 상한에 걸리기
때문**이다. 도구를 ONNX Runtime 등으로 바꿔도 이 상한은 안 변한다.

SAM3 내부 분해 실측(test4 640x480, 프롬프트 3개, 중앙값 10회):

| 구간 | ms | SAM3 내 비중 |
|---|---|---|
| 전처리(CPU) | 6.8 | 2% |
| **vision_encoder** (프레임당 1회) | **237.3** | **57%** ← export 가능 확인 |
| text_encoder (프롬프트마다) | 29.7 | 7% |
| 디코더류 (프롬프트마다) | 138.4 | 33% |
| 후처리 | 1.9 | 0% |
| 합계 | 414.0 | |

파이프라인 전체 시간 분해(CSV proc_ms 실측): 파이프라인 밖(mp4·bag) 28~30%,
키프레임은 5프레임에 1회이고 그 91%가 SAM3 → **SAM3 는 전체 wall 의 약 56%**.
(교차 검증: export 팀의 SAM3 414ms 와 CSV 의 test4 키프레임 455ms 가 91% 로 일치.
2026-08-20 devlog 의 "키프레임 180ms" 는 이 측정과 안 맞는다 — 91% 만 맞았다.)

암달의 법칙:

| 태우는 범위 | 가속 2배 시 전체 체감 | 상한(∞) |
|---|---|---|
| vision 만 (오늘 검증된 범위) | **1.20배** | 1.51배 |
| SAM3 전부 (디코더까지) | 1.39배 | 2.27배 |

deploy 조사의 GO 기준은 "SAM3 단독 ≥2배" 인데, vision 이 SAM3 의 57% 라
**vision 을 무한대로 빠르게 해도 SAM3 는 1.75배가 상한**이다. 기준을 구조적으로
만족할 수 없다. 1.2배를 위해 아래의 배포 대가(지원 GPU 축소·Docker 빌드 단계·
CUDA 버전 함정)를 지불할 이유가 없다.

**디코더까지 태우는 안(상한 2.27배)을 안 고른 이유**: 이 프로젝트의 핵심 가치는
제로샷이고, 그 실사용 형태가 `/perception/prompt` 런타임 프롬프트 교체다.
디코더를 고정 shape 엔진으로 구우면 그 유연성이 위태로워진다. 속도를 위해
제품 정체성을 거는 교환이라 성격이 다르다.

### 관문 통과 기록 (다시 열 때 여기서 이어받을 것)
- `torch.onnx.export(dynamo=True)`, opset 18, fp32/CUDA, 입력 `(1,3,1008,1008)` 고정
  → **첫 시도 성공**. 산출 1.8GB(외부 데이터 분리)
- 출력 9개를 평탄한 튜플로 래핑해야 한다(dataclass + 중첩 튜플은 ONNX 가 못 받음)
- 수치 일치(onnxruntime CPU vs torch CUDA): 최대 상대오차 **4.7e-4**, 위치인코딩 6e-8
- 재현 스크립트: `scripts/bench_sam3_parts.py`, `scripts/export_sam3_vision.py`,
  `scripts/verify_sam3_onnx.py`
- **설치 함정**: `pip install tensorrt` 가 `tensorrt_cu13`(CUDA 13)로 해결된다.
  우리 torch 는 cu128 → `tensorrt-cu12` 로 명시 고정 필요(dry-run 확인)

### 재개 조건
1. 엣지 타겟 하드웨어가 정해져서 **그 장비 한 대에만** 엔진을 구우면 되는 때
   (이식성 문제가 사라진다 — 위 배포 대가의 대부분이 여기서 나온다)
2. 디코더까지 태울 필요가 생겼고, 런타임 프롬프트 교체를 포기해도 되는 용도가 명확할 때
3. INT8 이 필요할 만큼 속도 요구가 올라갔을 때 — 단 그때는 소물체 42~76px 생존을
   `scripts/check_accuracy.py` 로 먼저 재고 판단할 것

---

조사 전용 문서. 코드 변경 없음. **결론: 조건부 가능 — 단 현행 배포 약속을 깨거나
`TensorRT-RTX` 로 경로를 바꿔야 하고, 손익분기가 생각보다 훨씬 높다.**

우리가 깨게 되는 약속(README 원문):
> Docker (recommended — **any Ubuntu PC with an NVIDIA GPU**)
> the host only needs an NVIDIA driver, Docker + nvidia-container-toolkit, and a Hugging Face account
> NVIDIA GPU with **≥ 6 GB VRAM**

TensorRT 엔진은 GPU 아키텍처에 묶인다. 이 두 문장이 그대로 유지되지 않는다.

> **인용한 `≥ 6 GB VRAM` 은 전용 VRAM 이 있는 카드 전제다** — 아래 §2 의 "2080 Ti
> (11 GB)는 이 요구사항을 만족한다" 는 논증도 그 전제 위에 선다. 통합 메모리
> 기계에서는 그 비교 자체가 안 된다(DGX Spark/GB10 은 `nvidia-smi` 가 총량을
> 못 낸다). README 쪽에도 같은 조건을 달아 뒀다
> → [shared_server](shared_server_2026-08-31.md) §1

---

## 1. 설치 조합

현재 환경: torch 2.10.0+cu128, CUDA 12.8, 드라이버 591.86, RTX 4070 Ti(Ada, **CC 8.9**),
Python 3.12, 베이스 이미지 `ros:jazzy-ros-base`(CUDA 베이스 아님 — CUDA 런타임은
pip torch 휠이 들고 온다).

| 조합 | 판정 | 근거 |
|---|---|---|
| `torch-tensorrt==2.10.0` | ⚠️ **위험** | torch 핀은 정확히 맞는다(`torch>=2.10.0,<2.11.0`). 그러나 `nvidia-cuda-runtime-cu13==0.0.0a0` 을 요구한다 — **CUDA 13 런타임인데 우리 torch 는 cu128**. PyPI 메타데이터 직접 조회 |
| `torch-tensorrt==2.13.0`(최신) | ❌ | PyTorch 2.13 / CUDA 13.2 / TensorRT 11.0 기준. torch 2.10 → 2.13 업그레이드 필요 = **중대 비용**(지시서상 금지 항목) |
| plain `tensorrt`(ONNX 경로) | ⚠️ **조건부** | **torch 에 의존하지 않는다** → torch 를 안 건드린다. 이게 가장 큰 장점. 단 최신 `tensorrt` 는 `tensorrt_cu13==11.2.1.2` 를 끌고 온다(CUDA 13). CUDA 12 를 쓰려면 `tensorrt-cu12` 를 **명시 지정**해야 한다 |
| `tensorrt-cu12` 를 10.14.x 로 핀 | ❓ **미확인** | `torch-tensorrt 2.10.0` 이 요구하는 범위는 `tensorrt>=10.14.1,<10.15.0`. `tensorrt-cu12` 의 최신은 11.2.1.2 라 10.14.x 계열이 CUDA12 빌드로 존재하는지 **해석 테스트를 해야 확정**된다(설치 금지라 미실행) |
| `tensorrt-rtx` | ⚠️ 조건부 | `tensorrt_rtx_cu13==1.6.1.120` (CUDA 13). **배포 이식성은 이 중 최고** — 3절 참고 |
| NGC PyTorch 컨테이너로 이전 | ❌ 현실적으로 | 검색 결과가 권장하는 회피책이지만, 우리 베이스는 `ros:jazzy-ros-base` 다. ROS 2 Jazzy 전체를 NGC 위로 다시 쌓는 작업 |

**PyPI 함정 하나**: `tensorrt*` PyPI 항목은 전부 **0 MB 스텁 sdist** 다(직접 조회 확인).
설치 시 NVIDIA 인덱스에서 실제 페이로드를 받아온다. 즉 **Docker 빌드가 NVIDIA 서버
네트워크에 의존**하고, 용량을 PyPI 로는 알 수 없다.

> 우리 자신의 전례: 이 Dockerfile 은 이미 `--ignore-installed`(데비안 psutil 충돌)와
> `--break-system-packages` 가 필요했고, cv_bridge 는 NumPy 2.5 ABI 충돌로 아예 버렸다.
> 이 환경의 의존성은 깨지기 쉽다고 가정하는 편이 실측에 맞는다.

---

## 2. 엔진이 묶이는 축

| 축 | 기본 동작 | 완화 수단 | 대가 |
|---|---|---|---|
| **TensorRT 버전** | 빌드한 버전에서만 동작 | version-compatible 빌드 | "특정 런타임 버전용 엔진보다 느릴 수 있다" |
| **Compute capability** | major·minor 를 plan 에 기록하고 **정확히 일치**해야 함. 불일치 시 **역직렬화 실패** | `kAMPERE_PLUS`(CC ≥ 8.0) / `kSAME_COMPUTE_CAPABILITY` | "throughput 이 낮거나 latency 가 높을 수 있다". 공유메모리를 많이 쓰는 tactic 이 배제됨 |
| **OS + CPU 아키텍처** | 동일 플랫폼에서만 (Linux↔Windows 불가) | cross-platform 빌드 | "네이티브 빌드와 성능 차이가 있을 수 있다" |
| **GPU 모델** | 명시적 구속은 아님. 메모리 버스폭·L2·shared memory 등 device property 를 검사해 **불일치 시 경고** | — | 같은 CC 안에서의 이식 가능 여부는 문서에 명시 없음(**미확인**) |

### 배포상 결과 — 누가 탈락하는가

| 엔진 모드 | 돌아가는 GPU | 탈락하는 GPU |
|---|---|---|
| 기본(구속) | **4070 Ti 계열(CC 8.9)뿐** | 30xx(8.6) 포함 사실상 전부 |
| `kAMPERE_PLUS` | CC ≥ 8.0 — 30xx / 40xx / 50xx / A100 등 | **Turing(RTX 20xx, CC 7.5)**, **Pascal(GTX 10xx)** |

`kAMPERE_PLUS` 로 넓혀도 **RTX 2080 Ti(11 GB)** 같은 GPU 가 배제된다. 이건 현행
요구사항 "≥ 6 GB VRAM" 을 만족하는 장비다 — 즉 **지금 문서상 지원 대상인 사용자를
새로 탈락시킨다.** Docker Hub 이미지 하나로 배포하는 현재 모델과 정면으로 충돌하며,
선택지는 (a) CC 별 엔진 다중 동봉 (b) 런타임 빌드 (c) 지원 범위 축소 명시 셋뿐이다.

---

## 3. TensorRT-RTX — 배포 문제의 실질적 해법 후보

`TensorRT for RTX` 는 소비자 RTX 배포를 겨냥한 별도 제품이고, 우리 문제에 정확히 대응한다.

- **CPU-only AOT 컴파일** — 빌드에 GPU 가 필요 없고, 산출 엔진이 "Windows 와 Linux,
  CC 8.6 이상 RTX GPU 로 이식 가능"
- 기본 지원 CC: **8.6 / 8.9 / 12.0 / 12.1 이상**. Turing(7.5)은 API 로 명시 타겟 가능하나
  "새 아키텍처와 섞으면 성능이 떨어질 수 있다"
- 실행 컨텍스트 생성 시점에 **weightless 엔진 메타데이터로부터 커널을 JIT 컴파일**

**이게 진짜 해법인지는 JIT 소요 시간에 달려 있다 — 그리고 그 값은 미확인이다.**
JIT 시간은 곧 기동 워밍업 비용이고, **`torch.compile` 을 탈락시킨 항목이 정확히 그것**이다
(7% 이득 / 워밍업 30초, 2026-08-10). 같은 저울에 같은 항목이 다시 올라온다.
컨테이너 첫 실행마다 수십 초가 든다면 `docker compose run --rm` 을 쓰는 현재 UX 에서
매 실행 비용이 된다(캐시 지속 여부 **미확인**).

---

## 4. Docker 이미지 영향

**정확한 증가분은 미확인이다** — PyPI 스텁 구조 때문에 조회로는 알 수 없다.
확인된 사실만:

- 현행 이미지 ~21 GB (README)
- 베이스가 CUDA 이미지가 아니라 `ros:jazzy-ros-base` 이므로, CUDA 런타임은 pip torch 휠이
  들고 온 `nvidia-*` 패키지들이다. TensorRT 도 같은 계열 패키지를 별도로 들고 온다 →
  **CUDA 라이브러리가 이중으로 들어갈 소지**가 있고, 그게 용량과 충돌 양쪽의 원인이다
- 런타임만 넣는 것과 빌더(엔진을 굽는 쪽)까지 넣는 것의 차이가 크다. 런타임 빌드(2절 (b))를
  택하면 **빌더를 반드시 이미지에 넣어야** 한다

**측정 방법(부모용)**: Dockerfile 의 pip 레이어에만 TensorRT 를 추가한 브랜치를 빌드해
`docker image ls` 로 전후를 비교. 빌드 1회면 확정된다.

---

## 5. 폴백 설계

**필요하다.** 위 2절대로 엔진이 안 맞으면 **역직렬화 실패**로 죽는다. 조용히 느려지는 게
아니라 기동이 안 된다.

이 저장소에는 이미 맞는 관례가 있다 — `enable_merge`, `pub_score_min`, `use_belt_plane`
전부 **기본 꺼짐 + 씬별로 켜기**이고, 그 이유가 "한 환경에서만 보정/검증된 것을 기본으로
켜면 순수한 위험"이다. TensorRT 는 그 논리가 더 강하게 적용된다(검증된 환경이 문자 그대로
이 GPU 한 대다).

권고 형태: `use_tensorrt` 파라미터 **기본 꺼짐**, 엔진 로드 실패 시 경고 로그 1줄과 함께
현행 PyTorch 경로로 자동 폴백. 두 경로의 출력이 같은지는 verify 팀 하네스로 판정.

---

## 6. 철수 기준 (go/no-go)

### 먼저 — 손익분기가 생각보다 훨씬 높다

"키프레임 180 ms 중 SAM3 91%" 는 **키프레임 한 장의 지연**이다. 우리는 하이브리드라
**5프레임에 1번만** 키프레임이므로, 이 수치를 전체 처리량 개선으로 읽으면 안 된다.

파이프라인 내부만 놓고 detect_interval=5, flow 프레임 13 ms(오늘 로그 실측) 가정:

| SAM3 배속 | 5프레임 사이클 | 파이프라인 FPS |
|---|---|---|
| 현행 1× | 180 + 4×13 = 232 ms | 21.6 |
| 2× | 98 + 52 = 150 ms | 33.3 |
| 3× | 71 + 52 = 123 ms | 40.7 |
| ∞ | 16 + 52 = 68 ms | 73.5 |

그런데 **실측 end-to-end 는 offline runner 기준 5.6~8.7 FPS** 다(오늘 로그: isaac 8.70,
test4 7.14, test2 5.59). 파이프라인 추정 21.6 FPS 와의 차이는 전부 파이프라인 **밖**
비용이다(bag 디코드·오버레이·mp4 인코딩 등, 프레임당 대략 60~70 ms 로 추정 — **미확인**).

**결론: SAM3 를 아무리 깎아도 그 바깥 비용이 그대로면 end-to-end 개선은 작다.**
TensorRT 에 반나절 이상 쓰기 전에 **"end-to-end 시간이 실제로 어디에 쓰이는지"를 먼저
분해**하는 게 순서다. 그건 GPU 도 설치도 필요 없고 한 시간이면 끝난다.

### GO 조건 (전부 만족해야)
1. SAM3 서브모듈 단독 **≥ 2배** — `torch.compile` 7% 전례가 있으므로 1.5배 미만이면 노이즈
2. verify 팀 하네스의 **소물체 합격선 통과**(스마트폰 42~76 px 생존)
3. **torch 를 건드리지 않고** 설치 가능 (1절: plain `tensorrt` 경로만 해당)
4. 기동 워밍업 **≤ 5초** 또는 캐시로 1회성 — 30초면 `torch.compile` 과 같은 이유로 탈락
5. 폴백 경로가 동작 (엔진 불일치 환경에서 죽지 않고 PyTorch 로 내려감)

### NO-GO (하나라도 걸리면 접는다)
- 이득 < 1.5배
- 소물체 회귀 — **속도가 몇 배든 무조건 탈락.** 8/10 에 560px 로 스마트폰을 잃고 되돌린 전례
- torch 업그레이드 또는 CUDA 13 전환이 필요
- 워밍업 > 30초이고 캐시 불가
- `kAMPERE_PLUS` 로도 못 살리는 지원 범위 축소를 감수 못 함

### 지금 시점 권고
**본격 투자 전에 6절 첫머리의 end-to-end 분해부터.** 그 결과 파이프라인 밖 비용이
지배적이면 TensorRT 는 우선순위가 아니고, 지배적이지 않으면 그때 GO 조건 1·2 를 잰다.

---

## 인용 출처
- [Engine Compatibility — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [Version Compatibility — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/version-compatibility.html)
- [CPU-Only AOT and TensorRT-RTX Engines — NVIDIA TensorRT for RTX](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/cpu-engines.html)
- [torch-tensorrt · PyPI](https://pypi.org/project/torch-tensorrt/) (및 PyPI JSON API 직접 조회)
- [Pip install fails due to deprecated nvidia-cuda-runtime-cu13 · NVIDIA/TensorRT#4614](https://github.com/NVIDIA/TensorRT/issues/4614)
- [Installing tensorrt 10.0.1 pypi package installs wrong version of tensorrt-cu12 · NVIDIA/TensorRT#3945](https://github.com/NVIDIA/TensorRT/issues/3945)
- 우리 쪽 사실: `README.md`, `Dockerfile`, `docker-compose.yml`, 2026-08-21 회귀 로그
