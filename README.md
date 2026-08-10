<div align="center">

# pose-anything

**Zero-shot object detection, tracking, and 6-DoF pose estimation on ROS 2.**

*SAM 3 · Open3D · no training · no CAD models*

[![SAM 3](https://img.shields.io/badge/SAM_3-Meta_AI-0467DF?logo=meta&logoColor=white)](https://github.com/facebookresearch/sam3)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#requirements)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_bf16-EE4C2C?logo=pytorch&logoColor=white)](#requirements)
[![Open3D](https://img.shields.io/badge/Open3D-0.19-000000)](http://www.open3d.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

![Tracking three objects on a moving conveyor](docs/images/demo_tracking.png)

<sub>**Moving conveyor, three text prompts** (`블록`/block, `책`/book, `장갑`/glove) — each object keeps
a single track ID for the entire 20-second sequence with zero axis flips. Overlay shows mask,
projected 3D bounding box, local XYZ axes, distance, size, and RPY per object.</sub>

![Multi-object detection on a static scene](docs/images/demo_static.png)

<sub>**Static scene, four prompts** — thermos, laptop, book, smartphone. Estimated OBB sizes match
the real objects within ±1 cm.</sub>

## How it works

```mermaid
flowchart LR
    A["RGB + aligned depth<br/>(rosbag or RealSense D455)"] --> B{"every Nth frame"}
    B -- "keyframe" --> C["SAM 3<br/>text-prompted detection<br/>+ segmentation"]
    B -- "in between" --> D["Lucas-Kanade optical flow<br/>mask propagation"]
    C --> E["IoU tracker<br/>persistent IDs"]
    D --> E
    E --> F["mask + depth + K<br/>→ point cloud (MAD filter)<br/>→ Open3D OBB"]
    F --> G["temporal stabilization<br/>axis continuity · deadband · slerp"]
    G --> H["Detection3DArray<br/>RViz markers · debug video · CSV"]
```

Running the 848M-parameter SAM 3 on every frame caps throughput at ~3 FPS.
The hybrid scheme runs it only on keyframes (default: every 5th frame) and tracks
masks with optical flow in between, reaching **~9 FPS** on an RTX 4070 Ti —
while the 3D pose is still recomputed **every frame** from that frame's real depth.

Since OBB axes are arbitrary up to permutation and sign, the stabilizer matches each
new OBB against the previous frame's axes, ignores sub-2° jitter (deadband), and
slerps the rest — bringing yaw noise down from 3.5° to **0.94° per frame** with zero
90°/180° axis flips.

## Measured results

Validated on self-recorded rosbags (13 s static / 20 s moving conveyor, not included in this repo):

| Metric | Result |
|---|---|
| Track ID persistence (3 moving objects, 20 s) | single ID each, 0 axis flips |
| OBB size vs. real objects | within ±1 cm |
| Center jitter (static objects) | ≤ 1.3 mm std |
| Yaw jitter | 0.94°/frame avg, 0% jumps > 5° |
| Throughput (3 prompts, RTX 4070 Ti) | ~9 FPS |

## Requirements

**With Docker** the host only needs an NVIDIA driver, Docker + nvidia-container-toolkit,
and a Hugging Face account — everything below ships inside the image. For native installs:

- Ubuntu 24.04 (verified on WSL2) with **ROS 2 Jazzy**
- NVIDIA GPU with ≥ 6 GB VRAM, PyTorch CUDA build (bf16 inference)
- Python 3.12 — `transformers>=5.5`, `open3d`, `rosbags`, `scipy`, `opencv-python`
- Intel RealSense D455 + [`realsense2_camera`](https://github.com/IntelRealSense/realsense-ros) for live input
- A Hugging Face account with access to the gated
  [`facebook/sam3`](https://huggingface.co/facebook/sam3) checkpoint
  (click *"Agree and access repository"* on the model page — approval is immediate)

## Quick start

### Docker (recommended — any Ubuntu PC with an NVIDIA GPU)

Requires Docker and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone https://github.com/ingon1026/pose-anything.git
cd pose-anything
export HF_TOKEN=hf_xxxx            # token of an account with facebook/sam3 access
docker compose build               # ~21 GB (CUDA PyTorch)
docker compose run --rm perception                                    # live camera
docker compose run --rm perception ./run.sh bags/mybag --prompts "book"   # rosbag
```

Model weights are **not** baked into the image (Meta's gated license) — they are
downloaded once on first run into a mounted cache volume and reused afterwards.
Put rosbags under `./bags/` (mounted into the container); add `--headless` on
machines without a display.

### Native install

```bash
git clone https://github.com/ingon1026/pose-anything.git
cd pose-anything
pip install --user transformers open3d rosbags scipy opencv-python
hf auth login                      # account with facebook/sam3 access
colcon build --symlink-install
```

```bash
./run.sh                           # live RealSense camera
./run.sh path/to/rosbag            # play a recorded bag
./run.sh path/to/rosbag --prompts "book,glove"   # non-interactive
```

The script asks one question — which objects to find. Type a name like `book`
(or a Korean alias like `책`) and **only the prompted objects are detected and
tracked** — one best-scoring instance per prompt, with a persistent track ID.
Everything else in the scene is ignored.

It then starts the perception node, RViz (preset with 3D boxes and axes), and a
full-size debug window. Per-frame results are logged to `output/ros_<timestamp>.csv`.

Offline processing (no ROS runtime — reads the bag directly, writes mp4 + CSV):

```bash
python3 scripts/run_offline.py --bag path/to/rosbag --prompts "book" --show
```

### Prompts

SAM 3's text encoder is English-based. These Korean words are translated
automatically (see `PROMPT_ALIASES` in `sam3_detector.py`); use English for
anything else:

> 물통 · 마우스 · 필통 · 노트북 · 책 · 스마트폰 · 장갑 · 천 · 블록

**If detection is unstable, change the word before touching the threshold.**
The same green thermos scores 0.45 as `"water bottle"` but 0.91 as `"thermos"`.
Prompts can be swapped at runtime:

```bash
ros2 topic pub --once /perception/prompt std_msgs/String "data: thermos"
```

## ROS 2 interface

![ROS 2 node graph](docs/images/rqt_graph.png)

**Subscribes** — `/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`,
`.../camera_info`, `/perception/prompt`

**Publishes**

| Topic | Type | Content |
|---|---|---|
| `/perception/detections` | `vision_msgs/Detection3DArray` | label, score, track ID, pose (center + quaternion), size — in `camera_color_optical_frame` |
| `/perception/markers` | `visualization_msgs/MarkerArray` | OBB cube, XYZ axes, label text for RViz |
| `/perception/debug_image` | `sensor_msgs/Image` | mask + 3D box + status overlay |

**Parameters** — `prompts`, `score_threshold` (0.4), `detect_interval` (5, SAM keyframe period),
`max_per_prompt` (1, tracks per prompt), `csv_path`, `display`

## Project structure

```
src/roboworld_perception/         ROS 2 package (ament_python)
  roboworld_perception/
    sam3_detector.py              SAM 3 wrapper, prompt aliases
    pipeline.py                   hybrid detect/track orchestration
    tracker.py                    IoU tracker, pose stabilization
    geometry.py                   back-projection, OBB, axis matching
    overlay.py                    debug rendering
    perception_node.py            ROS 2 node
  rviz/perception.rviz            RViz preset
  test/                           unit tests (pytest, 14 cases)
scripts/run_offline.py            bag → mp4/CSV without ROS
run.sh                            single entry point
```

## Acknowledgments

This project builds on the following open-source work:

- **[SAM 3 — Segment Anything with Concepts](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)**
  (Meta AI) — open-vocabulary detection, segmentation, and tracking.
  Used via [Hugging Face Transformers](https://github.com/huggingface/transformers)
  (`Sam3Model`). Model weights are **not** distributed with this repository; they are
  downloaded from [`facebook/sam3`](https://huggingface.co/facebook/sam3) under Meta's
  own license terms after gated-access approval.
- **[Open3D](http://www.open3d.org/)** (Zhou, Park, Koltun) — point-cloud processing
  and oriented-bounding-box estimation.
- **[ROS 2](https://docs.ros.org/en/jazzy/)** / **[vision_msgs](https://github.com/ros-perception/vision_msgs)** — middleware and message definitions.
- **[librealsense / realsense-ros](https://github.com/IntelRealSense/realsense-ros)** (Intel) — D455 camera driver.
- **[OpenCV](https://opencv.org/)** — optical flow, rendering; **[SciPy](https://scipy.org/)** — rotation math.

## License

Code in this repository is released under the [MIT License](LICENSE).
The SAM 3 model weights are governed by Meta's separate license (see above).
