# Dataset inventory (rosbags)

Bags are **not** committed to this repo (see `.gitignore`) — they live on the
dev machine at `~/roboworld/bags/` with tar backups on the Windows desktop
(`bags.tar.gz`, `bags2.tar.gz`). **Do not delete** — besides their original
test purpose, all bags are reserved as seed data for the planned
SAM3-auto-label → lightweight-YOLO distillation pipeline.

| Bag | Recorded | Scene | Camera | Used for | Distillation value |
|---|---|---|---|---|---|
| `test2` | 2026-08-07 | static conveyor, 4+ objects (thermos, laptop, book, smartphone) | 1280×720 top-down | detection quality, size accuracy (±1 cm), prompt tuning (water bottle → thermos) | multi-object variety |
| `test3` | 2026-08-07 | hand-pushed rollers, 3 moving objects (book, glove, pink block) | 1280×720 top-down | tracking persistence, pose stability, hybrid-tracking validation. NOT usable for constant-velocity work (speed varies 40–137 mm/s) | motion blur / moving scenes |
| `test4` | 2026-08-11 | static conveyor, 3 objects (black bag, keyboard, book), hand+black folder occlusions | 640×480, closer mount | occlusion signal measurement (score collapse, depth intrusion up to 554 mm), occlusion-handling development, reappearance latency (median 200 ms) | occlusion-hard examples |
| `test5` | 2026-08-11 | same as test4 + gray notebook (4 objects), 68 s | 640×480 | held-out validation set for occlusion handling (params tuned on test4 only) | occlusion-hard examples, longer sequence |

## Measured facts derived from these bags

- Occlusion: full occlusion = detection loss (max 2.6 s); partial occlusion =
  score below 50 % of the track's own baseline; occluder depth intrusion
  554 mm at 918 mm nominal → 20 % depth gate
- Reappearance-to-pose-resume latency (27 occlusion events, test4+test5,
  well-detected objects): median ~0.3 s, 90 % within 1.7 s, max 3.2 s for
  gradually-clearing occlusions. Lower bound is the 5-frame SAM keyframe
  period (333 ms at 15 fps). Caveat: weak-prompt objects ("gray notebook",
  score ≈ 0.46 avg) show multi-second resume gaps that are a detection
  -threshold issue, not occlusion logic — fix the prompt word first
- EMA position lag at this belt speed (46 mm/s): 2.5–7 mm → Kalman deferred
- 560 px input loses smartphone-sized objects entirely (test2)

## Data still needed (blockers for next work)

| Needed bag | Enables |
|---|---|
| Motorized conveyor, constant speed, objects passing | Kalman coasting + constant-velocity validation |
| Motorized conveyor + occlusion during motion | moving-occlusion handling (Kalman-gated re-matching) |
| Scene with printed ArUco markers | absolute pose error measurement |
