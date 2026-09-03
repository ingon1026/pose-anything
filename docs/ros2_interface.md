# ROS 2 interface

> Moved verbatim from the top-level README on 2026-09-02, when the README was cut down to a front page. Nothing here was rewritten.


![ROS 2 node graph](images/rqt_graph.png)

**Subscribes** — `/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`,
`.../camera_info`, `/perception/prompt`

**Publishes**

| Topic | Type | Content |
|---|---|---|
| `/perception/detections` | `vision_msgs/Detection3DArray` | label, score, track ID, geometric OBB pose, size, covariance — in the camera optical frame reported by `camera_info` |
| `/perception/markers` | `visualization_msgs/MarkerArray` | OBB cube, XYZ axes, label text for RViz |
| `/perception/debug_image` | `sensor_msgs/Image` | mask + 3D box + status overlay |
| `/perception/odom` | `nav_msgs/Odometry` | one message per published object, `child_frame_id = obj_<id>`; pose as in detections, **twist in the object frame** (nav_msgs convention), angular velocity not estimated (covariance 1e3) — **on by default**, see below |
| `/perception/tracks` | `diagnostic_msgs/DiagnosticArray` | one status per live track: `visible` / `held` / `pending` / `occluded` / `lost` plus a `reason` (`border`, `footprint`, `chi2`, …) — tells a consumer whether an object missing from `detections` is occluded or gone — **on by default**, see below |
| `/perception/points` | `sensor_msgs/PointCloud2` | per-object points of this frame's raw observation, coloured by track ID — **off by default**, see below |
| `/perception/status` | `diagnostic_msgs/DiagnosticArray` | 입력 건강 하트비트, **1 Hz**. `status[0].name = roboworld_perception/input`, `hardware_id = rgbd_camera` |

Each published object also gets a TF frame `obj_<id>` (parent: the camera
optical frame), refreshed every frame.

**Parameters** — `prompts`, `detect_interval` (5, SAM keyframe period),
`max_per_prompt` (1, tracks per prompt), `csv_path`, `display`,
`score_threshold` (0.4 — a new-track gate, *not* a publish gate; see below),
`publish_score_min` (0.0 = off — the actual publish gate; see below),
`publish_points` (false — see below),
`publish_odom` (true — turns off `/perception/odom`; see Velocity and
per-object TF),
`publish_tracks` (true — turns off `/perception/tracks`; see Track state —
occluded vs. gone),
`publish_object_tf` (true — turns off the per-object `obj_<id>` TF frames;
see Velocity and per-object TF),
`publish_world_tf` (defaults to the `rviz` launch argument, so runs that open
RViz enable it; never use this nominal, uncalibrated RViz-only TF as robot
coordinates — the node warns when it publishes one — supply a calibrated
camera-to-world/base TF externally instead),
`use_sim_time` (false for normal camera/bag launches; `isaac.launch.py` sets it
true for both perception and RViz)

`score_threshold` (0.4) is **not** a publish threshold. The detector itself
returns everything above `min(0.1, score_threshold)`; the value only decides
which detections are strong enough to *start a new track* (and which go to the
ByteTrack low-score second pass). A track that is kept alive by low-score
observations keeps publishing, and its `score` field follows those low
observations down — that is deliberate, since low-score matches are what carry
a track through partial occlusion.

`publish_score_min` (0.0 = off) is the actual publish gate, for scenes where
persistent low-score fragments would otherwise reach a consumer. It is an
absolute threshold, so a track sitting near the value flickers in and out —
set it well below the scores you expect. `launch/isaac.launch.py` carries the
measurement that decides its value — read that comment before changing it.

`/perception/points` is an inspection tool, not a steady-state output. Turn it
on with `ros2 launch roboworld_perception perception.launch.py
publish_points:=true` — `run.sh` does not expose the flag. Its publish gate is
literally the same as `/perception/detections` — same loop, same `publishable`
decision — so the two topics always talk about the same objects, and a frame
with nothing to publish is sent as `width=0` rather than not sent. It costs
about **8 bytes per mask pixel**, so the frame size follows object size, not
object count: measured 4.6 KB per Isaac block (~592 mask px) and 55.8 KB for a
book/keyboard-sized mask (7,310 px) — a 12x spread, roughly 15-170 KB per frame
for three objects. Measure it on your own scene rather than quoting one number.
There is also per-track packing work in the publish path. The RViz
preset carries a PointCloud2 display for it, **disabled by default**: it is
meant to *replace* the MarkerArray, not to overlay it — boxes and points drawn
on top of each other are both unreadable — so tick it and untick MarkerArray.

> **⚠ The cloud and the box are the same object, but not the same geometry.**
> A box's center and extent come from the track's **filter state**; the cloud is
> **this frame's raw observation**, which the χ² and footprint gates may have
> rejected. Seeing the two disagree is the point of this topic — do not use it to
> validate the boxes. Full contract: `docs/bridge_contract.md` §6.3.

What a consumer actually receives (captured from a real run — the per-axis
position variance is filled in by the fusion filter. The OBB quaternion is for
geometric visualisation, not a calibrated semantic orientation; its roll,
pitch, yaw covariance diagonals are conservatively set to \(\pi^2\) rad²
(180° 1σ), so an orientation-gated grasp must reject it unless a separate
orientation estimator is added):

```yaml
# ros2 topic echo /perception/detections --once   (truncated)
detections:
- results:
  - hypothesis: {class_id: black bag, score: 0.914}
    pose:
      pose:
        position: {x: -0.488, y: 0.053, z: 0.894}
        orientation: {x: -0.112, y: -0.103, z: -0.641, w: 0.753}
      covariance: [7.7e-05, 0, 0, ..., 2.7e-04, ..., 2.8e-05, ..., 9.87, ..., 9.87, ..., 9.87]  # σxyz=8.8/16.5/5.3mm; σrpy=180° (unestimated)
  bbox: {size: {x: 0.462, y: 0.529, z: 0.180}}
  id: black bag#2
```

### Velocity and per-object TF

`/perception/odom` and the `obj_<id>` TF frames exist for lead calculations:
on a moving belt, an object's detected position is already stale by the time
a gripper reaches it, and the fused per-track velocity lets a consumer
predict where the object will be after a lead time `Δt`. Following
`nav_msgs/Odometry` convention, `twist.linear` is reported in the object
frame (`child_frame_id = obj_<id>`), not the camera optical frame — to get it
in optical-frame coordinates, rotate by the same orientation `R` carried in
the accompanying `pose.pose.orientation`: `v_optical = R · v_body`.
`twist.angular` is always zero with covariance diagonal `1e3`, meaning
angular velocity is not estimated. Both outputs are on by default
(`publish_odom`, `publish_object_tf`); disable either with
`publish_odom:=false` / `publish_object_tf:=false` on `perception.launch.py`.
Neither has an empty form — on withdrawal, no `Odometry` message and no TF
update is sent for that object — so a consumer must judge validity from
`header.stamp` age rather than waiting for an explicit withdrawal;
`/perception/detections` remains the source of truth for which objects are
currently valid, since it degrades to an empty array on occlusion or
staleness instead of going silent per object. Check both directly:

```bash
ros2 topic echo /perception/odom --once
ros2 run tf2_ros tf2_echo camera_color_optical_frame obj_1
```

For example, to reach an object with a gripper `Δt` seconds from now,
predict its optical-frame position as `p + R · v_body · Δt`, using the same
`R` as above.

### Track state — occluded vs. gone

`/perception/detections` never carries a stale pose — once a track's
current-frame observation is rejected, the object simply drops out of the
array, and that topic alone cannot say whether it is briefly occluded or
actually gone. `/perception/tracks` (`diagnostic_msgs/DiagnosticArray`)
exists to answer that: every frame it reports one `DiagnosticStatus` per
**live** track, whether or not that track is currently in `/perception/detections`,
plus a final `lost` status the frame a track is deleted. `name` is
`<label>#<id>` and `hardware_id` is `obj_<id>`, matching the IDs used
elsewhere; `message` carries the state string, and `level` mirrors it
(`OK`=visible, `WARN`=held/pending, `ERROR`=occluded, `STALE`=lost).

| State | Meaning | In `/perception/detections`? |
|---|---|---|
| `visible` | this frame's observation was accepted | yes, current pose |
| `held` | observation rejected, but within `T_STALE` (0.5 s) | yes, last accepted pose |
| `pending` | seen and fresh, but not yet publishable — not confirmed (needs 3 accepted observations), position uncertainty too high, or below the score gate | no — likely to appear soon |
| `occluded` | beyond `T_STALE`, or the track is frozen | no — this is what looks like "disappeared" |
| `lost` | track deleted this frame (final message) | no |

`values` carries `state`, `reason`, `since_accept_s`, `missed`, `confirmed`,
`published`. `reason` explains why *this frame's* observation was not used:
`none`; `border` (mask touches the image edge — object partly out of frame);
`footprint` (footprint gate — partial occlusion); `chi2` (observation
rejected by the filter — an occluder intruding, or a look-alike);
`thickness`; `no_obb`; `convention`; `unconfirmed`; `pos_std`; `score`. For
`pending`, when no per-frame rejection applies, `reason` names the gate
instead. As a rule of thumb: `occluded` with a small `since_accept_s` means
wait, `pending` means the object is being seen but the filter has not yet
promoted it — wait a few frames, `lost` means move on, and `held` means the
pose already in `detections` is fresh enough to use but was not re-confirmed
this frame. By construction, an object is in `/perception/detections`
exactly when its state is `visible` or `held`. On input withdrawal,
`/perception/tracks` publishes an empty `DiagnosticArray` — with no input,
track state is unknown. Check it directly:

```bash
ros2 topic echo /perception/tracks --once
```

