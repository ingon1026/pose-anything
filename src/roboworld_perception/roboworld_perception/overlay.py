"""OpenCV debug rendering: masks, projected 3D OBB, local axes, labels."""
import cv2
import numpy as np

PALETTE = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
           (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66)]

# 12 edges of a box as corner-index pairs (corners = binary order of signs)
_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def _project(points_3d, K):
    p = points_3d @ K.T
    return (p[:, :2] / p[:, 2:3]).astype(int)


def draw_objects(bgr, objects, K):
    """objects: list of pipeline.TrackedObject. Draws in place, returns bgr."""
    for obj in objects:
        color = PALETTE[obj.track_id % len(PALETTE)]

        overlay = bgr.copy()
        overlay[obj.mask] = color
        cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0, dst=bgr)
        contours, _ = cv2.findContours(obj.mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, color, 2)

        x1, y1 = int(obj.box[0]), int(obj.box[1])
        lines = [f"{obj.label}#{obj.track_id} {obj.score:.2f}"]
        if obj.obb is not None:
            o = obj.obb
            signs = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1]
                              for i in range(8)]) * 2 - 1
            corners = o.center + (signs * o.extent / 2) @ o.R.T
            if np.all(corners[:, 2] > 0.05):
                px = _project(corners, K)
                for a, b in _EDGES:
                    cv2.line(bgr, tuple(px[a]), tuple(px[b]), color, 2)
                # local axes: X red, Y green, Z blue (OpenCV BGR)
                axis_len = o.extent / 2 + 0.03
                c2d = _project(o.center[None], K)[0]
                for k, axc in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    tip = o.center + o.R[:, k] * axis_len[k]
                    if tip[2] > 0.05:
                        cv2.arrowedLine(bgr, tuple(c2d), tuple(_project(tip[None], K)[0]),
                                        axc, 2, tipLength=0.2)
            r, p, y = o.rpy
            w, d, h = o.extent
            lines += [f"d={o.distance:.3f}m xyz=({o.center[0]:.3f},{o.center[1]:.3f},{o.center[2]:.3f})",
                      f"whd=({w:.3f},{d:.3f},{h:.3f})m",
                      f"rpy=({r:.1f},{p:.1f},{y:.1f})deg"]
        for i, text in enumerate(lines):
            cv2.putText(bgr, text, (x1, max(15, y1 - 8 - 16 * (len(lines) - 1 - i))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, text, (x1, max(15, y1 - 8 - 16 * (len(lines) - 1 - i))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr
