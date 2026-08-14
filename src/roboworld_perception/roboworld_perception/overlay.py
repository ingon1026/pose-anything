"""OpenCV debug rendering: masks, projected 3D OBB, local axes, labels."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# cv2.putText cannot render Korean labels; use PIL with a CJK font.
try:
    _FONT = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 15)
except OSError:  # 폰트 없는 환경(미설치 컨테이너 등)에서는 기본 폰트로 동작
    _FONT = ImageFont.load_default(15)

# 라벨 외곽선 두께. 폭 측정(_label_x)과 실제 렌더(draw.text)가 같은 값을
# 봐야 하므로 상수로 묶는다 — 한쪽만 건드리면 딱 그만큼 다시 잘려 나간다.
_STROKE = 2

# 폭 측정 전용 스크래치 draw. 실제 렌더용 draw 는 draw_objects 후반부에서야
# 만들어지는데 x 보정은 texts 를 모으는 시점에 이미 필요하다. 1x1 이라
# 생성 비용은 무시할 수준이고, 모듈 로드 시 한 번만 만든다.
_MEASURE = ImageDraw.Draw(Image.new("L", (1, 1)))

PALETTE = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
           (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66)]

# 12 edges of a box as corner-index pairs (corners = binary order of signs)
_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def _project(points_3d, K):
    p = points_3d @ K.T
    return (p[:, :2] / p[:, 2:3]).astype(int)


def _obb_corners(obb):
    signs = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1]
                      for i in range(8)]) * 2 - 1
    return obb.center + (signs * obb.extent / 2) @ obb.R.T


def _draw_obb_edges(bgr, obb, K, color):
    corners = _obb_corners(obb)
    if np.all(corners[:, 2] > 0.05):
        px = _project(corners, K)
        for a, b in _EDGES:
            cv2.line(bgr, tuple(px[a]), tuple(px[b]), color, 2)


def _label_y(box, n_lines, frame_h, rank=0):
    """라벨 y 위치 — 박스 위 기본, 상단에 붙으면 박스 아래로 (단일 규칙).

    rank 는 같은 프레임 안에서 몇 번째 물체인지. 컨베이어처럼 같은 높이에
    물체가 늘어선 장면은 박스 y 가 전부 같아 라벨이 한 자리에 겹쳐 쌓인다
    (실측: 블록 7 개 라벨이 통째로 뭉개져 못 읽음). 계단식으로 어긋낸다.
    """
    step = 17 * n_lines + 4
    y = int(box[1]) - 8 - 17 * n_lines - (rank % 3) * step
    if y < 2:
        y = min(int(box[3]) + 6 + (rank % 3) * step,
                frame_h - 17 * n_lines - 2)
    return y


def _label_x(x, lines, frame_w):
    """라벨 x 위치 — 오른쪽으로 넘칠 만큼 왼쪽으로 당긴다.

    라벨은 박스 왼쪽 x1 에서 시작해 그려지므로 물체가 화면 오른쪽에 있으면
    글자가 그대로 잘려 나갔다 (실측 640x360: "blue plastic block#5 0.9" 에서
    끝 자리가 사라지고 "whd=(0.161,0.057,0.0" 처럼 괄호가 안 닫힘).

    폭은 textlength() 가 아니라 textbbox(stroke_width=) 로 잰다. 외곽선이
    글자 좌우로 _STROKE 만큼 번져 나가는데 textlength 는 그걸 포함하지 않아
    딱 그만큼 덜 당기게 된다. bbox 는 앵커 기준 상대 좌표라 left 가 보통
    -_STROKE 로 음수다 — 그래서 왼쪽 여유도 이 값에서 그대로 얻는다.

    한 물체의 줄들은 폭이 제각각이고(가장 긴 건 대개 d=/xyz= 줄, 241px)
    전부 같은 x 를 공유하므로 최댓값으로 맞춰야 한 줄도 안 잘린다.
    """
    if not lines:
        return x
    boxes = [_MEASURE.textbbox((0, 0), s, font=_FONT, stroke_width=_STROKE)
             for s in lines]
    left = min(b[0] for b in boxes)
    right = max(b[2] for b in boxes)
    x = min(x, frame_w - right)
    # 왼쪽 보정을 오른쪽 다음에 두는 게 핵심: 라벨 블록이 화면보다 넓으면
    # (좁은 프레임 + 긴 한글 라벨) 오른쪽은 포기하고 왼쪽 끝에 붙여야
    # 앞부분의 라벨명·track_id 라도 읽힌다.
    return int(max(x, -left))


def draw_status(bgr, text):
    """좌상단 상태 표시줄 (ASCII 전용, 처리 상태 확인용)."""
    cv2.putText(bgr, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def show_window(bgr):
    """디버그 창 표시. 디스플레이 없는 환경이면 False (호출측은 플래그 끄기)."""
    try:
        cv2.imshow("roboworld perception", bgr)
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def draw_objects(bgr, objects, K):
    """objects: list of tracker.Track. Draws in place, returns bgr."""
    texts = []
    # 라벨 계단 순서는 화면 왼→오른쪽 기준이어야 이웃끼리 안 겹친다.
    # track_id 순서로 매기면 벨트 위 인접 물체가 같은 단에 걸릴 수 있다.
    rank_of = {id(o): i for i, o in
               enumerate(sorted(objects, key=lambda o: o.box[0]))}
    for obj in objects:
        rank = rank_of[id(obj)]
        color = PALETTE[obj.track_id % len(PALETTE)]
        x1, y1 = int(obj.box[0]), int(obj.box[1])

        if obj.occluded:
            # 가림 상태: 마지막으로 알던 3D 박스를 회색으로 유지 표시 —
            # "사라짐"이 아니라 "잠시 못 보는 중"임을 보여준다
            if obj.obb is not None:
                _draw_obb_edges(bgr, obj.obb, K, (150, 150, 150))
            else:
                cv2.rectangle(bgr, (x1, y1), (int(obj.box[2]), int(obj.box[3])),
                              (150, 150, 150), 2)
            lines = [f"{obj.label}#{obj.track_id} OCCLUDED"]
            texts.append((_label_x(x1, lines, bgr.shape[1]),
                          _label_y(obj.box, len(lines), bgr.shape[0], rank), lines))
            continue

        # 마스크 픽셀만 블렌드 (전체 프레임 복사 회피 — 물체당 ~3ms 절약)
        bgr[obj.mask] = (0.35 * np.array(color) + 0.65 * bgr[obj.mask]) \
            .astype(np.uint8)
        contours, _ = cv2.findContours(obj.mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, color, 2)

        lines = [f"{obj.label}#{obj.track_id} {obj.score:.2f}"]
        if obj.obb is not None:
            o = obj.obb
            _draw_obb_edges(bgr, o, K, color)
            corners = _obb_corners(o)
            if np.all(corners[:, 2] > 0.05):
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
        texts.append((_label_x(x1, lines, bgr.shape[1]),
                      _label_y(obj.box, len(lines), bgr.shape[0], rank), lines))

    if texts:
        pil = Image.fromarray(bgr[:, :, ::-1])
        draw = ImageDraw.Draw(pil)
        for x, y, lines in texts:
            for i, line in enumerate(lines):
                draw.text((x, y + 17 * i), line, font=_FONT, fill=(255, 255, 255),
                          stroke_width=_STROKE, stroke_fill=(0, 0, 0))
        bgr[:] = np.asarray(pil)[:, :, ::-1]
    return bgr
