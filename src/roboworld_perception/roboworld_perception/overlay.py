"""OpenCV debug rendering: masks, projected 3D OBB, local axes, labels."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# cv2.putText cannot render Korean labels; use PIL with a CJK font.
_FONT_PX = 15
# 줄 간격 = 글자 크기 + leading. _FONT_PX 만 바꿔도 다단 라벨이 겹치지 않게
# 함께 움직여야 한다 — 예전에는 17 이 폰트 크기와 무관한 상수로 떠 있었다.
_LINE_H = _FONT_PX + 2

try:
    _FONT = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", _FONT_PX)
except OSError:  # 폰트 없는 환경(미설치 컨테이너 등)에서는 기본 폰트로 동작
    _FONT = ImageFont.load_default(_FONT_PX)

# 라벨 외곽선 두께. 폭 측정(_extent)과 실제 렌더(draw.text)가 같은 값을
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


# 라벨 y 후보를 방향별로 이만큼까지만 만든다. 프레임 경계로도 종료되므로
# 종료 조건이 아니라 탐색량 상한이다.
_MAX_TIERS = 12


def _extent(lines):
    """라벨 블록의 (left, right). 실제 렌더(draw.text)와 같은 stroke 기준으로 잰다."""
    boxes = [_MEASURE.textbbox((0, 0), s, font=_FONT, stroke_width=_STROKE)
             for s in lines]
    return min(b[0] for b in boxes), max(b[2] for b in boxes)


def _clamp_x(x, left, right, frame_w):
    """측정된 (left, right) 로 라벨 x 를 프레임 안에 밀어 넣는다.

    왼쪽 보정을 오른쪽 다음에 두는 게 핵심: 라벨 블록이 화면보다 넓으면
    (좁은 프레임 + 긴 한글 라벨) 오른쪽은 포기하고 왼쪽 끝에 붙여야
    앞부분의 라벨명·track_id 라도 읽힌다.
    """
    return int(max(min(x, frame_w - right), -left))


def _label_ys(box, n_lines, frame_h):
    """라벨 y 후보 — 박스 위쪽을 가까운 순으로, 다 막히면 아래쪽.

    같은 높이에 물체가 늘어선 컨베이어 장면은 박스 y 가 전부 같아 라벨이
    한 자리에 쌓인다. 예전에는 rank % 3 으로 3 단만 어긋냈는데, 블록 9 개가
    한 줄로 서면 0·3·6 번이 같은 단에 걸린다. 라벨 폭이 241 px 라 640 px
    프레임에 3 개가 못 들어가 글자끼리 섞였다 (실측: "block#2 0.95" 와
    "ue plastic block#7" 이 한 줄에 겹쳐 그려짐).

    그래서 고정 단수를 버린다. 여기서는 후보만 넉넉히 내고, 실제로 비어
    있는 자리를 고르는 일은 _place_label 이 한다.
    """
    h = _LINE_H * n_lines
    step = h + 4
    above, below = [], []      # 위/아래를 따로 센다 — 누적으로 세면 위가 꽉 찬
    top = int(box[1]) - 8 - h  # 박스일수록 아래 후보가 줄어드는 비대칭이 생긴다
    while top >= 2 and len(above) < _MAX_TIERS:
        above.append(top)
        top -= step
    bot = int(box[3]) + 6
    while bot + h <= frame_h - 2 and len(below) < _MAX_TIERS:
        below.append(bot)
        bot += step
    ys = above + below
    if not ys:                       # 프레임보다 라벨이 큰 극단
        ys.append(max(2, frame_h - h - 2))
    return ys


def _hits(rect, placed):
    """placed 중 rect 와 겹치는 개수. 2 px 여유를 둬 글자끼리 붙지 않게 한다."""
    x0, y0, x1, y1 = rect[0] - 2, rect[1] - 2, rect[2] + 2, rect[3] + 2
    return sum(1 for p in placed
               if not (x1 <= p[0] or p[2] <= x0 or y1 <= p[1] or p[3] <= y0))


def _place_label(box, forms, placed, frame_w, frame_h):
    """빈 자리를 찾아 (x, y, lines, rect) 를 돌려준다.

    forms 는 긴 형식부터 짧은 형식 순으로 준다. 긴 형식이 어디에도 안
    들어가면 짧은 형식으로 내려간다 — 자리가 없는데 그대로 그려서 글자가
    섞이는 것보다, 정보를 줄여서라도 읽히게 하는 편이 낫다.
    전부 실패하면 겹침이 가장 적은 자리를 쓴다 (아무것도 안 그리는 것보다 낫다).
    """
    best = None
    for lines in forms:
        left, right = _extent(lines)
        # x 는 y 에 의존하지 않는다 — 이 탐색은 세로 방향으로만 자리를 옮긴다.
        # 루프 안에서 다시 구하면 후보 수만큼 폰트 측정이 되풀이된다
        # (textbbox 1 회 ≈ 171 us, 12 개 장면에서 프레임당 30~50 ms).
        x = _clamp_x(int(box[0]), left, right, frame_w)
        for y in _label_ys(box, len(lines), frame_h):
            rect = (x + left, y, x + right, y + _LINE_H * len(lines))
            n = _hits(rect, placed)
            if n == 0:
                return x, y, lines, rect
            # 동점이면 짧은 형식을 고른다. 어차피 겹칠 바에는 덮는 면적이
            # 작은 쪽이 아래 라벨을 덜 가린다 — 먼저 시도한 긴 형식이
            # 이기면 자리가 아무 데도 없을 때 화면이 가장 지저분해진다.
            key = (n, len(lines))
            if best is None or key < best[0]:
                best = (key, (x, y, lines, rect))
    return best[1]


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
    placed = []
    frame_h, frame_w = bgr.shape[0], bgr.shape[1]
    # 왼→오른쪽 순으로 자리를 잡아야 이웃끼리 예측 가능하게 나눠 갖는다.
    # track_id 순으로 하면 벨트 위 인접 물체가 뒤죽박죽 자리를 다툰다.
    for obj in sorted(objects, key=lambda o: o.box[0]):
        color = PALETTE[obj.track_id % len(PALETTE)]

        if obj.occluded:
            # 가림 상태: 마지막으로 알던 3D 박스를 회색으로 유지 표시 —
            # "사라짐"이 아니라 "잠시 못 보는 중"임을 보여준다
            if obj.obb is not None:
                _draw_obb_edges(bgr, obj.obb, K, (150, 150, 150))
            else:
                cv2.rectangle(bgr,
                              (int(obj.box[0]), int(obj.box[1])),
                              (int(obj.box[2]), int(obj.box[3])),
                              (150, 150, 150), 2)
            forms = [[f"{obj.label}#{obj.track_id} OCCLUDED"],
                     [f"#{obj.track_id} OCC"]]
        else:
            # 마스크 픽셀만 블렌드 (전체 프레임 복사 회피 — 물체당 ~3ms 절약)
            bgr[obj.mask] = (0.35 * np.array(color) + 0.65 * bgr[obj.mask]) \
                .astype(np.uint8)
            contours, _ = cv2.findContours(obj.mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bgr, contours, -1, color, 2)

            head = f"{obj.label}#{obj.track_id} {obj.score:.2f}"
            lines = [head]
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
            # 전체 → 머리줄만 → track_id 와 점수만. 좁을수록 아래로 내려간다.
            forms = [lines, [head], [f"#{obj.track_id} {obj.score:.2f}"]]

        x, y, lines, rect = _place_label(obj.box, forms, placed,
                                         frame_w, frame_h)
        placed.append(rect)
        texts.append((x, y, lines))

    if texts:
        pil = Image.fromarray(bgr[:, :, ::-1])
        draw = ImageDraw.Draw(pil)
        for x, y, lines in texts:
            for i, line in enumerate(lines):
                draw.text((x, y + _LINE_H * i), line, font=_FONT, fill=(255, 255, 255),
                          stroke_width=_STROKE, stroke_fill=(0, 0, 0))
        bgr[:] = np.asarray(pil)[:, :, ::-1]
    return bgr
