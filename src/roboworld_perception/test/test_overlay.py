"""라벨 배치 회귀 테스트.

overlay.py 는 눈으로만 확인해 왔고 테스트가 없었다. 배치 로직은 순수 함수라
프레임 없이도 검증할 수 있다 — 실제로 라벨이 겹쳤던 장면(컨베이어 위 블록
9 개)을 좌표로 재현해 두면, 다음에 누가 건드려도 조용히 깨지지 않는다.
"""
from roboworld_perception.overlay import (_FONT_PX, _LINE_H, _STROKE,
                                          _extent, _hits, _label_x,
                                          _label_ys, _place_label)

W, H = 640, 360

# 실제로 잘렸던 라벨 (blue plastic block, 640x360). 폭 241 px 근처.
FULL = ["blue plastic block#7 0.90",
        "d=0.975m xyz=(0.590,-0.002,0.975)",
        "whd=(0.155,0.050,0.000)m",
        "rpy=(180.0,-0.0,-1.0)deg"]
HEAD = [FULL[0]]
TINY = ["#7 0.90"]


def overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


# --- _extent -------------------------------------------------------------

def test_extent_includes_stroke_bleed():
    """폭은 stroke 번짐을 포함해야 한다.

    textlength 로 재면 외곽선이 좌우로 _STROKE 만큼 삐져나온 만큼 덜 당기게
    되어 딱 그 폭만큼 다시 잘린다. left 가 음수(-_STROKE)인 것이 그 증거다.
    """
    left, right = _extent(FULL)
    assert left <= -_STROKE + 1
    assert right > 200


def test_extent_uses_widest_line():
    """한 물체의 줄들은 x 를 공유하므로 가장 넓은 줄 기준이어야 한다."""
    _, wide = _extent(FULL)
    _, narrow = _extent(HEAD)
    assert wide >= narrow


# --- _label_x ------------------------------------------------------------

def test_label_x_pulls_back_from_right_edge():
    """오른쪽 끝 물체의 라벨이 프레임 밖으로 안 나가야 한다."""
    _, right = _extent(FULL)
    x = _label_x(600, FULL, W)
    assert x + right <= W


def test_label_x_keeps_left_edge_visible():
    """프레임보다 넓은 라벨은 왼쪽에 붙여 앞부분이라도 읽히게 한다."""
    narrow_frame = 100
    x = _label_x(80, FULL, narrow_frame)
    left, _ = _extent(FULL)
    assert x + left >= 0


# --- _label_ys -----------------------------------------------------------

def test_label_ys_prefers_above_box():
    """첫 후보는 박스 바로 위여야 한다 — 가까울수록 어느 물체인지 명확하다."""
    box = (100, 200, 200, 240)
    ys = _label_ys(box, len(FULL), H)
    assert ys[0] < box[1]


def test_label_ys_falls_below_when_top_blocked():
    """박스가 상단에 붙으면 위에 자리가 없으므로 아래 후보가 나와야 한다."""
    box = (100, 4, 200, 40)
    ys = _label_ys(box, len(FULL), H)
    assert any(y > box[3] for y in ys)


def test_label_ys_stays_inside_frame():
    box = (100, 200, 200, 240)
    for y in _label_ys(box, len(FULL), H):
        assert y >= 2
        assert y + _LINE_H * len(FULL) <= H


# --- _hits ---------------------------------------------------------------

def test_hits_counts_only_overlapping():
    a = (0, 0, 50, 20)
    far = (200, 200, 250, 220)
    near = (40, 10, 90, 30)
    assert _hits(a, [far]) == 0
    assert _hits(a, [near]) == 1
    assert _hits(a, [far, near]) == 1


def test_hits_has_padding():
    """딱 붙은 라벨도 겹침으로 본다 — 글자끼리 닿으면 읽기 어렵다."""
    a = (0, 0, 50, 20)
    touching = (50, 0, 100, 20)
    assert _hits(a, [touching]) == 1


# --- _place_label --------------------------------------------------------

def test_place_label_avoids_existing():
    """이미 놓인 라벨 자리는 피해야 한다."""
    box = (100, 200, 200, 240)
    _, _, _, rect0 = _place_label(box, [FULL], [], W, H)
    _, _, _, rect1 = _place_label(box, [FULL], [rect0], W, H)
    assert not overlap(rect0, rect1)


def test_place_label_shortens_when_no_room():
    """긴 형식이 안 들어가면 짧은 형식으로 내려간다.

    자리가 없는데 그대로 그리면 글자가 섞여 읽을 수 있는 정보가 0 이 된다.
    줄이면 최소한 track_id 와 점수는 남는다.
    """
    box = (100, 200, 200, 240)
    # 프레임 전체를 이미 점유한 상태를 만든다
    blocked = [(0, 0, W, H)]
    _, _, lines, _ = _place_label(box, [FULL, HEAD, TINY], blocked, W, H)
    assert len(lines) < len(FULL)


def test_place_label_always_returns_something():
    """전부 실패해도 자리를 돌려줘야 한다 — 안 그리는 것보다 낫다."""
    box = (100, 200, 200, 240)
    blocked = [(0, 0, W, H)]
    _, _, lines, rect = _place_label(box, [FULL], blocked, W, H)
    assert lines
    assert rect[2] > rect[0]


def test_nine_blocks_in_a_row_do_not_overlap():
    """회귀: 컨베이어 위 블록 9 개.

    고정 3 단(rank % 3) 시절 0·3·6 번이 같은 단에 걸렸고, 라벨 폭이 241 px 라
    640 px 프레임에 3 개가 못 들어가 글자가 섞였다. 실측된 겹침 예:
        blue plastic block#2 0.95ue plastic block#7 0.90
    """
    boxes = [(20 + i * 68, 170, 20 + i * 68 + 60, 200) for i in range(9)]
    placed = []
    for i, box in enumerate(boxes):
        forms = [FULL, HEAD, [f"#{i} 0.90"]]
        _, _, _, rect = _place_label(box, forms, placed, W, H)
        placed.append(rect)
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            assert not overlap(placed[i], placed[j]), \
                f"라벨 {i} 와 {j} 가 겹침: {placed[i]} vs {placed[j]}"


def test_line_height_follows_font_size():
    """줄 간격이 글자 크기에서 파생되는지.

    예전에는 _LINE_H 가 17 로 박혀 있어 폰트만 키우면 다단 라벨이 저 혼자
    겹쳤다. 테스트가 _LINE_H 를 import 해 쓰므로 값이 틀려도 함께 틀려서
    아무도 못 잡는다 — 관계 자체를 고정한다.
    """
    assert _LINE_H > _FONT_PX
    assert _LINE_H - _FONT_PX <= 4


def test_placement_is_deterministic():
    """같은 입력이면 같은 자리 — 프레임마다 라벨이 튀면 읽을 수 없다."""
    boxes = [(20 + i * 68, 170, 20 + i * 68 + 60, 200) for i in range(5)]

    def run():
        placed = []
        for box in boxes:
            _, _, _, rect = _place_label(box, [FULL, HEAD], placed, W, H)
            placed.append(rect)
        return placed

    assert run() == run()
