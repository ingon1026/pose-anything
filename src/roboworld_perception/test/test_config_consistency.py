"""설정 파일들이 서로 갈라지지 않았는지 — 오늘(2026-08-28) 두 번 진 자리다.

둘 다 "노드는 멀쩡하고 에러도 로그도 없는데 원하는 일이 안 일어나는" 형태였다:

1. `publish_points` 를 노드에만 선언하고 launch 에 안 실어서 `ros2 launch ...
   publish_points:=true` 가 **조용히 무시**됐다(`110e9b0`). launch 파일
   주석이 이미 그 함정을 경고하고 있었는데도 났다 — 경고로는 안 막힌다.
2. `rviz/perception.rviz` 의 Fixed Frame 이 노드가 못박은 `world` 와 갈라져
   광학 프레임으로 남아 있었다(`71437fa`). RViz 가 그리는 모든 것의 높이가
   위아래로 뒤집혔고, 마커만 있을 때는 상자가 상자로 보여 안 드러났다.

파일을 AST/YAML 로 읽기만 한다 — rclpy 도 torch 도 안 붙어 CI 에서 돈다.
"""
import ast

import yaml

from conftest import PKG_DIR


def _first_str_args(path, func_name):
    """`func_name("이름", ...)` 호출들의 첫 문자열 인자 집합."""
    out = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if (name == func_name and n.args
                and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            out.add(n.args[0].value)
    return out


def _dict_str_keys(path):
    out = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Dict):
            out |= {k.value for k in n.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return out


# launch 로 일부러 안 내보내는 것. **줄이는 것이 목표가 아니라, 새로 늘 때
# 사람이 한 번 판단하게 만드는 것이 목표다.** 여기 안 적고 노드에만 선언하면
# 아래 테스트가 깨진다 — 오늘 publish_points 가 그렇게 조용히 빠졌다.
_LAUNCH_EXEMPT = {
    "display",       # run.sh 가 `ros2 run -p` 로 직접 준다(HEADLESS 분기)
    "belt_plane",    # 캘리브 주입용. 문자열 규약이라 launch 인자로 쓰기 나쁘다
    "color_topic",   # 입력 토픽 3종은 Isaac/bag/실기가 갈리는 지점인데
    "depth_topic",   # 아직 launch 로 못 준다 — 내보내려면 프리셋도 같이 봐야 한다
    "info_topic",
    "input_qos_depth",  # 노드가 검증·경고까지 갖고 있는데 launch 로는 못 준다
}


def test_every_node_parameter_is_reachable_from_launch():
    """노드 파라미터가 launch 양쪽(인자·parameters dict)에 다 실려야 한다.

    한쪽만 있으면 `ros2 launch ... foo:=x` 가 에러 없이 무시된다.
    """
    node = _first_str_args(PKG_DIR / "roboworld_perception/perception_node.py",
                           "declare_parameter")
    lp = PKG_DIR / "launch/perception.launch.py"
    args = _first_str_args(lp, "DeclareLaunchArgument")
    keys = _dict_str_keys(lp)
    want = node - _LAUNCH_EXEMPT
    assert not (want - args), f"launch 인자 누락: {sorted(want - args)}"
    assert not (want - keys), f"parameters dict 누락: {sorted(want - keys)}"
    assert not (_LAUNCH_EXEMPT - node), \
        f"제외 목록에 없는 파라미터가 남았다: {sorted(_LAUNCH_EXEMPT - node)}"


def _rviz():
    with open(PKG_DIR / "rviz/perception.rviz", encoding="utf-8") as f:
        return yaml.safe_load(f)["Visualization Manager"]


def test_rviz_fixed_frame_matches_the_node_default():
    """Fixed Frame 은 노드의 world_frame 기본값과 같아야 한다.

    갈라지면 RViz 가 조용히 다른 좌표계로 그린다 — 광학 프레임으로 갈라져
    있던 동안 높이가 통째로 뒤집혀 있었다(71437fa).
    """
    src = (PKG_DIR / "roboworld_perception/perception_node.py").read_text(
        encoding="utf-8")
    default = None
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "declare_parameter"
                and n.args and getattr(n.args[0], "value", None) == "world_frame"):
            default = n.args[1].value
    assert default == "world"
    assert _rviz()["Global Options"]["Fixed Frame"] == default


def test_rviz_topics_are_topics_the_node_publishes():
    """디스플레이가 가리키는 토픽이 노드가 만드는 이름과 같아야 한다."""
    src = (PKG_DIR / "roboworld_perception/perception_node.py").read_text(
        encoding="utf-8")
    published = {n.value for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and n.value.startswith("/perception/")}
    for d in _rviz()["Displays"]:
        topic = (d.get("Topic") or {}).get("Value") if isinstance(
            d.get("Topic"), dict) else d.get("Topic")
        if isinstance(topic, str) and topic.startswith("/perception/"):
            assert topic in published, f"{d['Name']} 가 없는 토픽을 본다: {topic}"
