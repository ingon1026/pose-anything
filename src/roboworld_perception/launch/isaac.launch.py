"""Isaac Sim 구성 프리셋.

perception.launch.py 는 bag 재생을 기본으로 잡고 있다. Isaac Sim 이 이미지를
직접 발행하는 구성에서는 매번 인자를 여러 개 붙여야 하는데, 빠뜨렸을 때
증상이 조용해서 원인 찾기에 시간이 든다 (마커만 안 보이거나, 며칠 뒤 CUDA
에러로 터지거나). 그래서 실기동으로 검증된 조합을 여기 묶는다.

    ros2 launch roboworld_perception isaac.launch.py

인자는 전부 그대로 덮어쓸 수 있다. 각 값을 왜 이렇게 잡았는지는
docs/isaac_sim_stability_2026-08-14.md 에 실측과 함께 적혀 있다.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

# (인자, Isaac 기본값, 왜 base 와 다른가)
_PRESET = [
    # RealSense 드라이버가 없어 camera_link -> camera_color_optical_frame 을
    # 아무도 안 채운다. 끄면 RViz 에 마커가 통째로 안 보인다.
    ("publish_optical_tf", "true"),
    # VRAM 고갈로 렌더프로덕트 텍스처가 재할당되면 ROS2 발행 노드의 CUDA
    # 핸들이 무효가 된다. SAM3 입력을 1008 -> 672 로 낮춰 여유를 만든다.
    # 카메라가 640x360 이라 1008 은 어차피 업스케일이었다. 실측 23,410 -> 0 회.
    ("image_size", "672"),
    # Isaac 실시간 입력은 SAM3 를 함께 돌리면 0.5 Hz 까지 떨어진다. 기본 5 면
    # SAM3 가 10 초에 한 번 도는 셈이라 추적기가 설 틈이 없다.
    # 실측: detect_interval=8 -> 평균 0.1 개, =1 -> 평균 4.7 개.
    ("detect_interval", "1"),
    # detect_interval=1 에 RViz 까지 얹으면 12 GB GPU 에서 넘친다.
    # 실측: CUDA 에러 37,151 회 뒤 [omni.rtx] GPU crash, Isaac SIGSEGV.
    # 영상만 볼 거면 grab_debug.py 쪽이 안전하다.
    ("rviz", "false"),
    # base 기본값 1 은 벨트 위 블록을 하나만 잡는다. 씬에는 8 개가 올라가 있어
    # 나머지 7 개가 조용히 버려진다 — 에러가 없어서 알아채기 어렵다.
    # 실측(2026-08-19, 96 프레임): 프레임당 고유 트랙 평균 7.7 개 / 최대 8 개.
    # 8 에 딱 맞추면 블록이 하나 더 들어오거나 마스크가 갈릴 때 잘리므로 여유를 둔다.
    ("max_per_prompt", "10"),
    # 허수 조각 발행을 막는다. 왜 이 게이트가 존재하고 무엇을 주의해야 하는지는
    # Track.publishable 의 주석에 있다. 여기엔 이 씬의 실측만 적는다.
    #
    # 실측(2026-08-19, 43 사이클)으로 두 무리가 깨끗이 갈린다:
    #     정상 블록 7 개  score 0.840~0.945   최대 extent 0.152~0.198
    #     허수 조각 2 개  score 0.416~0.432   최대 extent 0.0447~0.0498
    # 조각의 최대 extent 가 정상 블록의 두 번째 extent 와 같다 - 길이의 1/4
    # 짜리 조각이다. score_threshold(0.4) 는 신규 트랙 생성 문턱일 뿐이라
    # 이들을 못 막는다. 조각이 0.4 를 아슬아슬하게 넘겨 트랙이 되기 때문이다.
    #
    # 0.4 로는 부족하다 - 실측에서 조각이 0.416/0.432 로 그대로 통과했다.
    # 두 무리 사이(0.43 ~ 0.84)에서 조각 쪽에 여유를 두고 0.6 으로 잡는다.
    # 경계에 걸린 트랙은 깜빡이므로 여유를 크게 둔다(0.93 실증: 86 중 6 프레임).
    ("publish_score_min", "0.6"),
    # 한 물체에 트랙이 둘 붙는 중복을 사후 삭제한다. 생성 시점 가드는
    # "태어날 때 남의 박스 안"만 막아서, 떨어져 태어나 나중에 같은 자리로
    # 수렴한 중복은 통과한다 — 실측(isaac_belt_moving): EndStop_03 에
    # 트랙 2개가 1703프레임 공존하며 생존 트랙의 z-std 를 0.72 -> 3.58mm 로
    # 5배 악화시켰다. 발행만 막아서는 안 되는 이유가 이것이다 — 중복 트랙이
    # 매칭 층에서 계속 검출을 뺏어간다.
    #
    # base 기본값이 false 인 이유: 판정 상수 KAPPA_PHYS 가 벨트 씬에서만
    # 보정됐다. 실기 bag(test2~5)은 라벨이 전부 유일하고 max_per_prompt=1
    # 이라 중복이 구조적으로 생기지 않아 음성 표본만 기여한다.
    ("enable_merge", "true"),
    # 프롬프트 A/B 실측(2026-08-19, 12 종 x 20 초, 재현성 ±0.2 mm)으로 고른 값.
    # 기하는 어휘에 전혀 반응하지 않는다 — 검출된 9 종 모두 우측 3 개의 길이가
    # 152.1/154.9/154.6 mm 로 0.4 mm 이내 동일했다(그 원인은 geometry 의 depth
    # 클립이고 별도로 고쳤다). 따라서 프롬프트로 고를 수 있는 것은 두 가지뿐이다:
    #   조각 트랙의 유무, 그리고 publish_score_min 대비 score 여유.
    #   "blue plastic block"  조각 1~2 개(프레임마다 들락) / score 0.937·0.888
    #   "blue bar with holes" 조각 0 개(60 초 291 프레임 전부) / score 0.929·0.880
    # 조각을 공짜로 없애면서 score 손실이 0.01 이하라 이쪽을 기본값으로 올린다.
    # 탈락: "blue plastic beam" 은 조각 0 이지만 최저 score 0.60 으로 게이트에
    # 닿는다. "blue plastic blocks"(복수형)는 score 가 게이트 아래로 내려가
    # 발행이 통째로 멈춘다 — 복수형은 쓰지 말 것.
    ("prompts", "blue bar with holes"),
]

def generate_launch_description():
    # base 의 나머지 인자는 다시 적지 않는다. IncludeLaunchDescription 은
    # 설정 스코프를 새로 만들지 않고, DeclareLaunchArgument 는 아직 값이
    # 없을 때만 기본값을 넣는다. 그래서 CLI 로 준 값은 그대로 base 까지
    # 흘러가고, 안 준 것은 base 자신의 기본값이 살아난다.
    # 여기서 굳이 다시 적으면 base 의 기본값을 옛 값으로 못 박게 된다 —
    # score_threshold 를 옛 값으로 고정해 버리는 식의 조용한 사고가 난다.
    # (프리셋에 올리는 것은 실측 근거가 있는 값만. 위 _PRESET 참고.)
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=v) for n, v in _PRESET]
        + [IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("roboworld_perception"), "launch",
                 "perception.launch.py"])))]
    )
