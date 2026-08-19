"""Isaac Sim 구성 프리셋.

perception.launch.py 는 bag 재생을 기본으로 잡고 있다. Isaac Sim 이 이미지를
직접 발행하는 구성에서는 매번 인자를 네 개씩 붙여야 하는데, 빠뜨렸을 때
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
    # 브리지를 TCP(FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA)로 바꾸기 전에는
    # color 와 depth 가 서로 다른 틱에서 따로 유실돼 slop=0.05 로 붙는 짝이
    # 11~16% 뿐이었다. 그래서 여기서 1.0 까지 올려 두었었다.
    #
    # 2026-08-19 브리지 수정 후 실측: 3509/3509 (100%) 가 slop=0.05 안에서
    # 붙고 시간차 중앙값이 0.0000 초다. 두 스트림이 같은 틱에서 나오므로
    # 스탬프가 비트 단위로 같다. 더 이상 늘릴 이유가 없어 base 기본값으로
    # 되돌린다 — slop 을 크게 두면 서로 다른 순간의 프레임을 잘못 묶는다.
    #
    # 만약 다시 "입력 없음" 이 나오면 slop 을 올리기 전에 브리지 설정부터
    # 확인할 것: 양쪽(cellomni 실행.bat 과 ~/.bashrc)에 LARGE_DATA 가 있는가.
    ("sync_slop", "0.05"),
    # 발행률이 39 Hz 로 올라가 큐가 빨리 도니 기본값 5 로도 충분하지만,
    # 여유를 조금 둔다. 메모리 비용은 무시할 수준이다.
    ("sync_queue_size", "10"),
    # base 기본값 1 은 벨트 위 블록을 하나만 잡는다. 씬에는 8 개가 올라가 있어
    # 나머지 7 개가 조용히 버려진다 — 에러가 없어서 알아채기 어렵다.
    # 실측(2026-08-19, 96 프레임): 프레임당 고유 트랙 평균 7.7 개 / 최대 8 개.
    # 8 에 딱 맞추면 블록이 하나 더 들어오거나 마스크가 갈릴 때 잘리므로 여유를 둔다.
    ("max_per_prompt", "10"),
    # 계속 저점수인 허수 조각(정상 블록의 1/4 크기)이 발행되는 것을 막는다.
    # 저점수 2 차 매칭이 트랙을 살리는 것 자체는 의도된 설계이므로(57c3198,
    # 가림 커버리지 +13~51%) base 기본값은 0.0(끔) 그대로 두고 여기서만 켠다.
    #
    # 실측(2026-08-19, 43 사이클)으로 두 무리가 깨끗이 갈린다:
    #     정상 블록 7 개  score 0.840~0.945   최대 extent 0.152~0.198
    #     허수 조각 2 개  score 0.416~0.432   최대 extent 0.0447~0.0498
    # 조각의 최대 extent 가 정상 블록의 두 번째 extent 와 같다 - 길이의 1/4
    # 짜리 조각이다. score_threshold(0.4) 는 신규 트랙 생성 문턱일 뿐이라
    # 이들을 못 막는다. 조각이 0.4 를 아슬아슬하게 넘겨 트랙이 되기 때문이다.
    #
    # 0.4 로는 부족하다 - 실측에서 조각이 0.416/0.432 로 그대로 통과했다.
    # 두 무리 사이(0.47 ~ 0.84)의 가운데인 0.6 으로 잡는다.
    #
    # 절대 임계라 경계에 걸린 트랙은 깜빡인다 - 0.93 으로 실증했을 때 0.934
    # 트랙이 86 프레임 중 6 번만 발행됐다. 여유를 크게 둘 것.
    ("publish_score_min", "0.6"),
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
