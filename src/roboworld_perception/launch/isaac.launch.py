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
    # color 와 depth 가 서로 다른 틱에서 따로 유실돼 스탬프가 거의 안 겹친다.
    # 실측(2026-08-19, 40 초): color 19 / depth 11 을 받고도 slop=0.05 로는
    # 동기화가 2 번뿐이었다. Isaac 입력 주기(2 초)의 절반이면 짝이 안정적으로
    # 붙으면서도 서로 다른 순간의 프레임을 잘못 묶을 위험은 낮다.
    ("sync_slop", "1.0"),
    # 두 스트림의 발행률이 다르면(color 0.5 Hz / depth 0.3 Hz) 느린 쪽을
    # 기다리는 동안 빠른 쪽 큐가 밀려난다.
    ("sync_queue_size", "30"),
]

def generate_launch_description():
    # base 의 나머지 인자는 다시 적지 않는다. IncludeLaunchDescription 은
    # 설정 스코프를 새로 만들지 않고, DeclareLaunchArgument 는 아직 값이
    # 없을 때만 기본값을 넣는다. 그래서 CLI 로 준 값은 그대로 base 까지
    # 흘러가고, 안 준 것은 base 자신의 기본값이 살아난다.
    # 여기서 굳이 다시 적으면 base 의 기본값을 옛 값으로 못 박게 된다 —
    # max_per_prompt 를 1 로 고정해 버리는 식의 조용한 사고가 난다.
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=v) for n, v in _PRESET]
        + [IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("roboworld_perception"), "launch",
                 "perception.launch.py"])))]
    )
