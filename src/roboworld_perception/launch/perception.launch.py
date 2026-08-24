from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        # 계약서 docs/bridge_contract.md §1.1 — "launch 파일에 직접 박아 넣는
        # 것을 권한다. env 를 잊는 사고가 이미 두 번 났다".
        #
        # scripts/ros_env.sh 와 값이 같고 둘 다 필요하다. 셸 쪽은 run.sh 가
        # ros2 run / rviz2 로 launch 를 우회하는 경로를 덮고, 이쪽은 사람이
        # 직접 `ros2 launch ...` 를 치는 경로를 덮는다 — 후자는 지금까지
        # ~/.bashrc 로만 덮여 있었고, 그건 대화형 셸에서만 읽힌다.
        # 값 사본 위치는 scripts/ros_env.sh 주석 참고.
        SetEnvironmentVariable(
            "FASTDDS_BUILTIN_TRANSPORTS",
            "LARGE_DATA?max_msg_size=190KB&sockets_size=200KB"
            "&non_blocking=true&tcp_negotiation_timeout=50"),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        DeclareLaunchArgument("prompts", default_value="물통"),
        DeclareLaunchArgument("score_threshold", default_value="0.4"),
        DeclareLaunchArgument("csv_path", default_value=""),
        DeclareLaunchArgument("rviz", default_value="true"),
        # 모든 시간 소비자(perception/RViz)가 같은 clock을 써야 한다.
        # 일반 카메라/bag은 wall time, Isaac 프리셋은 이 값을 true로 덮는다.
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        # 아래 셋은 노드가 declare_parameter 로 갖고 있는데 여기서 넘기지
        # 않으면 launch 인자로 줘도 조용히 무시되고 노드 기본값이 남는다.
        # max_per_prompt 기본 1 은 프롬프트당 1 개만 남기므로, 같은 물체가
        # 여러 개인 장면에서는 반드시 올려야 한다.
        DeclareLaunchArgument("max_per_prompt", default_value="1"),
        DeclareLaunchArgument("detect_interval", default_value="5"),
        DeclareLaunchArgument("stale_timeout", default_value="5.0"),
        DeclareLaunchArgument("sync_slop", default_value="0.05"),
        DeclareLaunchArgument("sync_queue_size", default_value="1"),
        DeclareLaunchArgument("publish_score_min", default_value="0.0"),
        DeclareLaunchArgument("enable_merge", default_value="true"),
        # 아래 둘은 2026-08-21 도입, 실측 근거는 docs/belt_plane_2026-08-21.md
        # 와 docs/footprint_gate_2026-08-21.md. 둘 다 기본 켜짐이고, 끄면
        # 그날 이전 동작으로 돌아간다.
        DeclareLaunchArgument("use_belt_plane", default_value="true"),
        DeclareLaunchArgument("enable_footprint_gate", default_value="true"),
        DeclareLaunchArgument("image_size", default_value="0"),
        # 공칭 1m camera-above-belt TF는 로봇 좌표가 아니다. 실제 셀의
        # world/base 변환은 캘리브레이션된 외부 TF가 책임진다.
        DeclareLaunchArgument("publish_world_tf", default_value="false"),
        # Isaac Sim 이 이미지를 직접 발행하면 RealSense 드라이버가 없어
        # camera_link -> camera_color_optical_frame 이 비므로 여기서 켠다.
        # bag 재생에는 그 링크가 이미 녹화돼 있어 켜면 TF 가 깨진다 — 기본 false.
        DeclareLaunchArgument("publish_optical_tf", default_value="false"),
        # 프레임 이름. 정적 TF 는 camera_info 보다 먼저 발행되므로 실제
        # optical frame 이름을 기다릴 수 없다 — 다르면 여기서 맞춘다.
        DeclareLaunchArgument("world_frame", default_value="world"),
        DeclareLaunchArgument("camera_link_frame",
                              default_value="camera_link"),
        DeclareLaunchArgument("optical_frame",
                              default_value="camera_color_optical_frame"),
        Node(
            package="rviz2",
            executable="rviz2",
            condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=["-d", PathJoinSubstitution(
                [FindPackageShare("roboworld_perception"), "rviz",
                 "perception.rviz"])],
            output="log",
            parameters=[{
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool),
            }],
        ),
        Node(
            package="roboworld_perception",
            executable="perception_node",
            output="screen",
            parameters=[{
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool),
                "prompts": LaunchConfiguration("prompts"),
                "score_threshold": LaunchConfiguration("score_threshold"),
                "csv_path": LaunchConfiguration("csv_path"),
                "max_per_prompt": ParameterValue(
                    LaunchConfiguration("max_per_prompt"), value_type=int),
                "detect_interval": ParameterValue(
                    LaunchConfiguration("detect_interval"), value_type=int),
                "stale_timeout": ParameterValue(
                    LaunchConfiguration("stale_timeout"), value_type=float),
                "sync_slop": ParameterValue(
                    LaunchConfiguration("sync_slop"), value_type=float),
                "sync_queue_size": ParameterValue(
                    LaunchConfiguration("sync_queue_size"), value_type=int),
                "publish_score_min": ParameterValue(
                    LaunchConfiguration("publish_score_min"), value_type=float),
                "enable_merge": ParameterValue(
                    LaunchConfiguration("enable_merge"), value_type=bool),
                "use_belt_plane": ParameterValue(
                    LaunchConfiguration("use_belt_plane"), value_type=bool),
                "enable_footprint_gate": ParameterValue(
                    LaunchConfiguration("enable_footprint_gate"),
                    value_type=bool),
                "image_size": ParameterValue(
                    LaunchConfiguration("image_size"), value_type=int),
                "publish_world_tf": ParameterValue(
                    LaunchConfiguration("publish_world_tf"), value_type=bool),
                "publish_optical_tf": ParameterValue(
                    LaunchConfiguration("publish_optical_tf"),
                    value_type=bool),
                "world_frame": LaunchConfiguration("world_frame"),
                "camera_link_frame": LaunchConfiguration(
                    "camera_link_frame"),
                "optical_frame": LaunchConfiguration("optical_frame"),
            }],
        ),
    ])
