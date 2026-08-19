from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("prompts", default_value="물통"),
        DeclareLaunchArgument("score_threshold", default_value="0.4"),
        DeclareLaunchArgument("csv_path", default_value=""),
        DeclareLaunchArgument("rviz", default_value="true"),
        # 아래 셋은 노드가 declare_parameter 로 갖고 있는데 여기서 넘기지
        # 않으면 launch 인자로 줘도 조용히 무시되고 노드 기본값이 남는다.
        # max_per_prompt 기본 1 은 프롬프트당 1 개만 남기므로, 같은 물체가
        # 여러 개인 장면에서는 반드시 올려야 한다.
        DeclareLaunchArgument("max_per_prompt", default_value="1"),
        DeclareLaunchArgument("detect_interval", default_value="5"),
        DeclareLaunchArgument("stale_timeout", default_value="5.0"),
        DeclareLaunchArgument("sync_slop", default_value="0.05"),
        DeclareLaunchArgument("sync_queue_size", default_value="5"),
        DeclareLaunchArgument("image_size", default_value="0"),
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
        ),
        Node(
            package="roboworld_perception",
            executable="perception_node",
            output="screen",
            parameters=[{
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
                "image_size": ParameterValue(
                    LaunchConfiguration("image_size"), value_type=int),
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
