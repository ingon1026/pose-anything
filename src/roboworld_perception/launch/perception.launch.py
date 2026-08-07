from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("prompts", default_value="물통"),
        DeclareLaunchArgument("score_threshold", default_value="0.4"),
        DeclareLaunchArgument("csv_path", default_value=""),
        Node(
            package="roboworld_perception",
            executable="perception_node",
            output="screen",
            parameters=[{
                "prompts": LaunchConfiguration("prompts"),
                "score_threshold": LaunchConfiguration("score_threshold"),
                "csv_path": LaunchConfiguration("csv_path"),
            }],
        ),
    ])
