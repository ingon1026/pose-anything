from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("prompts", default_value="물통"),
        DeclareLaunchArgument("score_threshold", default_value="0.4"),
        DeclareLaunchArgument("csv_path", default_value=""),
        DeclareLaunchArgument("rviz", default_value="true"),
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
            }],
        ),
    ])
