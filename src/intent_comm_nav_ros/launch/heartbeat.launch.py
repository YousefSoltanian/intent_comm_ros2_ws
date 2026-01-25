from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="intent_comm_nav_ros",
            executable="heartbeat",
            name="intent_comm_heartbeat",
            output="screen",
        )
    ])

