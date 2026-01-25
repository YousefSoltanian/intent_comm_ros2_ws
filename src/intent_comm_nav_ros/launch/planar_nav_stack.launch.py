from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 1) True environment simulator: publish truth only
        Node(
            package="intent_comm_nav_ros",
            executable="planar_nav_sim",
            name="planar_nav_sim",
            output="screen",
            parameters=[
                {"dt_sim": 0.01},
                {"pose_pub_rate": 120.0},
                {"publish_mocap": False},   # IMPORTANT: turn off mocap here
                {"v_max": 1.0},
                {"w_max": 0.4},
                {"x1_init": -4.0},
                {"y1_init": 0.0},
                {"th1_init": 0.0},
                {"x2_init": 4.0},
                {"y2_init": 0.0},
                {"th2_init": 3.141592653589793},
                {"frame_id": "map"},
            ],
        ),

        # 2) Mocap measurement generator: noisy pose only
        Node(
            package="intent_comm_nav_ros",
            executable="mocap_noiser",
            name="mocap_noiser",
            output="screen",
            parameters=[
                {"sigma_xy": 0.01},
                {"sigma_theta": 0.005},
                {"seed": 0},
                {"truth_human_topic": "/truth/human/pose"},
                {"truth_robot_topic": "/truth/robot/pose"},
                {"mocap_human_topic": "/mocap/human/pose"},
                {"mocap_robot_topic": "/mocap/robot/pose"},
            ],
        ),
    ])

