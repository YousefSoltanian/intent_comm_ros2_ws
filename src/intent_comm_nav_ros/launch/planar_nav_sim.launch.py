from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nodes = []

    # 1) Truth simulator (ONLY in sim)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="planar_nav_sim",
        name="planar_nav_sim",
        output="screen",
        parameters=[
            {"dt_sim": 0.01},
            {"pose_pub_rate": 120.0},
            {"publish_mocap": False},   # IMPORTANT: turn off mocap publishing inside sim now
            {"seed": 0},
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
    ))

    # 2) Mocap noiser (human)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="mocap_noiser_agent",
        name="mocap_noiser_human",
        output="screen",
        parameters=[{
            "input_pose_topic": "/truth/human/pose",
            "output_pose_topic": "/mocap/human/pose",
            "sigma_xy": 0.01,
            "sigma_theta": 0.005,
            "seed": 1,
        }],
    ))

    # 3) Mocap noiser (robot)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="mocap_noiser_agent",
        name="mocap_noiser_robot",
        output="screen",
        parameters=[{
            "input_pose_topic": "/truth/robot/pose",
            "output_pose_topic": "/mocap/robot/pose",
            "sigma_xy": 0.01,
            "sigma_theta": 0.005,
            "seed": 2,
        }],
    ))

    # 4) Pose -> Pose2D/RPY (human)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="mocap_pose_to_2d",
        name="mocap_human_pose_to_2d",
        output="screen",
        parameters=[{
            "input_pose_topic": "/mocap/human/pose",
            "output_pose2d_topic": "/mocap/human/pose2d",
            "output_rpy_topic": "/mocap/human/rpy",
        }],
    ))

    # 5) Pose -> Pose2D/RPY (robot)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="mocap_pose_to_2d",
        name="mocap_robot_pose_to_2d",
        output="screen",
        parameters=[{
            "input_pose_topic": "/mocap/robot/pose",
            "output_pose2d_topic": "/mocap/robot/pose2d",
            "output_rpy_topic": "/mocap/robot/rpy",
        }],
    ))

    return LaunchDescription(nodes)

