from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nodes = []

    # --- sim (truth)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="planar_nav_sim",
        name="planar_nav_sim",
        output="screen",
        parameters=[
            {"dt_sim": 0.01},
            {"pose_pub_rate": 120.0},
            {"publish_mocap": False},
        ],
    ))

    # --- mocap noise (separate nodes like real OptiTrack setup)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="mocap_noiser_agent",
        name="mocap_noiser_human",
        output="screen",
        parameters=[
            {"input_topic": "/truth/human/pose"},
            {"output_topic": "/mocap/human/pose"},
            {"sigma_xy": 0.01},
            {"sigma_theta": 0.005},
            {"seed": 1},
        ],
    ))
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="mocap_noiser_agent",
        name="mocap_noiser_robot",
        output="screen",
        parameters=[
            {"input_topic": "/truth/robot/pose"},
            {"output_topic": "/mocap/robot/pose"},
            {"sigma_xy": 0.01},
            {"sigma_theta": 0.005},
            {"seed": 2},
        ],
    ))

    # --- pose -> pose2d
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="mocap_pose_to_2d",
        name="mocap_human_pose_to_2d",
        output="screen",
        parameters=[{"in_topic": "/mocap/human/pose"}],
    ))
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="mocap_pose_to_2d",
        name="mocap_robot_pose_to_2d",
        output="screen",
        parameters=[{"in_topic": "/mocap/robot/pose"}],
    ))

    # --- EKF stack (fast)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="ekf_stack",
        name="ekf_stack",
        output="screen",
        parameters=[
            {"ekf_rate": 100.0},
            {"publish_rate": 100.0},
            {"human_meas_topic": "/mocap/human/pose2d"},
            {"robot_meas_topic": "/mocap/robot/pose2d"},
            {"r_xy": 1e-4},
            {"r_theta": 1e-4},
        ],
    ))

    # --- HL bridge (fast->slow)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="high_level_bridge",
        name="high_level_bridge",
        output="screen",
        parameters=[
            {"dt_hl": 0.5},
            {"publish_rate": 2.0},
            {"x_topic": "/ekf/stacked_state"},
            {"u_h_topic": "/ekf/human/u_est"},
            {"u_r_topic": "/ekf/robot/u_est"},
        ],
    ))

    # --- HL runner (controllers @0.5s)
    nodes.append(Node(
        package="intent_comm_nav_ros",
        executable="high_level_runner",
        name="high_level_runner",
        output="screen",
        parameters=[
            {"dt_hl": 0.5},
            {"x_hat_topic": "/hl/x_hat"},
            {"u_h_obs_topic": "/hl/human/u_obs"},
            {"u_r_obs_topic": "/hl/robot/u_obs"},
        ],
    ))

    return LaunchDescription(nodes)

