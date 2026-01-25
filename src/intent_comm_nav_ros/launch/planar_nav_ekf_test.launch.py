# launch/planar_nav_ekf_test.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nodes = []

    # --- 1) True simulator (environment)
    nodes.append(
        Node(
            package="intent_comm_nav_ros",
            executable="planar_nav_sim",
            name="planar_nav_sim",
            output="screen",
            parameters=[
                {"dt_sim": 0.01},
                {"pose_pub_rate": 120.0},
                {"publish_mocap": False},  # we use separate mocap noiser nodes
            ],
        )
    )

    # --- 2) Mocap noiser (HUMAN) - uses defaults /truth/human/pose -> /mocap/human/pose
    # If your mocap_noiser_agent node supports input_topic/output_topic params, keep them.
    # Even if it doesn't, the defaults will still work for HUMAN.
    nodes.append(
        Node(
            package="intent_comm_nav_ros",
            executable="mocap_noiser_agent",
            name="mocap_noiser_human",
            output="screen",
            parameters=[
                {"sigma_xy": 0.01},
                {"sigma_theta": 0.005},
                {"seed": 1},
                # keep these if your node actually reads them (harmless if ignored)
                {"input_topic": "/truth/human/pose"},
                {"output_topic": "/mocap/human/pose"},
            ],
        )
    )

    # --- 3) Mocap noiser (ROBOT) - force correct topics via remapping
    # This FIXES your bug where the robot instance was still subscribing/publishing human topics.
    nodes.append(
        Node(
            package="intent_comm_nav_ros",
            executable="mocap_noiser_agent",
            name="mocap_noiser_robot",
            output="screen",
            parameters=[
                {"sigma_xy": 0.01},
                {"sigma_theta": 0.005},
                {"seed": 2},
                {"input_topic": "/truth/robot/pose"},   # keep if node reads it
                {"output_topic": "/mocap/robot/pose"},  # keep if node reads it
            ],
            remappings=[
                ("/truth/human/pose", "/truth/robot/pose"),
                ("/mocap/human/pose", "/mocap/robot/pose"),
            ],
        )
    )

    # --- 4) Pose -> Pose2D + RPY (HUMAN)
    nodes.append(
        Node(
            package="intent_comm_nav_ros",
            executable="mocap_pose_to_2d",
            name="mocap_human_pose_to_2d",
            output="screen",
            parameters=[
                {"in_topic": "/mocap/human/pose"},  # keep if node reads it
            ],
        )
    )

    # --- 5) Pose -> Pose2D + RPY (ROBOT) via remapping
    nodes.append(
        Node(
            package="intent_comm_nav_ros",
            executable="mocap_pose_to_2d",
            name="mocap_robot_pose_to_2d",
            output="screen",
            parameters=[
                {"in_topic": "/mocap/robot/pose"},  # keep if node reads it
            ],
            remappings=[
                ("/mocap/human/pose", "/mocap/robot/pose"),
                ("/mocap/human/pose2d", "/mocap/robot/pose2d"),
                ("/mocap/human/rpy", "/mocap/robot/rpy"),
            ],
        )
    )

    # --- 6) EKF stack: listens to pose2d, publishes stacked_state + u_est
    nodes.append(
        Node(
            package="intent_comm_nav_ros",
            executable="ekf_stack",
            name="ekf_stack",
            output="screen",
            parameters=[
                {"ekf_rate": 100.0},
                {"publish_rate": 100.0},
                {"human_meas_topic": "/mocap/human/pose2d"},
                {"robot_meas_topic": "/mocap/robot/pose2d"},

                # measurement noise (example)
                {"r_xy": 1e-4},
                {"r_theta": 5e-4},

                # process noise for augmented input states (example)
                {"human.q_v": 1e-2}, {"human.q_w": 1e-2},
                {"robot.q_v": 1e-2}, {"robot.q_w": 1e-2},
            ],
        )
    )

    # --- 7) Optional: scripted cmd_vel to excite the system (debug EKF)
    nodes.append(
        Node(
            package="intent_comm_nav_ros",
            executable="cmd_vel_sine",
            name="cmd_vel_sine",
            output="screen",
            parameters=[
                {"rate_hz": 20.0},
                {"v0": 0.6}, {"v_amp": 0.25}, {"v_freq_hz": 0.2},
                {"w0": 0.0}, {"w_amp": 0.3}, {"w_freq_hz": 0.12},
            ],
        )
    )
    
    nodes.append(Node(
    package="intent_comm_nav_ros",
    executable="ekf_eval_logger",
    name="ekf_eval_logger",
    output="screen",
    parameters=[
        {"duration_sec": 10.0},
        {"out_dir": "/home/ssoltan2/intent_comm_ros2_ws/ekf_eval_out"},
        {"human_cmd_topic": "/human/cmd_vel"},
        {"robot_cmd_topic": "/robot/cmd_vel"},
        {"human_u_est_topic": "/ekf/human/u_est"},
        {"robot_u_est_topic": "/ekf/robot/u_est"},
    ],
))


    return LaunchDescription(nodes)

