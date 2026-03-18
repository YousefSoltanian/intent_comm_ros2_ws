# launch/experiment.launch.py
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler, DeclareLaunchArgument, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os
import random

def generate_launch_description():
    # Randomly assign robot intent (0 or 1) once per run
    # theta_r_true = random.randint(0, 1)
    # print("[LAUNCH FILE] True Robot Theta: " + str(theta_r_true))

    # CLI inputs allowed
    controller = LaunchConfiguration("controller")
    # trial_id = LaunchConfiguration("trial_id")
    # subject_name = LaunchConfiguration("subject_name")
    params_file = LaunchConfiguration("params_file")  # path to YAML on the container

    # default out_dir; will be combined with subject and trial for per-run folder
    base_out_dir = os.path.join(os.getcwd(), "rollouts")

    # Duration (kept as before)
    DURATION_S = 120.0

    # Topics used by nodes
    MOCAP_H_TOPIC = "/mocap/human/pose"
    MOCAP_R_TOPIC = "/mocap/robot/pose"

    # MQTT Bridge — load params from YAML (params_file). YAML keys override these dict values if present.
    mqtt_bridge = Node(
        package="intent_comm_nav_ros",
        executable="mqtt_bridge",
        name="mqtt_bridge",
        output="screen",
        parameters=[
            params_file,
            {
                "mqtt_client_id": "ros2_mqtt_bridge",
                "mqtt_keepalive": 60,
                "mqtt_reconnect_max_delay": 60,
                "ros_topic_human": MOCAP_H_TOPIC,
                "ros_topic_robot": MOCAP_R_TOPIC,
                # "theta_r_true": theta_r_true,
            },
        ],
    )

    ekf_stack = Node(
        package="intent_comm_nav_ros",
        executable="ekf_stack",
        name="ekf_stack",
        output="screen",
        parameters=[
            params_file,
            {
                "topic_h_pose": MOCAP_H_TOPIC,
                "topic_r_pose": MOCAP_R_TOPIC,
                "topic_x_hat": "/ekf/stacked_state",
                "topic_h_u_est": "/ekf/human/u_est",
                "topic_r_u_est": "/ekf/robot/u_est",
                # "theta_r_true": theta_r_true,
            },
        ],
    )

    high_level_bridge = Node(
        package="intent_comm_nav_ros",
        executable="high_level_bridge",
        name="high_level_bridge",
        output="screen",
        parameters=[
            params_file,
            {
                "x_hat_in_topic": "/ekf/stacked_state",
                "human_u_est_topic": "/ekf/human/u_est",
                "robot_u_est_topic": "/ekf/robot/u_est",
                "x_hat_topic": "/hl/x_hat",
                "x_hat_prev_topic": "/hl/x_hat_prev",
                "human_u_obs_topic": "/hl/human/u_obs",
                "robot_u_obs_topic": "/hl/robot/u_obs",
                # "theta_r_true": theta_r_true,
            },
        ],
    )

    runner_params = {
        "controller": ParameterValue(controller, value_type=str),
        "topic_x_hat": "/hl/x_hat",
        "topic_h_u_obs": "/hl/human/u_obs",
        "topic_r_cmd": "/robot/cmd_vel",
        "max_runtime_s": DURATION_S,
        # "theta_r_true": theta_r_true,
    }

    high_level_runner = Node(
        package="intent_comm_nav_ros",
        executable="high_level_robot_runner",
        name="high_level_robot_runner",
        output="screen",
        parameters=[params_file, runner_params],
    )

    recorder_params = {
        "controller": ParameterValue(controller, value_type=str),
        # "trial_id": ParameterValue(trial_id, value_type=int),

        # recorder config keys: file can override via params_file
        "out_dir": base_out_dir,
        "duration_s": 10.0,
        "plot_dt": 0.02,
        "gif_stride": 10,

        # topics
        "topic_truth_h": MOCAP_H_TOPIC,
        "topic_truth_r": MOCAP_R_TOPIC,
        "topic_cmd_h": "/human/cmd_vel",
        "topic_cmd_r": "/robot/cmd_vel",
        "topic_u_est_h": "/ekf/human/u_est",
        "topic_u_est_r": "/ekf/robot/u_est",
        "topic_u_obs_h": "/hl/human/u_obs",
        "topic_u_obs_r": "/hl/robot/u_obs",
        "topic_belief_h": "/hl/beliefs/human_about_robot",
        "topic_belief_r": "/hl/beliefs/robot_about_human",
        # "theta_r_true": theta_r_true,
    }

    recorder = Node(
        package="intent_comm_nav_ros",
        executable="nav_rollout_recorder",
        name="nav_rollout_recorder",
        output="screen",
        parameters=[params_file, recorder_params],
    )

    shutdown_on_recorder_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=recorder,
            on_exit=[EmitEvent(event=Shutdown(reason="Recorder finished"))],
        )
    )

    hard_shutdown = TimerAction(
        period=DURATION_S + 2.0,
        actions=[EmitEvent(event=Shutdown(reason="Experiment duration reached"))],
    )

    return LaunchDescription([
        DeclareLaunchArgument("controller", default_value="npace"),
        # DeclareLaunchArgument("trial_id", default_value="0"),
        # DeclareLaunchArgument("subject_name", default_value="subject_0"),
        DeclareLaunchArgument("params_file", default_value=os.path.join("/ws", "config", "experiment_params.yaml")),

        mqtt_bridge,
        ekf_stack,
        high_level_bridge,
        high_level_runner,
        recorder,
        shutdown_on_recorder_exit,
        hard_shutdown
    ])
