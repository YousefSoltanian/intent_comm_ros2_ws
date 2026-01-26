from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler, DeclareLaunchArgument, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    # Only CLI inputs allowed (same pattern as your sim demo)
    controller = LaunchConfiguration("controller")
    trial_id = LaunchConfiguration("trial_id")

    out_dir = os.path.join(os.getcwd(), "rollouts")

    # -----------------------
    # EDIT THESE MANUALLY ONLY
    # Experiment timing + controller params
    # -----------------------
    EXP = {
        "dt_hl": 0.5,
        "horizon": 10,

        "intents_h": [0, 1],
        "intents_r": [0, 1],
        "theta_h_true": 0,
        "theta_r_true": 0,

        "goals_h_flat": [4.0, 1.6, 4.0, -1.6],
        "goals_r_flat": [-4.0, 1.6, -4.0, -1.6],

        "w_goal_pos": 60.0,
        "w_head": 10.108,
        "w_speed": 0.0,
        "w_eff": 500.08,
        "v_nom": 0.9,

        "w_lat": 0.2,
        "w_wall": 100.0,
        "w_coll": 20.0,
        "r_safe_coll": 3.0,

        "hall_y0": 0.0,
        "hall_half_width": 2.41,

        "v_lo": 0.0,
        "v_hi": 0.5,     # SAFETY: start lower on real robot
        "w_lo": -0.4,
        "w_hi": 0.4,

        "max_iter": 25,
        "beta_r": 0.1,
        "beta_state": 1.1,
        "rho_forget": 0.0,
        "sigma2_state_robot_flat": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],

        "gamma_teach": 100.0,
        "effort_w_qmdp": 0.0,

        "beta_action_like": 0.1,
        "sigma2_action_obs_flat": [0.1, 0.1],
    }

    # -----------------------
    # EDIT THESE MANUALLY ONLY
    # EKF tuning (defaults match EKFStackNode)
    # -----------------------
    EKF = {
        "ekf_rate": 100.0,

        "p0_xy": 0.5,
        "p0_th": 0.5,
        "p0_v": 1.0,
        "p0_w": 1.0,

        "q_xy": 1e-6,
        "q_th": 1e-6,
        "q_v": 1e-4,
        "q_w": 1e-4,

        "r_xy": 1e-4,
        "r_th": 1e-3,

        "freeze_vel_s": 0.5,
        "freeze_vel_cov": 1e-4,
        "max_v": 1.5,
        "max_w": 1.0,
    }

    # -----------------------
    # EDIT THESE MANUALLY ONLY
    # Real experiment run length (seconds)
    # -----------------------
    DURATION_S = 30.0

    # Mocap -> EKF
    # IMPORTANT: replace these with your REAL mocap topics
    MOCAP_H_TOPIC = "/mocap/human/pose"   # TODO: set to your mocap human PoseStamped topic
    MOCAP_R_TOPIC = "/mocap/robot/pose"   # TODO: set to your mocap fetch PoseStamped topic

    ekf_stack = Node(
        package="intent_comm_nav_ros",
        executable="ekf_stack",
        name="ekf_stack",
        output="screen",
        parameters=[{
            "ekf_rate": EKF["ekf_rate"],

            "p0_xy": EKF["p0_xy"],
            "p0_th": EKF["p0_th"],
            "p0_v": EKF["p0_v"],
            "p0_w": EKF["p0_w"],

            "q_xy": EKF["q_xy"],
            "q_th": EKF["q_th"],
            "q_v": EKF["q_v"],
            "q_w": EKF["q_w"],

            "r_xy": EKF["r_xy"],
            "r_th": EKF["r_th"],

            "freeze_vel_s": EKF["freeze_vel_s"],
            "freeze_vel_cov": EKF["freeze_vel_cov"],
            "max_v": EKF["max_v"],
            "max_w": EKF["max_w"],

            "topic_h_pose": MOCAP_H_TOPIC,
            "topic_r_pose": MOCAP_R_TOPIC,
            "topic_x_hat": "/ekf/stacked_state",
            "topic_h_u_est": "/ekf/human/u_est",
            "topic_r_u_est": "/ekf/robot/u_est",
        }],
    )

    high_level_bridge = Node(
        package="intent_comm_nav_ros",
        executable="high_level_bridge",
        name="high_level_bridge",
        output="screen",
        parameters=[{
            "dt_hl": EXP["dt_hl"],
            "x_hat_in_topic": "/ekf/stacked_state",
            "human_u_est_topic": "/ekf/human/u_est",
            "robot_u_est_topic": "/ekf/robot/u_est",
            "x_hat_topic": "/hl/x_hat",
            "x_hat_prev_topic": "/hl/x_hat_prev",
            "human_u_obs_topic": "/hl/human/u_obs",
            "robot_u_obs_topic": "/hl/robot/u_obs",
        }],
    )

    runner_params = dict(EXP)
    runner_params.update({
        "controller": ParameterValue(controller, value_type=str),

        "topic_x_hat": "/hl/x_hat",
        "topic_h_u_obs": "/hl/human/u_obs",

        # Publish directly to /cmd_vel for easier ROS1 bridging
        "topic_r_cmd": "/cmd_vel",

        # enforce stop by time (node also publishes zero on shutdown)
        "max_runtime_s": DURATION_S,
    })

    high_level_robot_runner = Node(
        package="intent_comm_nav_ros",
        executable="high_level_robot_runner",
        name="high_level_robot_runner",
        output="screen",
        parameters=[runner_params],
    )

    # Recorder: treat mocap pose as "truth" for logging
    recorder_params = {
        "controller": ParameterValue(controller, value_type=str),
        "trial_id": ParameterValue(trial_id, value_type=int),

        "duration_s": DURATION_S,
        "plot_dt": 0.02,
        "gif_stride": 10,
        "out_dir": out_dir,

        "topic_truth_h": MOCAP_H_TOPIC,
        "topic_truth_r": MOCAP_R_TOPIC,
        "topic_cmd_h": "/human/cmd_vel",   # TODO: leave if unused OR point to a real human cmd topic if you have one
        "topic_cmd_r": "/cmd_vel",
        "topic_u_est_h": "/ekf/human/u_est",
        "topic_u_est_r": "/ekf/robot/u_est",
        "topic_u_obs_h": "/hl/human/u_obs",
        "topic_u_obs_r": "/hl/robot/u_obs",
        "topic_belief_h": "/hl/beliefs/human_about_robot",
        "topic_belief_r": "/hl/beliefs/robot_about_human",

        "intents_h": EXP["intents_h"],
        "intents_r": EXP["intents_r"],
        "theta_h_true": EXP["theta_h_true"],
        "theta_r_true": EXP["theta_r_true"],
        "goals_h_flat": EXP["goals_h_flat"],
        "goals_r_flat": EXP["goals_r_flat"],
    }

    recorder = Node(
        package="intent_comm_nav_ros",
        executable="nav_rollout_recorder",
        name="nav_rollout_recorder",
        output="screen",
        parameters=[recorder_params],
    )

    shutdown_on_recorder_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=recorder,
            on_exit=[EmitEvent(event=Shutdown(reason="Recorder finished"))],
        )
    )

    # Extra hard shutdown even if recorder doesn’t exit (belt & suspenders)
    hard_shutdown = TimerAction(
        period=DURATION_S + 2.0,
        actions=[EmitEvent(event=Shutdown(reason="Experiment duration reached"))],
    )

    return LaunchDescription([
        DeclareLaunchArgument("controller", default_value="npace"),
        DeclareLaunchArgument("trial_id", default_value="0"),

        ekf_stack,
        high_level_bridge,
        high_level_robot_runner,
        recorder,
        shutdown_on_recorder_exit,
        hard_shutdown,
    ])

