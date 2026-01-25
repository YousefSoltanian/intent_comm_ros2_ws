from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    # Only CLI inputs allowed
    controller = LaunchConfiguration("controller")
    trial_id = LaunchConfiguration("trial_id")

    out_dir = os.path.join(os.getcwd(), "rollouts")

    # -----------------------
    # EDIT THESE MANUALLY ONLY
    # (defaults match your current code)
    # -----------------------
    EXP = {
        # controller timing
        "dt_hl": 0.5,
        "horizon": 10,

        # intents + true indices
        "intents_h": [0, 1],
        "intents_r": [0, 1],
        "theta_h_true": 0,
        "theta_r_true": 0,

        # goals (human goes to +4, robot goes to -4)
        "goals_h_flat": [4.0, 1.6, 4.0, -1.6],
        "goals_r_flat": [-4.0, 1.6, -4.0, -1.6],

        # costs / weights
        "w_goal_pos": 60.0,
        "w_head": 10.108,
        "w_speed": 0.0,
        "w_eff": 500.08,
        "v_nom": 0.9,

        "w_lat": 0.2,
        "w_wall": 100.0,
        "w_coll": 20.0,
        "r_safe_coll": 3.0,

        # corridor
        "hall_y0": 0.0,
        "hall_half_width": 2.41,

        # control limits
        "v_lo": 0.0,
        "v_hi": 1.2,
        "w_lo": -0.6,
        "w_hi": 0.6,

        # solver/noise
        "max_iter": 25,
        "beta_h": 0.1,
        "beta_r": 0.1,
        "beta_state": 1.1,
        "rho_forget": 0.0,
        "sigma2_state_human_flat": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "sigma2_state_robot_flat": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],

        # NPACE only
        "gamma_teach": 100.0,
        "effort_w_qmdp": 0.0,

        # Blame-Me only
        "beta_action_like": 0.1,
        "sigma2_action_obs_flat": [0.1, 0.1],
    }

    # -----------------------
    # EDIT THESE MANUALLY ONLY
    # EKF tuning (defaults match EKFStackNode)
    # -----------------------
    EKF = {
        "ekf_rate": 100.0,

        # initial covariance (variances)
        "p0_xy": 0.5*2,
        "p0_th": 0.5*2,
        "p0_v": 1.0*2,
        "p0_w": 1.0*2,

        # process noise (variances)
        "q_xy": 1e-6,
        "q_th": 1e-6,
        "q_v": 1e-4,
        "q_w": 1e-4,

        # measurement noise (variances)
        "r_xy": 1e-4,
        "r_th": 1e-3,

        # freeze/clamp
        "freeze_vel_s": 0.5,
        "freeze_vel_cov": 1e-4,
        "max_v": 2.0,
        "max_w": 1.0,
    }

    planar_sim = Node(
        package="intent_comm_nav_ros",
        executable="planar_nav_sim",
        name="planar_nav_sim",
        output="screen",
        parameters=[{
            "dt_sim": 0.01,
            "pose_pub_rate": 120.0,
            "publish_mocap": True,
            "cmd_h_topic": "/human/cmd_vel",
            "cmd_r_topic": "/robot/cmd_vel",
        }],
    )

    mocap_noiser_human = Node(
        package="intent_comm_nav_ros",
        executable="mocap_noiser_agent",
        name="mocap_noiser_human",
        output="screen",
        parameters=[{
            "input_topic": "/truth/human/pose",
            "output_topic": "/mocap/human/pose",
            "sigma_xy": 0.005,
            "sigma_theta": 0.002,
            "seed": 1,
        }],
    )

    mocap_noiser_robot = Node(
        package="intent_comm_nav_ros",
        executable="mocap_noiser_agent",
        name="mocap_noiser_robot",
        output="screen",
        parameters=[{
            "input_topic": "/truth/robot/pose",
            "output_topic": "/mocap/robot/pose",
            "sigma_xy": 0.005,
            "sigma_theta": 0.002,
            "seed": 2,
        }],
    )

    mocap_pose_to_2d_human = Node(
        package="intent_comm_nav_ros",
        executable="mocap_pose_to_2d",
        name="mocap_human_pose_to_2d",
        output="screen",
        parameters=[{
            "in_topic": "/mocap/human/pose",
            "output_pose2d_topic": "/mocap/human/pose2d",
            "output_rpy_topic": "/mocap/human/rpy",
        }],
    )

    mocap_pose_to_2d_robot = Node(
        package="intent_comm_nav_ros",
        executable="mocap_pose_to_2d",
        name="mocap_robot_pose_to_2d",
        output="screen",
        parameters=[{
            "in_topic": "/mocap/robot/pose",
            "output_pose2d_topic": "/mocap/robot/pose2d",
            "output_rpy_topic": "/mocap/robot/rpy",
        }],
    )

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

            "topic_h_pose": "/mocap/human/pose",
            "topic_r_pose": "/mocap/robot/pose",
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
        # only CLI switch
        "controller": ParameterValue(controller, value_type=str),

        # IMPORTANT: use bridge output
        "topic_x_hat": "/hl/x_hat",
        "topic_h_u_obs": "/hl/human/u_obs",
        "topic_r_u_obs": "/hl/robot/u_obs",
        "topic_h_cmd": "/human/cmd_vel",
        "topic_r_cmd": "/robot/cmd_vel",
    })

    high_level_runner = Node(
        package="intent_comm_nav_ros",
        executable="high_level_runner",
        name="high_level_runner",
        output="screen",
        parameters=[runner_params],
    )

    recorder_params = {
        # ONLY identifiers
        "controller": ParameterValue(controller, value_type=str),
        "trial_id": ParameterValue(trial_id, value_type=int),

        # recorder config
        "duration_s": 15.0,
        "plot_dt": 0.02,
        "gif_stride": 10,
        "out_dir": out_dir,

        # topics
        "topic_truth_h": "/truth/human/pose",
        "topic_truth_r": "/truth/robot/pose",
        "topic_cmd_h": "/human/cmd_vel",
        "topic_cmd_r": "/robot/cmd_vel",
        "topic_u_est_h": "/ekf/human/u_est",
        "topic_u_est_r": "/ekf/robot/u_est",
        "topic_u_obs_h": "/hl/human/u_obs",
        "topic_u_obs_r": "/hl/robot/u_obs",
        "topic_belief_h": "/hl/beliefs/human_about_robot",
        "topic_belief_r": "/hl/beliefs/robot_about_human",

        # must match EXP (belief indexing + plotting)
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

    return LaunchDescription([
        DeclareLaunchArgument("controller", default_value="npace"),
        DeclareLaunchArgument("trial_id", default_value="0"),

        planar_sim,
        mocap_noiser_human,
        mocap_noiser_robot,
        mocap_pose_to_2d_human,
        mocap_pose_to_2d_robot,
        ekf_stack,
        high_level_bridge,
        high_level_runner,
        recorder,
        shutdown_on_recorder_exit,
    ])

