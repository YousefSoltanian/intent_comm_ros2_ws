from setuptools import setup, find_packages
from glob import glob
import os

package_name = "intent_comm_nav_ros"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name), ["requirements_pip.txt"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@todo.com",
    description="ROS2 wrappers for Intent-Communication-HRI navigation + estimation stack.",
    license="Apache-2.0",
    entry_points={
    "console_scripts": [
        "heartbeat = intent_comm_nav_ros.nodes.heartbeat_node:main",
        "planar_nav_sim = intent_comm_nav_ros.nodes.planar_nav_sim_node:main",
        'mocap_noiser = intent_comm_nav_ros.nodes.mocap_noiser_node:main',
        'mocap_noiser_agent = intent_comm_nav_ros.nodes.mocap_noiser_agent_node:main',
        'mocap_pose_to_2d = intent_comm_nav_ros.nodes.mocap_pose_to_2d_node:main',
        "ekf_stack = intent_comm_nav_ros.nodes.ekf_stack_node:main",
	"cmd_vel_sine = intent_comm_nav_ros.nodes.cmd_vel_sine_node:main",
	"ekf_eval_logger = intent_comm_nav_ros.nodes.ekf_eval_logger_node:main",
	"high_level_bridge = intent_comm_nav_ros.nodes.high_level_bridge_node:main",
	"high_level_runner = intent_comm_nav_ros.nodes.high_level_runner_node:main",
	"hl_downsample = intent_comm_nav_ros.nodes.hl_downsample_node:main",
    	"nav_rollout_recorder = intent_comm_nav_ros.nodes.nav_rollout_recorder_node:main",
    	'high_level_robot_runner = intent_comm_nav_ros.nodes.high_level_robot_runner_node:main',
    	"mqtt_bridge = intent_comm_nav_ros.nodes.mqtt_bridge_node:main",

    ],
},

)

