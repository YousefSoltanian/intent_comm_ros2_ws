#!/bin/bash
set -e

# Note: Mosquitto broker runs on host machine (see DOCKER_SETUP.md)
echo "Starting ROS 2 application..."
echo "Connecting to MQTT broker at: ${MQTT_BROKER_HOST:-localhost}:${MQTT_BROKER_PORT:-1883}"

# Source ROS setup
source /opt/ros/humble/setup.bash

# Activate Python virtual environment
source /ws/.venv_ros/bin/activate

# Source workspace setup
source /ws/install/setup.bash

# Run the demo
ros2 launch intent_comm_nav_ros navigation_real_record.launch.py controller:=${CONTROLLER:-npace} trial_id:=${TRIAL_ID:-0}
