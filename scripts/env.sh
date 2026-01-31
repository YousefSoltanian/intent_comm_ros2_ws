#!/usr/bin/env bash
set -e

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ROS environment
source /opt/ros/humble/setup.bash

# Venv with pinned scientific stack
source "${WS_DIR}/.venv_ros/bin/activate"

# Headless plotting
export MPLBACKEND=Agg

# Overlay workspace
if [[ -f "${WS_DIR}/install/setup.bash" ]]; then
  source "${WS_DIR}/install/setup.bash"
fi
