# intent_comm_ros2_ws

ROS 2 workspace for intent communication / mutual learning demos in planar navigation.

This repository contains a self-contained ROS 2 workspace (`intent_comm_ros2_ws/`) with:
- a lightweight 2D simulation,
- a mocap-style measurement pipeline (with configurable noise),
- an EKF stack that estimates stacked human/robot state,
- high-level controllers (e.g., N-PACE, Blame-Me),
- a recorder that logs + plots + exports rollouts.

> **Status / Scope**
> - Designed for **ROS 2 Humble** on Ubuntu (tested on Ubuntu 22.04).
> - Uses Python nodes and vendorized game/ILQ solver code.
> - GPU is optional; some controllers use JAX and will fall back to CPU if CUDA-enabled `jaxlib` is not installed.

---

## Repository layout

This repo is a ROS 2 workspace. The main code lives under `src/`:

```
intent_comm_ros2_ws/
├─ src/
│  ├─ intent_comm_hri_vendor/      # ILQ / game solver code (vendorized)
│  ├─ intent_comm_nav_vendor/      # high-level controllers (e.g., N-PACE, Blame-Me)
│  └─ intent_comm_nav_ros/         # ROS nodes + launch files (runner, mocap, EKF, bridge, recorder)
├─ scripts/                        # environment setup / helper scripts
│  ├─ env.sh
│  └─ bootstrap_venv.sh
└─ rollouts/                       # auto-created output directory for recorded trials (default)
```

**Packages**
- **`intent_comm_nav_ros`**: ROS nodes + launch files (simulation, mocap pipeline, EKF stack, bridge, runner, recorder).
- **`intent_comm_nav_vendor`**: High-level controllers called by the runner node.
- **`intent_comm_hri_vendor`**: iLQ / ILQGame solver code used by controllers.

---

## Quickstart

### 1) System requirements
- Ubuntu 22.04
- ROS 2 Humble installed and sourced
- Python 3.10+
- `colcon` and common ROS build tools

If you do not have ROS 2 Humble installed yet, follow the official ROS docs for Humble on Ubuntu 22.04. *(link placeholder)*

### 2) Clone
```bash
git clone <REPO_URL_PLACEHOLDER>
cd intent_comm_ros2_ws
```

### 3) Set up Python environment (recommended)
This workspace supports a Python virtual environment for non-ROS Python deps (e.g., JAX, numpy, matplotlib).

The `scripts/` folder contains helper scripts:
- `scripts/env.sh`: environment helpers (sourceable)
- `scripts/bootstrap_venv.sh`: creates / configures `.venv_ros` (recommended)

Example:
```bash
source /opt/ros/humble/setup.bash

# Create/activate venv + install python deps (see script for details)
bash scripts/bootstrap_venv.sh

# Activate venv (if not already activated by the script)
source .venv_ros/bin/activate
```

> If you prefer conda instead of venv, replace the venv steps with your own environment setup and ensure required Python packages are installed.

### 4) Build the workspace
```bash
source /opt/ros/humble/setup.bash
# If using venv:
source .venv_ros/bin/activate

colcon build --symlink-install
source install/setup.bash
```

### 5) Run a demo rollout (simulation + EKF + controller + recorder)
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
# If using venv:
source .venv_ros/bin/activate

ros2 launch intent_comm_nav_ros navigation_demo_record.launch.py controller:=npace trial_id:=0
```

Common controller options (depending on your code/config):
- `controller:=npace`
- `controller:=blame_me`

Recorded artifacts are saved under `./rollouts/` by default.

---

## What runs in `navigation_demo_record.launch.py`

The launch file starts a complete end-to-end pipeline:

1. **Simulation**
   - `planar_nav_sim`  
     Publishes truth poses for human/robot and accepts velocity commands.

2. **Mocap-like measurement pipeline**
   - `mocap_noiser_agent` (human)  
   - `mocap_noiser_agent` (robot)  
     Adds Gaussian noise to truth measurements (position + yaw).
   - `mocap_pose_to_2d` (human)  
   - `mocap_pose_to_2d` (robot)  
     Convenience conversion from `PoseStamped` to 2D outputs (Pose2D, RPY).

3. **Estimation**
   - `ekf_stack`  
     Runs EKF(s) for human and robot and publishes a stacked state:
     `x_hat = [h_px, h_py, h_theta, r_px, r_py, r_theta]` and estimated controls for each agent:
     `u_est = [v, w]`.

4. **High-rate to low-rate bridge**
   - `high_level_bridge`  
     Bridges EKF outputs at high rate (e.g., 100 Hz) to high-level control rate (e.g., 0.5 s).
     - Publishes `x_hat` (and `x_hat_prev`) to the runner.
     - Publishes observed actions `u_obs` as windowed summaries of EKF-estimated velocities.

5. **High-level control**
   - `high_level_runner`  
     Reads low-rate `x_hat`, `u_obs`, runs the selected controller, and publishes `cmd_vel` for both agents.

6. **Recorder**
   - `nav_rollout_recorder`  
     Logs topics and exports plots / GIFs / metadata for each trial.

---

## Key ROS nodes and topics

> Topic names can be configured via launch parameters; defaults shown below.

### Simulation
- Subscribes:
  - `/human/cmd_vel` (`geometry_msgs/Twist`)
  - `/robot/cmd_vel` (`geometry_msgs/Twist`)
- Publishes:
  - `/truth/human/pose` (`geometry_msgs/PoseStamped`)
  - `/truth/robot/pose` (`geometry_msgs/PoseStamped`)

### Mocap noise
- Input: `/truth/*/pose`
- Output: `/mocap/*/pose`

### EKF stack
- Inputs:
  - `/mocap/human/pose`
  - `/mocap/robot/pose`
- Outputs:
  - `/ekf/stacked_state` (`std_msgs/Float32MultiArray`)
  - `/ekf/human/u_est` (`std_msgs/Float32MultiArray`)
  - `/ekf/robot/u_est` (`std_msgs/Float32MultiArray`)

### Bridge (low-rate outputs)
- Outputs:
  - `/hl/x_hat` (`std_msgs/Float32MultiArray`)
  - `/hl/x_hat_prev` (`std_msgs/Float32MultiArray`)
  - `/hl/human/u_obs` (`geometry_msgs/Twist`)
  - `/hl/robot/u_obs` (`geometry_msgs/Twist`)

### Runner beliefs (if enabled by the controller)
- `/hl/beliefs/human_about_robot`
- `/hl/beliefs/robot_about_human`
*(message type placeholder — depends on implementation)*

---

## Configuration & tuning

### Editing experiment parameters
Open:
- `src/intent_comm_nav_ros/launch/navigation_demo_record.launch.py`

Look for the `EXP = {...}` block and update values as needed:
- high-level timestep (`dt_hl`), horizon
- goals and costs
- solver params (`max_iter`, betas, etc.)
- teaching params (`gamma_teach`)
- blame-me params (`beta_action_like`, `sigma2_action_obs_flat`)

### Tuning EKF parameters from the launch file
The EKF node (`ekf_stack`) exposes parameters such as:
- `p0_xy`, `p0_th`, `p0_v`, `p0_w` (initial covariance)
- `q_xy`, `q_th`, `q_v`, `q_w` (process noise)
- `r_xy`, `r_th` (measurement noise)
- `freeze_vel_s`, `freeze_vel_cov`, `max_v`, `max_w`

To tune from launch, add/edit the EKF node `parameters=[{...}]` block in the launch file.

*(If you do not see these parameters in your local copy, update your `ekf_stack_node.py` accordingly.)*

### Mocap noise parameters
In the launch file, `mocap_noiser_agent` uses:
- `sigma_xy` (meters)
- `sigma_theta` (radians, yaw noise)
- `seed`

Use these to simulate different motion capture quality levels.

---

## Outputs

By default, the recorder writes to:
- `./rollouts/`

Typical artifacts (names may vary by controller / configuration):
- rollout plots (trajectories, controls, beliefs)
- animated GIF
- trial metadata (JSON/PKL placeholder)

---

## Troubleshooting

### “JAX falling back to CPU”
If you see messages about CUDA-enabled `jaxlib` not being installed, it is safe to continue (CPU fallback).
To enable GPU, install the correct CUDA-compatible `jaxlib` build for your system. *(link placeholder)*

### “ImportError: initialization failed” in JAX / jaxlib
This often indicates an ABI mismatch (CUDA toolkit, driver, `jaxlib` build, or Python env mismatch).
Try:
1. Recreate the venv from scratch.
2. Ensure `jax`/`jaxlib` versions are compatible with your CUDA (or install CPU-only `jaxlib`).
3. Confirm you are running from the same environment used to build the workspace.

### ROS messages / type support errors
If message type support fails, verify:
- You sourced ROS (`source /opt/ros/humble/setup.bash`)
- You sourced the workspace (`source install/setup.bash`)
- You are not mixing system Python with venv Python in a way that breaks ROS Python packages.

---

## Citation
If you use this code in academic work, please cite:

- **Paper / preprint 1**: *(citation placeholder)*
- **Paper / preprint 2**: *(citation placeholder)*
- **Paper / preprint 3**: *(citation placeholder)*

A BibTeX entry will be provided in `CITATION.cff` or `bibtex.bib` in a future update. *(placeholder)*

---

## License
License: *(placeholder — e.g., MIT / BSD-3-Clause / Apache-2.0)*

---

## Contact
Maintainer: *(name placeholder)*  
Email: *(email placeholder)*  
Issues / PRs: welcome.
