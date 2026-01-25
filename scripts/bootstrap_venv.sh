#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
VENV_DIR="${WS_DIR}/.venv_ros"
REQ_FILE="${WS_DIR}/src/intent_comm_nav_ros/requirements_pip.txt"

echo "[bootstrap] Workspace: ${WS_DIR}"
echo "[bootstrap] ROS_DISTRO: ${ROS_DISTRO}"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "[bootstrap] ERROR: ROS setup not found: ${ROS_SETUP}"
  exit 1
fi

if [[ ! -x /usr/bin/python3.10 ]]; then
  echo "[bootstrap] ERROR: /usr/bin/python3.10 not found."
  echo "Install with:"
  echo "  sudo apt-get update"
  echo "  sudo apt-get install -y python3.10 python3.10-venv python3.10-dev"
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[bootstrap] ERROR: requirements file not found: ${REQ_FILE}"
  exit 1
fi

# --- If the venv was created with system-site-packages, it's NOT acceptable for pinned stacks.
if [[ -f "${VENV_DIR}/pyvenv.cfg" ]] && grep -q "include-system-site-packages = true" "${VENV_DIR}/pyvenv.cfg"; then
  echo "[bootstrap] WARNING: Existing venv includes system-site-packages (leaky). Recreating..."
  rm -rf "${VENV_DIR}"
fi

# --- Create clean venv
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[bootstrap] Creating clean venv at ${VENV_DIR}"
  /usr/bin/python3.10 -m venv "${VENV_DIR}"
fi

# Activate venv
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
echo "[bootstrap] Using python: $(which python)"
python -V

# Upgrade pip tooling (and keep setuptools pinned <81)
python -m pip install -U pip wheel
python -m pip install -U "setuptools<81"

# Install scientific stack (binary only to avoid building scipy)
echo "[bootstrap] Installing pinned deps from: ${REQ_FILE}"
python -m pip install --only-binary=:all: -r "${REQ_FILE}"

# Install colcon INSIDE the venv (so build uses venv python)
# This is critical to get console_scripts generated with venv interpreter.
python -m pip install -U colcon-core colcon-ros colcon-common-extensions

# Sanity checks
python -c "import numpy as np; print('numpy', np.__version__)"
python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"
echo "[bootstrap] colcon: $(which colcon)"

# Write env.sh (no nounset pitfalls)
ENV_SH="${WS_DIR}/scripts/env.sh"
cat > "${ENV_SH}" <<EOF
#!/usr/bin/env bash
set -e

WS_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"

# ROS environment
source /opt/ros/${ROS_DISTRO}/setup.bash

# Venv with pinned scientific stack
source "\${WS_DIR}/.venv_ros/bin/activate"

# Headless plotting
export MPLBACKEND=Agg

# Overlay workspace
if [[ -f "\${WS_DIR}/install/setup.bash" ]]; then
  source "\${WS_DIR}/install/setup.bash"
fi
EOF
chmod +x "${ENV_SH}"
echo "[bootstrap] Wrote: ${ENV_SH}"

# Build workspace with venv colcon
echo "[bootstrap] Building workspace with venv colcon..."

# IMPORTANT: ROS setup scripts can break under 'set -u', so temporarily disable nounset
set +u
source "${ROS_SETUP}"
set -u

cd "${WS_DIR}"
rm -rf build install log
colcon build --symlink-install

echo "[bootstrap] DONE."
echo "Next:"
echo "  cd ${WS_DIR}"
echo "  source scripts/env.sh"
echo "  head -n 1 install/intent_comm_nav_ros/lib/intent_comm_nav_ros/planar_nav_sim"
echo "  ros2 launch intent_comm_nav_ros planar_nav_sim.launch.py"

