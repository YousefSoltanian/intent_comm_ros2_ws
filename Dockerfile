# Use official ROS 2 Humble image based on Ubuntu 22.04
FROM ros:humble

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-colcon-common-extensions \
    build-essential \
    git \
    cmake \
    wget \
    curl \
    nano \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /ws

# Copy the entire workspace
COPY . /ws/

ENV VIRTUAL_ENV=/ws/.venv_ros
ENV PATH="/ws/.venv_ros/bin:${PATH}"

# Set up Python virtual environment for non-ROS deps
RUN python3 -m venv /ws/.venv_ros && \
    . /ws/.venv_ros/bin/activate && \
    pip install --upgrade pip setuptools wheel

RUN . /ws/.venv_ros/bin/activate && \
    pip install -U colcon-core colcon-ros colcon-common-extensions


RUN . /ws/.venv_ros/bin/activate && \
    pip install -r /ws/src/intent_comm_nav_ros/requirements_pip.txt && \
    pip install numpy scipy matplotlib jax jaxlib scikit-learn paho-mqtt

# Build the workspace
RUN bash -c 'source /opt/ros/humble/setup.bash && \
    source /ws/.venv_ros/bin/activate && \
    cd /ws && \
    colcon build --symlink-install'

# Create rollouts directory
RUN mkdir -p /ws/rollouts

# Copy entrypoint script
COPY entrypoint.sh /ws/entrypoint.sh
RUN chmod +x /ws/entrypoint.sh

# Source setup files in bashrc for interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ws/.venv_ros/bin/activate" >> /root/.bashrc && \
    echo "source /ws/install/setup.bash" >> /root/.bashrc

# Set entrypoint to run the demo
ENTRYPOINT ["/ws/entrypoint.sh"]
