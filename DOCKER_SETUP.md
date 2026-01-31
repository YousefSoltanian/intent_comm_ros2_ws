# Docker Setup Guide

This guide explains how to build and run the intent_comm_ros2_ws workspace using Docker.

## Prerequisites

- **Docker**: Install from [docker.com](https://docs.docker.com/get-docker/)
- **Docker Compose** (optional): Usually included with Docker Desktop

## Building the Docker Image

### Option 1: Using Docker directly

```bash
cd /path/to/intent_comm_ros2_ws
docker build -t intent-comm-ros2:latest .
```

### Option 2: Using Docker Compose

```bash
cd /path/to/intent_comm_ros2_ws
docker-compose build
```

> **Note**: Building may take 10-30 minutes depending on your system and internet speed, as it downloads ROS 2 Humble base image and installs all dependencies.

## Running the Docker Container

### Option 1: Interactive shell (recommended for development)

```bash
docker run -it --rm -v $(pwd)/rollouts:/ws/rollouts intent-comm-ros2:latest
```

### Option 2: Using Docker Compose

```bash
docker-compose run --rm intent_comm
```

### Option 3: Run a specific command with port exposure

```bash
docker run -it --rm \
    -p 1883:1883 \
    -v $(pwd)/rollouts:/ws/rollouts \
    intent-comm-ros2:latest
```

### Option 4: Expose ports for MQTT access from host

To allow external MQTT clients (e.g., a sensor node publishing poses):

```bash
# Expose MQTT port to host machine
docker run -it --rm \
    -p 1883:1883 \
    -v $(pwd)/rollouts:/ws/rollouts \
    intent-comm-ros2:latest
```

Then from your host (or another machine on the network):
```bash
# Subscribe to MQTT topics from the container
mosquitto_sub -h localhost -p 1883 -t "robot/pose/human"

# Publish test data to the container
mosquitto_pub -h localhost -p 1883 -t "robot/pose/human" -m '{"x": 1.0, "y": 2.0, "qx": 0.0, "qy": 0.0, "qz": 0.707, "qw": 0.707}'
```

## Running with MQTT Broker on Host

**Recommended approach**: Run Mosquitto on your host machine, container connects to it via exposed port.

### Step 1: Install and start Mosquitto on host

```bash
# On Ubuntu/Debian
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# Verify it's running
sudo systemctl status mosquitto
```

### Step 2: Build Docker image

```bash
docker build -t intent-comm-ros2:latest .
```

### Step 3: Run container with MQTT connection to host

```bash
# Linux - connect to Mosquitto on host
docker run -it --rm \
    -p 1883:1883 \
    -v $(pwd)/rollouts:/ws/rollouts \
    intent-comm-ros2:latest
```

The container will automatically connect to the host's Mosquitto broker at `localhost:1883`.

### Step 4: In another terminal, run the test MQTT publisher

```bash
python3 test_mqtt.py --host localhost --port 1883 --duration 15
```

The data flow:
```
Host: test_mqtt.py (publisher) → Mosquitto (broker)
                                     ↓
Docker: MQTT bridge (subscriber) → ROS 2 pipeline → Recorder
```

## Alternative: Running Mosquitto in a Separate Container

If you prefer a dedicated Mosquitto container:

```bash
# Start Mosquitto container
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto:latest

# Run your app container
docker run -it --rm \
    --link mosquitto:mosquitto \
    -p 1883:1883 \
    -v $(pwd)/rollouts:/ws/rollouts \
    intent-comm-ros2:latest \
    --mqtt_broker mosquitto
```

## GUI Support (Visualization)

If you need to view plots or GUIs from the container:

### Linux

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/rollouts:/ws/rollouts \
    intent-comm-ros2:latest
```

Then inside the container:

```bash
ros2 launch intent_comm_nav_ros navigation_demo_record.launch.py controller:=npace trial_id:=0
```

### macOS (using XQuartz)

1. Install XQuartz: `brew install xquartz`
2. Launch XQuartz and allow connections:
   ```bash
   defaults write org.xquartz.X11 nolisten_tcp 0
   open -a XQuartz
   ```
3. Get your IP:
   ```bash
   IP=$(ipconfig getifaddr en0)
   ```
4. Run container:
   ```bash
   docker run -it --rm \
       -e DISPLAY=$IP:0 \
       -v $(pwd)/rollouts:/ws/rollouts \
       intent-comm-ros2:latest
   ```

### Windows (using WSL2)

Use WSL 2 backend for Docker Desktop and follow Linux instructions.

## Useful Docker Commands

### List images
```bash
docker images | grep intent-comm
```

### Remove image
```bash
docker rmi intent-comm-ros2:latest
```

### View running containers
```bash
docker ps
```

### Stop container
```bash
docker stop <container_id>
```

### Access shell in running container
```bash
docker exec -it <container_id> bash
```

### Remove all stopped containers
```bash
docker container prune
```

## Publishing to Docker Hub (Optional)

To share your image on Docker Hub:

1. Create account on [hub.docker.com](https://hub.docker.com)
2. Tag image:
   ```bash
   docker tag intent-comm-ros2:latest <your_username>/intent-comm-ros2:latest
   ```
3. Login:
   ```bash
   docker login
   ```
4. Push:
   ```bash
   docker push <your_username>/intent-comm-ros2:latest
   ```

Others can then pull with:
```bash
docker pull <your_username>/intent-comm-ros2:latest
docker run -it --rm -v $(pwd)/rollouts:/ws/rollouts <your_username>/intent-comm-ros2:latest
```

## Troubleshooting

### Container exits immediately
- Run with `-it` flags for interactive mode
- Check image build logs: `docker build -t intent-comm-ros2:latest . 2>&1 | tail -50`

### Out of disk space
- Remove unused images: `docker image prune -a`
- Remove build cache: `docker builder prune`

### Slow performance
- Increase Docker memory limit (Docker Desktop → Preferences → Resources)
- Use `--cpus` and `-m` flags to limit resource usage

### Permission denied errors
- On Linux, add your user to docker group:
  ```bash
  sudo usermod -aG docker $USER
  newgrp docker
  ```

## Volume Mounts Explained

The Docker commands mount local directories into the container:

- `-v $(pwd)/rollouts:/ws/rollouts`: Output directory accessible on host machine
- `-v /tmp/.X11-unix:/tmp/.X11-unix:rw`: Linux X11 display socket for GUI apps

Remove volume mounts for data isolation or modify paths as needed.

## Next Steps

- See `README.md` for launch parameters and configuration
- Modify `Dockerfile` to add additional system packages
- Customize `docker-compose.yml` for your specific use case
