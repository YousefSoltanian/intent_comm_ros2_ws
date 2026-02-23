#!/usr/bin/env python3
"""
Test MQTT Publisher - Generates dummy robot and human trajectory data

This script simulates a robot and human moving in the environment and publishes
their pose data to MQTT topics that the ROS 2 app subscribes to.

Trajectories:
- Human: Circular motion around (0, 0)
- Robot: Circular motion around (0, 0) in opposite direction

JSON Payload Format:
{
    "x": float,
    "y": float,
    "z": 0.0,
    "qx": 0.0,
    "qy": 0.0,
    "qz": float (sin(yaw/2)),
    "qw": float (cos(yaw/2))
}
"""

import argparse
import json
import math
import time
from typing import Tuple

import paho.mqtt.client as mqtt


class TrajectoryGenerator:
    """Generate synthetic trajectories for testing."""

    def __init__(self, duration: float = 15.0):
        """
        Initialize trajectory generator.

        Args:
            duration: Total duration of trajectory in seconds
        """
        self.duration = duration
        self.start_time = time.time()

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time

    def circular_motion(
        self, center_x: float, center_y: float, radius: float, direction: int = 1
    ) -> Tuple[float, float, float]:
        """
        Generate circular motion trajectory.

        Args:
            center_x: Center of circle (x)
            center_y: Center of circle (y)
            radius: Radius of circular motion
            direction: 1 for counterclockwise, -1 for clockwise

        Returns:
            (x, y, yaw)
        """
        elapsed = self.get_elapsed_time()
        # Complete one full circle in 10 seconds
        angle = direction * (2 * math.pi * elapsed / 10.0)

        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        yaw = angle + (math.pi / 2 if direction == 1 else -math.pi / 2)

        return x, y, yaw

    def yaw_to_quaternion(self, yaw: float) -> Tuple[float, float, float, float]:
        """
        Convert yaw angle to quaternion (x, y, z, w).

        Args:
            yaw: Yaw angle in radians

        Returns:
            (qx, qy, qz, qw)
        """
        half_yaw = yaw / 2.0
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def get_human_pose(self) -> dict:
        """Get human pose as JSON payload."""
        # Human: counterclockwise circle with radius 3m, center at (-4, 0)
        x, y, yaw = self.circular_motion(center_x=-4.0, center_y=0.0, radius=2.5, direction=1)
        qx, qy, qz, qw = self.yaw_to_quaternion(yaw)

        return {
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "timestamp_sec": int(time.time()),
        }

    def get_robot_pose(self) -> dict:
        """Get robot pose as JSON payload."""
        # Robot: clockwise circle with radius 3m, center at (4, 0)
        x, y, yaw = self.circular_motion(center_x=4.0, center_y=0.0, radius=2.5, direction=-1)
        qx, qy, qz, qw = self.yaw_to_quaternion(yaw)

        return {
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "timestamp_sec": int(time.time()),
        }

    def is_finished(self) -> bool:
        """Check if trajectory duration has elapsed."""
        return self.get_elapsed_time() > self.duration


class MQTTPublisher:
    """Publishes dummy trajectory data to MQTT broker."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        publish_rate: float = 10.0,
        duration: float = 15.0,
    ):
        """
        Initialize MQTT publisher.

        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            publish_rate: Publishing frequency in Hz
            duration: Total duration to publish
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.publish_rate = publish_rate
        self.duration = duration

        # MQTT topics
        self.topic_human = "robot/pose/human"
        self.topic_robot = "robot/pose/robot"

        # MQTT client
        self.client = mqtt.Client(client_id="test_mqtt_publisher")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

        self.connected = False
        self.published_count = 0

        # Trajectory generator
        self.traj_gen = TrajectoryGenerator(duration=duration)

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            print(f"✓ Connected to MQTT broker {self.broker_host}:{self.broker_port}")
            self.connected = True
        else:
            print(f"✗ Connection failed with code {rc}")
            self.connected = False

    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        if rc != 0:
            print(f"✗ Unexpected disconnection with code {rc}")
        else:
            print("Disconnected from MQTT broker")
        self.connected = False

    def _on_publish(self, client, userdata, mid):
        """MQTT publish callback."""
        self.published_count += 1

    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            print(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            time.sleep(1)  # Give it time to connect
            return self.connected
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False

    def publish_data(self):
        """Publish dummy trajectory data at specified rate."""
        if not self.connected:
            print("✗ Not connected to MQTT broker. Aborting.")
            return

        print(f"\nPublishing trajectory data for {self.duration} seconds at {self.publish_rate} Hz...")
        print(f"Topics: {self.topic_human}, {self.topic_robot}\n")

        interval = 1.0 / self.publish_rate

        try:
            while not self.traj_gen.is_finished():
                elapsed = self.traj_gen.get_elapsed_time()

                # Get poses
                human_pose = self.traj_gen.get_human_pose()
                robot_pose = self.traj_gen.get_robot_pose()

                # Publish
                self.client.publish(self.topic_human, json.dumps(human_pose), qos=1)
                self.client.publish(self.topic_robot, json.dumps(robot_pose), qos=1)

                # Print status
                print(
                    f"[{elapsed:6.2f}s] Human: ({human_pose['x']:6.2f}, {human_pose['y']:6.2f}) | "
                    f"Robot: ({robot_pose['x']:6.2f}, {robot_pose['y']:6.2f})"
                )

                time.sleep(interval)

            print(f"\n✓ Finished publishing {self.published_count} messages")

        except KeyboardInterrupt:
            print("\n✗ Interrupted by user")
        finally:
            self.client.loop_stop()
            self.client.disconnect()

    def run(self):
        """Connect and publish data."""
        if self.connect():
            self.publish_data()
        else:
            print("✗ Failed to connect to MQTT broker")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and publish dummy trajectory data to MQTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (localhost, 15s duration, 10 Hz)
  python3 test_mqtt.py
  
  # Custom broker and duration
  python3 test_mqtt.py --host 192.168.1.100 --port 1883 --duration 30 --rate 20
  
  # Connect to Docker container
  python3 test_mqtt.py --host localhost --port 1883
        """,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="MQTT broker hostname (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=1883, help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Duration to publish data in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Publishing rate in Hz (default: 10.0)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MQTT Test Publisher - Dummy Trajectory Generator")
    print("=" * 70)
    print(f"Broker: {args.host}:{args.port}")
    print(f"Duration: {args.duration}s")
    print(f"Rate: {args.rate} Hz")
    print(f"Interval: {1/args.rate*1000:.1f} ms\n")

    publisher = MQTTPublisher(
        broker_host=args.host,
        broker_port=args.port,
        publish_rate=args.rate,
        duration=args.duration,
    )
    publisher.run()


if __name__ == "__main__":
    main()
