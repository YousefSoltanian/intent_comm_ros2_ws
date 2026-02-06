#!/usr/bin/env python3
"""
MQTT to ROS 2 Bridge Node (Bidirectional)

Subscribes to MQTT topics and publishes as ROS 2 PoseStamped messages.
Subscribes to ROS 2 command topics and publishes to MQTT with bounds checking.

Incoming MQTT JSON Payload Format (Poses):
{
    "x": float,
    "y": float,
    "z": float (optional, default 0.0),
    "qx": float,
    "qy": float,
    "qz": float,
    "qw": float,
    "timestamp_sec": int (optional, Unix timestamp, uses current time if absent)
}

Outgoing MQTT JSON Payload Format (Commands):
{
    "linear_x": float,
    "linear_y": float,
    "linear_z": float,
    "angular_x": float,
    "angular_y": float,
    "angular_z": float
}
"""

import json
import time
from typing import Optional

import paho.mqtt.client as mqtt
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
import traceback
import logging

class MQTTBridgeNode(Node):
    """Bidirectional MQTT-ROS 2 bridge with bounds checking."""

    def __init__(self):
        super().__init__("mqtt_bridge")

        # MQTT parameters
        self.mqtt_broker = str(self.declare_parameter("mqtt_broker", "localhost").value)
        self.mqtt_port = int(self.declare_parameter("mqtt_port", 1883).value)
        self.mqtt_client_id = str(self.declare_parameter("mqtt_client_id", "ros2_mqtt_bridge").value)
        self.mqtt_keepalive = int(self.declare_parameter("mqtt_keepalive", 60).value)

        # Topic mappings: MQTT topic -> ROS 2 topic
        self.mqtt_topic_human = str(
            self.declare_parameter("mqtt_topic_human", "robot/pose/human").value
        )
        self.mqtt_topic_robot = str(
            self.declare_parameter("mqtt_topic_robot", "robot/pose/robot").value
        )
        self.ros_topic_human = str(
            self.declare_parameter("ros_topic_human", "/mocap/human/pose").value
        )
        self.ros_topic_robot = str(
            self.declare_parameter("ros_topic_robot", "/mocap/robot/pose").value
        )

        # Command topic mappings for ROS 2 -> MQTT
        self.ros_topic_human_cmd = str(
            self.declare_parameter("ros_topic_human_cmd", "/human/cmd_vel").value
        )
        self.ros_topic_robot_cmd = str(
            self.declare_parameter("ros_topic_robot_cmd", "/robot/cmd_vel").value
        )
        self.mqtt_topic_human_cmd = str(
            self.declare_parameter("mqtt_topic_human_cmd", "robot/cmd/human").value
        )
        self.mqtt_topic_robot_cmd = str(
            self.declare_parameter("mqtt_topic_robot_cmd", "robot/cmd/robot").value
        )

        # Bounds parameters (x, y position limits)
        self.x_min = float(self.declare_parameter("x_min", -2.0).value)
        self.x_max = float(self.declare_parameter("x_max", 2.0).value)
        self.y_min = float(self.declare_parameter("y_min", -1.0).value)
        self.y_max = float(self.declare_parameter("y_max", 1.0).value)

        # Reconnection parameters
        self.mqtt_reconnect_max_delay = int(
            self.declare_parameter("mqtt_reconnect_max_delay", 60).value
        )

        # ROS 2 publishers (poses from MQTT)
        self.pub_human = self.create_publisher(PoseStamped, self.ros_topic_human, qos_profile=10)
        self.pub_robot = self.create_publisher(PoseStamped, self.ros_topic_robot, qos_profile=10)

        # ROS 2 subscribers (commands to MQTT)
        self.sub_human_cmd = self.create_subscription(
            Twist, self.ros_topic_human_cmd, self._on_human_cmd, qos_profile=10
        )
        self.sub_robot_cmd = self.create_subscription(
            Twist, self.ros_topic_robot_cmd, self._on_robot_cmd, qos_profile=10
        )

        # State tracking for bounds checking
        self.human_pose = None  # PoseStamped
        self.robot_pose = None  # PoseStamped

        # MQTT client setup
        self.mqtt_client = mqtt.Client(client_id=self.mqtt_client_id, protocol=mqtt.MQTTv311)
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message
        paho_logger = logging.getLogger("paho.mqtt")
        self.mqtt_client.enable_logger(paho_logger)

        # Track connection state
        self._mqtt_connected = False
        self._reconnect_delay = 1  # Start with 1 second

        # Initial connection attempt
        self._connect_mqtt()

        # Periodic connection checker timer
        self.create_timer(5.0, self._check_mqtt_connection)

        self.get_logger().info(
            f"MQTT Bridge initialized:\n"
            f"  Broker: {self.mqtt_broker}:{self.mqtt_port}\n"
            f"  Bounds: x=[{self.x_min}, {self.x_max}], y=[{self.y_min}, {self.y_max}]\n"
            f"  MQTT -> ROS 2 mappings:\n"
            f"    {self.mqtt_topic_human} -> {self.ros_topic_human}\n"
            f"    {self.mqtt_topic_robot} -> {self.ros_topic_robot}\n"
            f"  ROS 2 -> MQTT mappings:\n"
            f"    {self.ros_topic_human_cmd} -> {self.mqtt_topic_human_cmd}\n"
            f"    {self.ros_topic_robot_cmd} -> {self.mqtt_topic_robot_cmd}"
        )

    def _connect_mqtt(self) -> None:
        """Attempt to connect to MQTT broker."""
        try:
            self.get_logger().info(
                f"Connecting to MQTT broker {self.mqtt_broker}:{self.mqtt_port}..."
            )
            self.mqtt_client.connect(
                self.mqtt_broker, self.mqtt_port, keepalive=self.mqtt_keepalive
            )
            self.mqtt_client.loop_start()
        except Exception as e:
            self.get_logger().warning(
                f"Failed to connect to MQTT broker: {traceback.format_exc()}. "
                f"Will retry in {self._reconnect_delay}s"
            )

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.get_logger().info("Connected to MQTT broker successfully")
            self._mqtt_connected = True
            self._reconnect_delay = 1  # Reset reconnect delay on successful connection

            # Subscribe to topics
            client.subscribe(self.mqtt_topic_human)
            client.subscribe(self.mqtt_topic_robot)
            self.get_logger().info(
                f"Subscribed to: {self.mqtt_topic_human}, {self.mqtt_topic_robot}"
            )
        else:
            self.get_logger().warning(f"MQTT connection failed with code {rc}")
            self._mqtt_connected = False

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self._mqtt_connected = False
        if rc != 0:
            self.get_logger().warning(f"Unexpected MQTT disconnection with code {rc}")
        else:
            self.get_logger().info("Disconnected from MQTT broker")

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            # self.get_logger().info(f"Received MQTT message on {msg.topic}: {payload}")

            # Determine target publisher based on topic
            if msg.topic == self.mqtt_topic_human:
                pub = self.pub_human
                agent_name = "human"
                state_var = "human_pose"
            elif msg.topic == self.mqtt_topic_robot:
                pub = self.pub_robot
                agent_name = "robot"
                state_var = "robot_pose"
            else:
                self.get_logger().warning(f"Received message from unexpected topic: {msg.topic}")
                return

            # Parse and publish
            pose_msg = self._parse_payload(payload, agent_name)
            if pose_msg is not None:
                pub.publish(pose_msg)
                # Store pose state for bounds checking
                setattr(self, state_var, pose_msg)

        except json.JSONDecodeError as e:
            self.get_logger().warning(
                f"Failed to parse JSON from {msg.topic}: {traceback.format_exc()}"
            )
        except Exception as e:
            self.get_logger().warning(f"Error processing MQTT message from {msg.topic}: {traceback.format_exc()}")

    def _parse_payload(self, payload: dict, agent_name: str) -> Optional[PoseStamped]:
        try:
            # Support nested format: payload["position"]["x"], payload["orientation"]["w"]
            if "position" in payload and isinstance(payload["position"], dict):
                pos = payload["position"]
                x = float(pos.get("x"))
                y = float(pos.get("y"))
                z = float(pos.get("z", 0.0))

            if "orientation" in payload and isinstance(payload["orientation"], dict):
                ori = payload["orientation"]
                qx = float(ori.get("x"))
                qy = float(ori.get("y"))
                qz = float(ori.get("z"))
                qw = float(ori.get("w"))
       

            # Timestamp: your publisher uses "t" as float unix seconds
            if "t" in payload:
                try:
                    t = float(payload.get("t"))
                    sec = int(t)
                    nsec = int((t - sec) * 1e9)
                    header_stamp = rclpy.time.Time(seconds=sec, nanoseconds=nsec).to_msg()
                except Exception:
                    self.get_logger().warning(
                        "Invalid 't' timestamp for %s, using current time. payload=%s" % (agent_name, str(payload))
                    )
                    header_stamp = self.get_clock().now().to_msg()
            elif "timestamp_sec" in payload:
                # Optional legacy field (int seconds)
                try:
                    timestamp_sec = int(payload.get("timestamp_sec"))
                    header_stamp = rclpy.time.Time(seconds=timestamp_sec).to_msg()
                except Exception:
                    header_stamp = self.get_clock().now().to_msg()
            else:
                header_stamp = self.get_clock().now().to_msg()

            pose_msg = PoseStamped()
            pose_msg.header.stamp = header_stamp
            pose_msg.header.frame_id = str(payload.get("frame_id", "map"))

            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.position.z = z

            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw

            return pose_msg

        except (TypeError, ValueError) as e:
            self.get_logger().warning(
                "Invalid data type in %s payload: %s. Payload: %s" % (agent_name, str(e), str(payload))
            )
            return None
        except Exception as e:
            self.get_logger().warning(
                "Error parsing %s payload: %s. Payload: %s" % (agent_name, str(e), str(payload))
            )
            return None

    def _on_human_cmd(self, msg: Twist) -> None:
        """Callback for human command velocity from ROS 2."""
        try:
            self._publish_cmd_to_mqtt(msg, "human")
        except Exception as e:
            self.get_logger().warning(f"Error processing human command: {traceback.format_exc()}")

    def _on_robot_cmd(self, msg: Twist) -> None:
        """Callback for robot command velocity from ROS 2."""
        try:
            self._publish_cmd_to_mqtt(msg, "robot")
        except Exception as e:
            self.get_logger().warning(f"Error processing robot command: {traceback.format_exc()}")

    def _publish_cmd_to_mqtt(self, cmd: Twist, agent_name: str) -> None:
        """
        Publish command velocity to MQTT with bounds checking.
        Only stops movement that would exceed bounds.
        """
        if not self._mqtt_connected:
            self.get_logger().warning(f"MQTT not connected, cannot publish {agent_name} command")
            return

        # Get current pose
        pose = self.human_pose if agent_name == "human" else self.robot_pose
        if pose is None:
            self.get_logger().warning(f"No pose received for {agent_name} yet, cannot apply bounds checking")
            return

        # Current position
        current_x = pose.pose.position.x
        current_y = pose.pose.position.y

        # Apply bounds checking: only stop movement that would exceed bounds
        linear_x = cmd.linear.x
        linear_y = cmd.linear.y
        linear_z = cmd.linear.z

        # Check X bounds
        if linear_x > 0 and current_x >= self.x_max:
            self.get_logger().warning(
                f"{agent_name}: X already at max bound ({current_x:.2f}), stopping positive X movement"
            )
            linear_x = 0.0
        elif linear_x < 0 and current_x <= self.x_min:
            self.get_logger().warning(
                f"{agent_name}: X already at min bound ({current_x:.2f}), stopping negative X movement"
            )
            linear_x = 0.0

        # Check Y bounds
        if linear_y > 0 and current_y >= self.y_max:
            self.get_logger().warning(
                f"{agent_name}: Y already at max bound ({current_y:.2f}), stopping positive Y movement"
            )
            linear_y = 0.0
        elif linear_y < 0 and current_y <= self.y_min:
            self.get_logger().warning(
                f"{agent_name}: Y already at min bound ({current_y:.2f}), stopping negative Y movement"
            )
            linear_y = 0.0

        # Create MQTT payload
        payload = {
            "linear_x": linear_x,
            "linear_y": linear_y,
            "linear_z": linear_z,
            "angular_x": cmd.angular.x,
            "angular_y": cmd.angular.y,
            "angular_z": cmd.angular.z,
        }

        # Select target MQTT topic
        mqtt_topic = self.mqtt_topic_human_cmd if agent_name == "human" else self.mqtt_topic_robot_cmd

        # Publish to MQTT
        try:
            payload_json = json.dumps(payload)
            self.mqtt_client.publish(mqtt_topic, payload_json, qos=1)
            self.get_logger().debug(f"Published {agent_name} command to {mqtt_topic}: {payload_json}")
        except Exception as e:
            self.get_logger().warning(f"Failed to publish {agent_name} command to MQTT: {traceback.format_exc()}")

    def _check_mqtt_connection(self) -> None:
        """Periodically check and attempt to reconnect to MQTT broker if disconnected."""
        if not self._mqtt_connected:
            self.get_logger().info(
                f"MQTT disconnected. Attempting reconnection (delay: {self._reconnect_delay}s)"
            )
            try:
                self.mqtt_client.reconnect()
            except Exception:
                self.get_logger().warning(f"Reconnection attempt failed: {traceback.format_exc()}")
                # Exponential backoff with max delay
                self._reconnect_delay = min(self._reconnect_delay * 2, self.mqtt_reconnect_max_delay)


def main(args=None):
    rclpy.init(args=args)
    node = MQTTBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
