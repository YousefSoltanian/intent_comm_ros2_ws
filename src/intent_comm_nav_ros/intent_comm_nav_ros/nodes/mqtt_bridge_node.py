#!/usr/bin/env python3
"""
MQTT to ROS 2 Bridge Node

Subscribes to MQTT topics and publishes as ROS 2 PoseStamped messages.
Expects JSON payloads with position, quaternion, and optional timestamp.

JSON Payload Format:
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
"""

import json
import time
from typing import Optional

import paho.mqtt.client as mqtt
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
import traceback
import logging

class MQTTBridgeNode(Node):
    """Bridges MQTT pose topics to ROS 2 PoseStamped publishers."""

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

        # Reconnection parameters
        self.mqtt_reconnect_max_delay = int(
            self.declare_parameter("mqtt_reconnect_max_delay", 60).value
        )

        # ROS 2 publishers
        self.pub_human = self.create_publisher(PoseStamped, self.ros_topic_human, qos_profile=10)
        self.pub_robot = self.create_publisher(PoseStamped, self.ros_topic_robot, qos_profile=10)

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
            f"  MQTT -> ROS 2 mappings:\n"
            f"    {self.mqtt_topic_human} -> {self.ros_topic_human}\n"
            f"    {self.mqtt_topic_robot} -> {self.ros_topic_robot}"
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
            self.get_logger().info(f"Received MQTT message on {msg.topic}: {payload}")

            # Determine target publisher based on topic
            if msg.topic == self.mqtt_topic_human:
                pub = self.pub_human
                agent_name = "human"
            elif msg.topic == self.mqtt_topic_robot:
                pub = self.pub_robot
                agent_name = "robot"
            else:
                self.get_logger().warning(f"Received message from unexpected topic: {msg.topic}")
                return

            # Parse and publish
            pose_msg = self._parse_payload(payload, agent_name)
            if pose_msg is not None:
                pub.publish(pose_msg)

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
            else:
                # Support flat format: payload["x"]
                x = float(payload.get("x"))
                y = float(payload.get("y"))
                z = float(payload.get("z", 0.0))

            if "orientation" in payload and isinstance(payload["orientation"], dict):
                ori = payload["orientation"]
                qx = float(ori.get("x"))
                qy = float(ori.get("y"))
                qz = float(ori.get("z"))
                qw = float(ori.get("w"))
            else:
                # Support flat format: payload["qx"], payload["qw"]
                qx = float(payload.get("qx"))
                qy = float(payload.get("qy"))
                qz = float(payload.get("qz"))
                qw = float(payload.get("qw"))

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
