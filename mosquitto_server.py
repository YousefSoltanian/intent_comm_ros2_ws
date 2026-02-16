#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Single ROS1 (Python 2) <-> MQTT bridge.

1) Subscribes to two ROS1 pose topics (e.g., VRPN/OptiTrack outputs) and publishes JSON to MQTT:
   - MQTT topic: /mocap/robot/pose
   - MQTT topic: /mocap/human/pose

2) Subscribes to an MQTT control topic and republishes to ROS1 /cmd_vel as geometry_msgs/Twist
   at 3 Hz (or configured rate). Control payload must provide:
     - velocity (linear.x)
     - angular (angular.z)

Dependencies (Ubuntu 18.04 / ROS Melodic / Python 2):
  sudo apt-get update
  sudo apt-get install -y python-pip
  sudo pip install paho-mqtt

Environment variables (optional):
  # MQTT
  MQTT_HOST=127.0.0.1
  MQTT_PORT=1883
  MQTT_CLIENT_ID=ros1_mqtt_bridge
  MQTT_QOS=0
  MQTT_RETAIN=0

  # ROS pose topics
  ROS_ROBOT_POSE_TOPIC=/vrpn_client_node/Fetch/pose
  ROS_HUMAN_POSE_TOPIC=/vrpn_client_node/human/pose

  # MQTT pose topics
  MQTT_ROBOT_POSE_TOPIC=/mocap/robot/pose
  MQTT_HUMAN_POSE_TOPIC=/mocap/human/pose

  # Control (MQTT -> ROS)
  MQTT_CONTROL_TOPIC=/robot/control
  ROS_CMD_VEL_TOPIC=/cmd_vel
  CMD_PUB_HZ=3
  CMD_TIMEOUT_SEC=0.75   # if no command received recently, publish zeros

Control message formats accepted (JSON):
  {"velocity": 0.2, "angular": 0.1}
  {"linear": {"x": 0.2}, "angular": {"z": 0.1}}   # also accepted

Pose message published to MQTT (JSON):
  PoseStamped:
    {"t": <unix_sec>, "frame_id": "...", "position": {...}, "orientation": {...}}

Run:
  source /opt/ros/melodic/setup.bash
  roscore
  python ros1_mqtt_server.py
"""

from __future__ import print_function

import os
import sys
import json
import time
import signal
import threading
import math
import rospy
from geometry_msgs.msg import PoseStamped, Twist

try:
    import paho.mqtt.client as mqtt
except ImportError:
    sys.stderr.write("ERROR: missing paho-mqtt. Install with: sudo pip install paho-mqtt\n")
    raise


def _get_env(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _to_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v, default):
    try:
        return float(v)
    except Exception:
        return default
    
def saturate(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def _to_bool(v, default=False):
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _ros_time_to_float(stamp):
    try:
        return stamp.to_sec()
    except Exception:
        return time.time()


class RosMqttServer(object):
    def __init__(self):
        # MQTT config
        self.mqtt_host = _get_env("MQTT_HOST", "127.0.0.1")
        self.mqtt_port = _to_int(_get_env("MQTT_PORT", "1883"), 1883)
        self.mqtt_client_id = _get_env("MQTT_CLIENT_ID", "ros1_mqtt_bridge")
        self.mqtt_qos = _to_int(_get_env("MQTT_QOS", "0"), 0)
        self.mqtt_retain = _to_bool(_get_env("MQTT_RETAIN", "0"), False)

        # ROS pose topics (inputs)
        self.ros_robot_pose_topic = _get_env("ROS_ROBOT_POSE_TOPIC", "/vrpn_client_node/Fetch/pose")
        self.ros_human_pose_topic = _get_env("ROS_HUMAN_POSE_TOPIC", "/vrpn_client_node/Human_6/pose")

        # MQTT pose topics (outputs)
        self.mqtt_robot_pose_topic = _get_env("MQTT_ROBOT_POSE_TOPIC", "robot/pose/robot")
        self.mqtt_human_pose_topic = _get_env("MQTT_HUMAN_POSE_TOPIC", "robot/pose/human")

        # Control (MQTT -> ROS)
        self.mqtt_control_topic = _get_env("MQTT_CONTROL_TOPIC", "robot/cmd/robot")
        self.ros_cmd_vel_topic = _get_env("ROS_CMD_VEL_TOPIC", "/cmd_vel")
        self.cmd_pub_hz = 100.0
        self.cmd_timeout_sec = _to_float(_get_env("CMD_TIMEOUT_SEC", "0.75"), 0.75)

        # State for last received command
        self._lock = threading.Lock()
        self._last_cmd_time = 0.0
        self._last_velocity = 0.0
        self._last_angular = 0.0
        
        self.min_y = -2.0
        self.max_y = 3.2
        self.min_x = -1.0
        self.max_x = 1.0
        self.safety_stop = False

        # ROS pub/sub
        self._cmd_pub = rospy.Publisher(self.ros_cmd_vel_topic, Twist, queue_size=10)

        self._robot_sub = rospy.Subscriber(
            self.ros_robot_pose_topic, PoseStamped, self._robot_pose_cb, queue_size=10
        )
        self._human_sub = rospy.Subscriber(
            self.ros_human_pose_topic, PoseStamped, self._human_pose_cb, queue_size=10
        )

        # MQTT client
        self._mqtt = mqtt.Client(client_id=self.mqtt_client_id, clean_session=True)
        self._mqtt.on_connect = self._on_mqtt_connect
        self._mqtt.on_disconnect = self._on_mqtt_disconnect
        self._mqtt.on_message = self._on_mqtt_message

        # Optional auth / TLS
        user = os.environ.get("MQTT_USERNAME")
        pwd = os.environ.get("MQTT_PASSWORD")
        if user:
            self._mqtt.username_pw_set(user, pwd)

        if _to_bool(os.environ.get("MQTT_TLS"), False):
            self._mqtt.tls_set()

        rospy.loginfo("MQTT broker: %s:%d", self.mqtt_host, self.mqtt_port)
        rospy.loginfo("ROS robot pose: %s  -> MQTT: %s", self.ros_robot_pose_topic, self.mqtt_robot_pose_topic)
        rospy.loginfo("ROS human pose: %s  -> MQTT: %s", self.ros_human_pose_topic, self.mqtt_human_pose_topic)
        rospy.loginfo("MQTT control: %s -> ROS cmd_vel: %s (%.2f Hz, timeout %.2fs)",
                      self.mqtt_control_topic, self.ros_cmd_vel_topic, self.cmd_pub_hz, self.cmd_timeout_sec)

    # ---------------- MQTT callbacks ----------------
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            rospy.loginfo("MQTT connected; subscribing to %s", self.mqtt_control_topic)
            try:
                client.subscribe(self.mqtt_control_topic, qos=self.mqtt_qos)
            except Exception as e:
                rospy.logerr("MQTT subscribe failed: %s", str(e))
        else:
            rospy.logwarn("MQTT connect failed rc=%s", str(rc))

    def _on_mqtt_disconnect(self, client, userdata, rc):
        if rc == 0:
            rospy.loginfo("MQTT disconnected")
        else:
            rospy.logwarn("MQTT disconnected unexpectedly rc=%s", str(rc))

    def _on_mqtt_message(self, client, userdata, msg):
        if msg.topic != self.mqtt_control_topic:
            return
        if self.safety_stop == True:
            rospy.logfatal_once("Robot is outside of safe zone!!!")
            return
        try:
            payload = msg.payload
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", "replace")
            data = json.loads(payload)

            # Accept either {"velocity":..,"angular":..} or nested dicts
            vel = None
            ang = None
            if isinstance(data, dict):
                if "linear_x" in data:
                    vel = float(data.get("linear_x", 0.0))
                    print("Linear Velocity")
                    print(vel)
                    print("type of vel: ", type(vel))

                if "angular_z" in data:
                    ang = float(data.get("angular_z", 0.0))
                    print("ANgular Velocity")
                    print(ang)
                    print("type of ang: ", type(ang))
                    
            vel = saturate(vel, -1, 1)
            ang = saturate(ang, -1, 1)

            with self._lock:
                self._last_velocity = vel
                self._last_angular = ang
                self._last_cmd_time = time.time()

        except Exception as e:
            rospy.logwarn_throttle(2.0, "Bad control payload on %s: %s", self.mqtt_control_topic, str(e))

    # ---------------- ROS pose callbacks ----------------
    def _pose_to_json(self, msg):
        c = 1/math.sqrt(2)
        qz= c*(msg.pose.orientation.z - msg.pose.orientation.w)
        qw = c*(msg.pose.orientation.z + msg.pose.orientation.w)
        json_pose =  {
            "t": _ros_time_to_float(msg.header.stamp),
            "frame_id": msg.header.frame_id or "",
            "position": {
                "x": msg.pose.position.x,
                "y": msg.pose.position.y,
                "z": msg.pose.position.z,
            },
            # "orientation": {
            #     "x": 0,
            #     "y": 0,
            #     # "z": -msg.pose.orientation.z,
            #     # "w":msg.pose.orientation.w,
            #     "z": qz,
            #     "w": qw
            # },
            "orientation": {
                "x": msg.pose.orientation.x,
                "y": msg.pose.orientation.y,
                "z": msg.pose.orientation.z,
                "w": msg.pose.orientation.w
            },
        }
        
        return json_pose

    def _publish_mqtt_json(self, topic, obj):
        try:
            s = json.dumps(obj, separators=(",", ":"))
            self._mqtt.publish(topic, payload=s, qos=self.mqtt_qos, retain=self.mqtt_retain)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "MQTT publish failed topic=%s: %s", topic, str(e))

    def _robot_pose_cb(self, msg):
        # rospy.loginfo(msg)
        pose = self._pose_to_json(msg)
        x = pose["position"]["x"]
        y = pose["position"]["y"]
        
        if (y > self.max_y or y<self.min_y or x > self.max_x or x<self.min_x) and self.safety_stop != True :
            self.safety_stop = True
            print("Safety Lock Activated")
        
        self._publish_mqtt_json(self.mqtt_robot_pose_topic, pose)

    def _human_pose_cb(self, msg):
        self._publish_mqtt_json(self.mqtt_human_pose_topic, self._pose_to_json(msg))

    # ---------------- Control publisher loop ----------------
    def _cmd_timer_cb(self, _evt):
        now = time.time()
        with self._lock:
            age = now - self._last_cmd_time
            if self._last_cmd_time <= 0.0 or age > self.cmd_timeout_sec:
                v = 0.0
                w = 0.0
            else:
                v = self._last_velocity
                w = self._last_angular

        t = Twist()
        t.linear.x = v
        t.linear.y = 0.0
        t.linear.z = 0.0
        t.angular.x = 0.0
        t.angular.y = 0.0
        t.angular.z = w
        self._cmd_pub.publish(t)

    # ---------------- Lifecycle ----------------
    def start(self):
        # MQTT connection and background loop
        self._mqtt.connect(self.mqtt_host, self.mqtt_port, keepalive=30)
        self._mqtt.loop_start()

        # Publish /cmd_vel at desired rate
        period = 1.0 / self.cmd_pub_hz
        rospy.Timer(rospy.Duration(period), self._cmd_timer_cb)
        rospy.loginfo("Bridge started")

    def stop(self):
        try:
            self._mqtt.loop_stop(force=False)
        except Exception:
            pass
        try:
            self._mqtt.disconnect()
        except Exception:
            pass


def main():
    rospy.init_node("vrpn_mqtt_server", anonymous=False)

    server = RosMqttServer()

    def _handle_sig(*_args):
        server.stop()
        try:
            rospy.signal_shutdown("signal")
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    server.start()
    rospy.spin()
    server.stop()


if __name__ == "__main__":
    main()
