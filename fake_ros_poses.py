#!/usr/bin/env python3
"""
fake_mqtt_poses.py

Publishes fake pose data to the MQTT topics expected by your MQTTBridgeNode:
  - robot/pose/human
  - robot/pose/robot

Payload matches your bridge parser (nested position/orientation + 't' timestamp):
{
  "position": {"x": float, "y": float, "z": float},
  "orientation": {"x": float, "y": float, "z": float, "w": float},
  "t": float,                 # unix seconds
  "frame_id": "map"           # optional
}

Usage examples:
  python3 fake_mqtt_poses.py
  python3 fake_mqtt_poses.py --host localhost --port 1883 --rate 60
  python3 fake_mqtt_poses.py --amp 0.2 --hx 1.2 --hy 0.6 --rx -1.2 --ry -0.6
  python3 fake_mqtt_poses.py --pattern line
  python3 fake_mqtt_poses.py --pattern circle --omega 0.4
"""

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass

import paho.mqtt.client as mqtt


def yaw_to_quat(yaw: float):
    """Planar yaw-only quaternion (x,y,z,w)."""
    h = 0.5 * yaw
    return 0.0, 0.0, math.sin(h), math.cos(h)


@dataclass
class AgentState:
    x0: float
    y0: float
    z: float
    yaw0: float


def make_pose_payload(x: float, y: float, z: float, yaw: float, frame_id: str):
    qx, qy, qz, qw = yaw_to_quat(yaw)
    return {
        "position": {"x": float(x), "y": float(y), "z": float(z)},
        "orientation": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
        "t": float(time.time()),
        "frame_id": frame_id,
    }


def connect_client(host: str, port: int, client_id: str, keepalive: int = 60):
    c = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)

    # Helpful callbacks for visibility
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"[fake_mqtt] connected to {host}:{port}")
        else:
            print(f"[fake_mqtt] connect failed rc={rc}", file=sys.stderr)

    def on_disconnect(client, userdata, rc):
        print(f"[fake_mqtt] disconnected rc={rc}", file=sys.stderr)

    c.on_connect = on_connect
    c.on_disconnect = on_disconnect

    c.connect(host, port, keepalive=keepalive)
    c.loop_start()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)

    ap.add_argument("--topic-human", default="robot/pose/human")
    ap.add_argument("--topic-robot", default="robot/pose/robot")

    ap.add_argument("--rate", type=float, default=30.0, help="publish rate (Hz)")
    ap.add_argument("--qos", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--frame-id", default="map")

    # Base positions (keep these inside your bounds if you want to avoid cmd limiting)
    ap.add_argument("--hx", type=float, default=1.2)
    ap.add_argument("--hy", type=float, default=0.6)
    ap.add_argument("--hz", type=float, default=0.0)
    ap.add_argument("--hyaw", type=float, default=0.0)

    ap.add_argument("--rx", type=float, default=-1.2)
    ap.add_argument("--ry", type=float, default=-0.6)
    ap.add_argument("--rz", type=float, default=0.0)
    ap.add_argument("--ryaw", type=float, default=math.pi)

    # Motion pattern
    ap.add_argument("--pattern", choices=["circle", "line", "static"], default="circle")
    ap.add_argument("--amp", type=float, default=0.15, help="position amplitude (m)")
    ap.add_argument("--omega", type=float, default=0.6, help="angular speed (rad/s)")
    ap.add_argument("--yaw-omega", type=float, default=0.4, help="yaw angular speed (rad/s)")

    args = ap.parse_args()

    human = AgentState(args.hx, args.hy, args.hz, args.hyaw)
    robot = AgentState(args.rx, args.ry, args.rz, args.ryaw)

    client = connect_client(args.host, args.port, client_id="fake_mqtt_poses")

    running = True

    def _sigint(_sig, _frm):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    dt = 1.0 / max(args.rate, 1e-6)
    t0 = time.time()

    print(
        "[fake_mqtt] publishing:\n"
        f"  human -> {args.topic_human}\n"
        f"  robot -> {args.topic_robot}\n"
        f"  rate={args.rate:.2f}Hz pattern={args.pattern} amp={args.amp} omega={args.omega}\n"
        "Ctrl-C to stop."
    )

    # phase offset so they don't overlap perfectly
    phase_r = math.pi

    while running:
        now = time.time()
        tau = now - t0

        if args.pattern == "static":
            hx, hy = human.x0, human.y0
            rx, ry = robot.x0, robot.y0

        elif args.pattern == "line":
            # simple back-and-forth along x, slight y wobble
            hx = human.x0 + args.amp * math.sin(args.omega * tau)
            hy = human.y0 + 0.5 * args.amp * math.sin(0.5 * args.omega * tau)

            rx = robot.x0 + args.amp * math.sin(args.omega * tau + phase_r)
            ry = robot.y0 + 0.5 * args.amp * math.sin(0.5 * args.omega * tau + phase_r)

        else:  # circle
            hx = human.x0 + args.amp * math.cos(args.omega * tau)
            hy = human.y0 + args.amp * math.sin(args.omega * tau)

            rx = robot.x0 + args.amp * math.cos(args.omega * tau + phase_r)
            ry = robot.y0 + args.amp * math.sin(args.omega * tau + phase_r)

        hyaw = human.yaw0 + args.yaw_omega * tau
        ryaw = robot.yaw0 - args.yaw_omega * tau

        human_payload = make_pose_payload(hx, hy, human.z, hyaw, args.frame_id)
        robot_payload = make_pose_payload(rx, ry, robot.z, ryaw, args.frame_id)

        client.publish(args.topic_human, json.dumps(human_payload), qos=args.qos)
        client.publish(args.topic_robot, json.dumps(robot_payload), qos=args.qos)

        time.sleep(dt)

    try:
        client.loop_stop()
    except Exception:
        pass
    try:
        client.disconnect()
    except Exception:
        pass

    print("[fake_mqtt] stopped")


if __name__ == "__main__":
    main()
