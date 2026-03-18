#!/usr/bin/env python3
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\" />
  <title>Intent Recognition</title>
  <style>
    :root {
      --bg0: #101820;
      --bg1: #1f2a37;
      --btn: #c81d25;
      --btn-pressed: #7f1d1d;
      --txt: #f7f7f7;
    }
    html, body {
      height: 100%;
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      color: var(--txt);
      background: radial-gradient(circle at 15% 20%, #2f3b4a 0%, var(--bg1) 35%, var(--bg0) 100%);
    }
    .wrap {
      height: 100%;
      display: grid;
      place-items: center;
      padding: 20px;
      box-sizing: border-box;
    }
    .panel {
      width: min(520px, 100%);
      text-align: center;
    }
    h1 {
      margin: 0 0 14px;
      font-size: clamp(1.1rem, 3.6vw, 1.5rem);
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    p {
      margin: 0 0 24px;
      opacity: 0.9;
      font-size: clamp(0.95rem, 3.4vw, 1.05rem);
    }
    #btn {
      width: min(360px, 84vw);
      aspect-ratio: 1 / 1;
      border-radius: 999px;
      border: 6px solid #ffb4b4;
      background: var(--btn);
      color: #fff;
      font-weight: 800;
      font-size: clamp(1.1rem, 6.2vw, 1.8rem);
      letter-spacing: 0.03em;
      cursor: pointer;
      touch-action: manipulation;
      box-shadow: 0 18px 50px rgba(0,0,0,0.45), inset 0 -10px 20px rgba(0,0,0,0.18);
    }
    #btn:active {
      transform: scale(0.985);
    }
    #btn.locked {
      background: var(--btn-pressed);
      border-color: #fecaca;
      cursor: not-allowed;
      opacity: 0.88;
    }
    #status {
      margin-top: 18px;
      min-height: 1.4em;
      font-size: clamp(0.9rem, 3.4vw, 1rem);
      opacity: 0.95;
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"panel\">
      <h1>Press when you recognize robot intent</h1>
      <p>Tap once as soon as you are confident.</p>
      <button id=\"btn\" aria-label=\"Intent recognized\">INTENT RECOGNIZED</button>
      <div id=\"status\">Waiting for input...</div>
    </div>
  </div>

  <script>
    const btn = document.getElementById('btn');
    const statusEl = document.getElementById('status');
    let locked = false;

    async function sendPress() {
      if (locked) {
        return;
      }
      locked = true;
      btn.classList.add('locked');
      btn.disabled = true;
      statusEl.textContent = 'Recording...';

      try {
        const res = await fetch('/press', { method: 'POST' });
        if (!res.ok) {
          throw new Error('request failed');
        }
        const data = await res.json();
        if (data.accepted) {
          statusEl.textContent = 'Recorded. Thank you.';
        } else {
          statusEl.textContent = 'Already recorded for this run.';
        }
      } catch (_err) {
        locked = false;
        btn.classList.remove('locked');
        btn.disabled = false;
        statusEl.textContent = 'Network error. Please tap again.';
      }
    }

    btn.addEventListener('pointerdown', function (ev) {
      ev.preventDefault();
      sendPress();
    });
  </script>
</body>
</html>
"""


class _IntentHttpHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, node=None, **kwargs):
        self._node = node
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        # Silence default request log spam.
        return

    def _send_json(self, payload, code=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        node = self._node
        if node is None:
            self._send_json({"ok": False, "error": "server_not_ready"}, code=503)
            return

        if self.path == "/" or self.path.startswith("/?"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/status":
            self._send_json(
                {
                    "ok": True,
              "sent": bool(node.press_sent),
              "t_unix": float(node.press_sent_unix) if node.press_sent_unix is not None else None,
                }
            )
            return

        self._send_json({"ok": False, "error": "not_found"}, code=404)

    def do_POST(self):
        node = self._node
        if node is None:
            self._send_json({"ok": False, "error": "server_not_ready"}, code=503)
            return

        if self.path != "/press":
            self._send_json({"ok": False, "error": "not_found"}, code=404)
            return

        accepted = node.enqueue_press()
        self._send_json({"ok": True, "accepted": bool(accepted)})


class WebIntentButtonNode(Node):
    def __init__(self):
        super().__init__("web_intent_button")

        self.topic_intent_recognized = str(
            self.declare_parameter("topic_intent_recognized", "/hl/intent_recognized").value
        )
        self.host = str(self.declare_parameter("host", "0.0.0.0").value)
        port_value = self.declare_parameter("port", 8000).value
        self.port = int(port_value) if port_value is not None else 8000

        self.pub_intent = self.create_publisher(Bool, self.topic_intent_recognized, 10)

        self._press_event = threading.Event()
        self._lock = threading.Lock()
        self.press_sent = False
        self.press_sent_unix = None

        self.create_timer(0.02, self._flush_press)

        self._server = None
        self._server_thread = None
        self._start_http_server()

        self.get_logger().info(
            f"[WebButton] serving at http://{self.host}:{self.port} topic={self.topic_intent_recognized}"
        )

    def _make_handler(self):
        node = self

        class _Handler(_IntentHttpHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, node=node, **kwargs)

        return _Handler

    def _start_http_server(self):
        handler = self._make_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

    def enqueue_press(self) -> bool:
        with self._lock:
            if self.press_sent:
                return False
            self._press_event.set()
            return True

    def _flush_press(self):
        if not self._press_event.is_set():
            return

        with self._lock:
            if self.press_sent:
                self._press_event.clear()
                return

            self.pub_intent.publish(Bool(data=True))
            self.press_sent = True
            self.press_sent_unix = float(time.time())
            self._press_event.clear()
            self.get_logger().info(
                f"[WebButton] intent press published at unix={self.press_sent_unix:.3f}"
            )

    def destroy_node(self):
        try:
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebIntentButtonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
