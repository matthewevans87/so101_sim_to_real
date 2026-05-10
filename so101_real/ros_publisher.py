"""ros_publisher.py — ROS2 joint state publisher for the deploy loop.

Publishes measured joint positions as ``sensor_msgs/JointState`` to
``/so101/joint_states`` at every control tick via a **subprocess** running
``/usr/bin/python3.12`` (which has the correct ABI for the system rclpy .so).

The lerobot conda env uses Python 3.10, which is ABI-incompatible with the
system ``rclpy`` C extension (built for 3.12).  This module works around that
by piping joint state data as newline-delimited JSON to a child process that
does the actual ROS2 publishing.

Gracefully degrades to a no-op if the subprocess fails to start.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Publisher script — runs in a /usr/bin/python3.12 subprocess
# ---------------------------------------------------------------------------

_PUBLISHER_SCRIPT = """\
import sys, json, os, select, threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

topic = sys.argv[1]
rclpy.init()
node = Node("so101_real_deploy")
pub = node.create_publisher(JointState, topic, 10)

# Read lines from stdin in a thread; spin rclpy in the main thread.
_queue = []
_lock = threading.Lock()
_shutdown = threading.Event()

def _stdin_reader():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            with _lock:
                _queue.append(data)
        except json.JSONDecodeError:
            pass
    _shutdown.set()

t = threading.Thread(target=_stdin_reader, daemon=True)
t.start()

executor = rclpy.executors.SingleThreadedExecutor()
executor.add_node(node)

while not _shutdown.is_set():
    with _lock:
        pending = _queue[:]
        _queue.clear()
    for data in pending:
        try:
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.name = data["names"]
            msg.position = data["positions"]
            msg.velocity = []
            msg.effort = []
            pub.publish(msg)
        except KeyError:
            pass
    try:
        executor.spin_once(timeout_sec=0.01)
    except Exception:
        break

node.destroy_node()
try:
    rclpy.shutdown()
except Exception:
    pass
"""

_PYTHON = "/usr/bin/python3.12"


class RosPublisher:
    """Publishes measured joint positions to a ROS2 ``sensor_msgs/JointState`` topic.

    Uses a ``/usr/bin/python3.12`` subprocess for rclpy compatibility — the
    lerobot conda env uses Python 3.10, which is ABI-incompatible with the
    system rclpy .so (built for 3.12).

    Gracefully degrades to a no-op if the subprocess cannot be started.
    """

    def __init__(
        self,
        joint_names: list[str],
        topic: str = "/so101/joint_states",
    ) -> None:
        self._joint_names = list(joint_names)
        self._topic = topic
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

        if not Path(_PYTHON).exists():
            print(
                f"[RosPublisher] WARNING: {_PYTHON!r} not found. "
                "Joint-state publishing is disabled."
            )
            return

        # Write publisher script to a temp file to avoid shell-quoting issues
        self._tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="ros_pub_"
        )
        self._tmpfile.write(_PUBLISHER_SCRIPT)
        self._tmpfile.flush()
        self._tmpfile.close()

        env = {
            "ROS_DISTRO": "jazzy",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "PYTHONPATH": "/opt/ros/jazzy/lib/python3.12/site-packages",
            "LD_LIBRARY_PATH": "/opt/ros/jazzy/lib",
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
        }

        try:
            self._proc = subprocess.Popen(
                [_PYTHON, self._tmpfile.name, topic],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
            )
            # Drain stderr in a daemon thread so the child never blocks
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop, daemon=True
            )
            self._stderr_thread.start()
            print(f"[RosPublisher] Publishing joint states -> {topic} "
                  f"(pid={self._proc.pid})")
        except Exception as exc:
            print(f"[RosPublisher] WARNING: failed to start publisher subprocess: {exc}")
            self._proc = None

    def _stderr_loop(self) -> None:
        for line in self._proc.stderr:
            line = line.rstrip()
            if line:
                print(f"[ros_pub] {line}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def publish(self, q_rad: torch.Tensor) -> None:
        """Send current joint positions to the publisher subprocess."""
        if not self.enabled:
            return
        payload = json.dumps({
            "names": self._joint_names,
            "positions": [float(q_rad[i]) for i in range(len(self._joint_names))],
        })
        with self._lock:
            try:
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def destroy(self) -> None:
        """Shut down the publisher subprocess."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if hasattr(self, "_tmpfile"):
            Path(self._tmpfile.name).unlink(missing_ok=True)
