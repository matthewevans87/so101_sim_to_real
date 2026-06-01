"""recorder.py — Episode recorder for real-robot inference.

Writes NPZ shards in the same schema used by collect_telemetry.py so that
existing analysis and viz_cnn tooling works on real-robot rollouts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


@dataclass
class _FrameBuffer:
    frames: list = field(default_factory=list)
    joint_positions: list = field(default_factory=list)
    joint_targets: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    episode_ids: list = field(default_factory=list)


class EpisodeRecorder:
    """Buffer episode data and flush to NPZ shards, with optional per-episode MP4.

    Parameters
    ----------
    output_dir:
        Directory where NPZ shard files and video files are written.
    shard_size:
        Number of steps to accumulate before flushing a shard.
    bundle_hash:
        SHA-256 hash of the deploy bundle (from bundle_provenance.json) for
        provenance tracking in the run manifest.
    fps:
        Frames-per-second for the output MP4. Should match the control loop
        frequency (``bundle.control_hz``). If None, video is not recorded.
    frame_wh:
        ``(width, height)`` of camera frames. If None and fps is set, inferred
        from the first frame received in ``record_step()``.
    """

    def __init__(
        self,
        output_dir: Path,
        shard_size: int = 2048,
        bundle_hash: Optional[str] = None,
        fps: Optional[float] = None,
        frame_wh: Optional[tuple[int, int]] = None,
    ) -> None:
        self._output_dir = output_dir
        self._shard_size = shard_size
        self._bundle_hash = bundle_hash
        self._fps = fps
        self._frame_wh = frame_wh
        self._buf = _FrameBuffer()
        self._shard_index = 0
        self._episode_count = 0
        self._step_count = 0
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._video_episode_id: Optional[int] = None
        self._video_start_time: Optional[float] = None
        self._video_frame_count: int = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def record_step(
        self,
        frame_rgb: np.ndarray,
        joint_positions_rad: torch.Tensor,
        joint_targets_rad: torch.Tensor,
        actions_raw: torch.Tensor,
        episode_id: int,
    ) -> None:
        """Buffer one step of data.

        Parameters
        ----------
        frame_rgb:
            Camera frame ``(H, W, 3)`` uint8 RGB.
        joint_positions_rad:
            Measured joint positions ``(n_joints,)`` float32.
        joint_targets_rad:
            Commanded joint targets ``(n_joints,)`` float32.
        actions_raw:
            Policy output before action→joint mapping ``(act_dim,)`` float32.
        episode_id:
            Integer episode index.
        """
        self._buf.frames.append(frame_rgb)
        self._buf.joint_positions.append(joint_positions_rad.cpu().numpy())
        self._buf.joint_targets.append(joint_targets_rad.cpu().numpy())
        self._buf.actions.append(actions_raw.cpu().numpy())
        self._buf.timestamps.append(time.time())
        self._buf.episode_ids.append(episode_id)
        self._step_count += 1

        if self._fps is not None:
            if self._video_writer is None or self._video_episode_id != episode_id:
                self._start_video(episode_id, frame_rgb.shape)
            self._video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            self._video_frame_count += 1

        if self._step_count >= self._shard_size:
            self.flush()

    def _start_video(self, episode_id: int, frame_shape: tuple) -> None:
        """Open a new VideoWriter for the given episode."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if self._frame_wh is None:
            # Infer (width, height) from frame shape (H, W, C)
            h, w = frame_shape[:2]
            self._frame_wh = (w, h)

        video_path = str(self._output_dir / f"episode_{episode_id:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            video_path, fourcc, float(self._fps), self._frame_wh
        )
        if not self._video_writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter failed to open: {video_path}\n"
                f"  fps={self._fps}, frame_wh={self._frame_wh}\n"
                "Check that OpenCV was built with video codec support."
            )
        self._video_episode_id = episode_id
        self._video_start_time = time.monotonic()
        self._video_frame_count = 0

    def _release_video(self) -> None:
        """Release the VideoWriter and remux to actual fps if needed."""
        if self._video_writer is None:
            return

        elapsed = time.monotonic() - self._video_start_time
        if elapsed > 0 and self._video_frame_count > 0:
            actual_fps = self._video_frame_count / elapsed
        else:
            actual_fps = self._fps

        video_path = self._output_dir / f"episode_{self._video_episode_id:03d}.mp4"
        self._video_writer.release()
        self._video_writer = None
        self._video_start_time = None
        self._video_frame_count = 0

        if abs(actual_fps - self._fps) / self._fps > 0.05:
            self._remux_video_fps(
                video_path, declared_fps=self._fps, actual_fps=actual_fps
            )

    def _remux_video_fps(
        self, video_path: Path, declared_fps: float, actual_fps: float
    ) -> None:
        """Remux video to correct playback speed using ffmpeg."""
        if not shutil.which("ffmpeg"):
            print(
                f"[EpisodeRecorder] Warning: ffmpeg not found; "
                f"{video_path.name} fps is {declared_fps:.1f} but actual was "
                f"{actual_fps:.1f}. Install ffmpeg to auto-correct."
            )
            return

        pts_factor = declared_fps / actual_fps
        tmp_path = video_path.with_suffix(".remux.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"setpts={pts_factor:.6f}*PTS",
            "-r",
            f"{actual_fps:.3f}",
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            tmp_path.replace(video_path)
            print(
                f"[EpisodeRecorder] {video_path.name}: "
                f"corrected fps {declared_fps:.1f} → {actual_fps:.1f}"
            )
        else:
            print(
                f"[EpisodeRecorder] Warning: ffmpeg remux failed for {video_path.name}: "
                f"{result.stderr.decode()[:200]}"
            )
            if tmp_path.exists():
                tmp_path.unlink()

    def end_episode(self) -> None:
        """Signal that an episode boundary has been reached."""
        self._release_video()
        self._video_episode_id = None
        self._episode_count += 1

    def flush(self) -> None:
        """Write the current buffer to a shard file."""
        if not self._buf.frames:
            return

        shard_path = self._output_dir / f"shard_{self._shard_index:05d}.npz"
        np.savez_compressed(
            str(shard_path),
            frames=np.stack(self._buf.frames, axis=0),
            joint_positions=np.stack(self._buf.joint_positions, axis=0),
            joint_targets=np.stack(self._buf.joint_targets, axis=0),
            actions=np.stack(self._buf.actions, axis=0),
            timestamps=np.array(self._buf.timestamps, dtype=np.float64),
            episode_ids=np.array(self._buf.episode_ids, dtype=np.int32),
        )
        self._shard_index += 1
        self._step_count = 0
        self._buf = _FrameBuffer()

    def write_manifest(self, bundle_dir: Path, robot_config_path: Path) -> None:
        """Write a run manifest JSON alongside the shard files."""
        manifest = {
            "type": "real_robot_rollout",
            "bundle_dir": str(bundle_dir),
            "bundle_sha256": self._bundle_hash,
            "robot_config": str(robot_config_path),
            "total_episodes": self._episode_count,
            "shard_count": self._shard_index,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest_path = self._output_dir / "rollout_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def close(self) -> None:
        """Flush any remaining buffered data and close open video writer."""
        self._release_video()
        self.flush()
