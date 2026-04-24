"""Atomic milestone event log for cross-run sample-efficiency comparison.

Writes a single ``milestones.json`` file to disk each time a training
milestone is reached for the first time.  Each entry records the exact
total number of environment transitions elapsed, making the values
comparable across runs with different ``num_envs`` or ``rollouts`` settings.

File format::

    {
      "first_approach": {"env_transitions": 51200},
      "first_grasp":    {"env_transitions": 204800},
      "first_lift":     {"env_transitions": 819200},
      "first_success":  {"env_transitions": 2457600}
    }

Keys are absent when a milestone was never reached before training ended.
Each ``record()`` call rewrites the file atomically so partial runs leave
a valid JSON file on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class MilestoneLog:
    """Records exact ``env_transitions`` counts at training milestones.

    Parameters
    ----------
    output_path:
        Absolute path to the output JSON file (e.g.
        ``/path/to/experiment/milestones.json``).  Parent directories are
        created if they do not exist.
    num_envs:
        Number of parallel environments.  Used to convert
        ``common_step_counter`` to total environment transitions via
        ``env_transitions = common_step_counter * num_envs``.
    """

    def __init__(self, output_path: str | Path, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be a positive integer; got {num_envs!r}.")
        self._output_path = Path(output_path)
        self._num_envs = num_envs
        self._records: dict[str, dict] = {}
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, name: str, common_step_counter: int) -> None:
        """Record a milestone.

        First-write-wins: if *name* has already been recorded, this call
        is a no-op.  On each new record the JSON file is rewritten
        atomically (write to a sibling ``.tmp`` file then
        :func:`os.replace`) so the file is never left in a truncated state
        even if the process is killed mid-write.

        Parameters
        ----------
        name:
            Milestone identifier, e.g. ``"first_lift"``.
        common_step_counter:
            Isaac Lab ``DirectRLEnv.common_step_counter`` value at the time
            the milestone fired.  Multiplied by ``num_envs`` to obtain
            ``env_transitions``.
        """
        if name in self._records:
            return

        env_transitions = common_step_counter * self._num_envs
        self._records[name] = {"env_transitions": env_transitions}
        self._write_atomic()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_atomic(self) -> None:
        """Rewrite the JSON file atomically via a sibling .tmp file."""
        tmp_path = self._output_path.with_suffix(".tmp")
        payload = json.dumps(self._records, indent=2)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, self._output_path)
