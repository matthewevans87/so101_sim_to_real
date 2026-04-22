# Copyright (c) 2026, so101_sim_to_real contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed manifest of a training run, written by train.py and consumed by evaluate.py.

The manifest is the *single source of truth* for the data-flow contract between
train and eval. It records the seed, the configs that were used, hashes of those
configs (so eval can detect drift), the checkpoint(s) produced, and the CNN
checkpoint with its provenance. evaluate.py loads this file rather than
auto-detecting files by filename convention.

Schema is versioned (``MANIFEST_VERSION``); load() rejects unknown versions so
that future changes are explicit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_VERSION = 1


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA256 digest of *path*.

    Reads in 1 MiB chunks to bound memory for large checkpoint files.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    """Return the current git HEAD commit, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


@dataclass
class RunManifest:
    """Typed manifest written at the end of a training run.

    Paths to files inside the experiment directory are stored *relative* to
    that directory so the experiment dir remains relocatable. Absolute paths
    are stored only for upstream-source provenance fields (e.g.
    ``cnn_checkpoint_source``).
    """

    manifest_version: int
    created_at: str
    git_commit: str | None
    task: str
    seed: int
    trainer_timesteps: int
    training_command: list[str]
    env_config_relpath: str
    env_config_sha256: str
    agent_config_relpath: str
    agent_config_sha256: str
    final_checkpoint_relpath: str
    final_checkpoint_sha256: str
    final_checkpoint_step: int | None
    cnn_checkpoint_relpath: str | None = None
    cnn_checkpoint_sha256: str | None = None
    cnn_checkpoint_source: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def build(
        cls,
        *,
        experiment_dir: Path,
        repo_root: Path,
        task: str,
        seed: int,
        trainer_timesteps: int,
        training_command: list[str],
        env_config_path: Path,
        agent_config_path: Path,
        final_checkpoint_path: Path,
        final_checkpoint_step: int | None,
        cnn_checkpoint_path: Path | None,
        cnn_checkpoint_source: Path | None,
        extras: dict[str, Any] | None = None,
    ) -> "RunManifest":
        """Hash inputs and assemble a manifest. Paths must already exist on disk."""
        for required, label in [
            (env_config_path, "env_config"),
            (agent_config_path, "agent_config"),
            (final_checkpoint_path, "final_checkpoint"),
        ]:
            if not required.is_file():
                raise FileNotFoundError(
                    f"RunManifest.build: required {label} file does not exist: {required}"
                )
        if cnn_checkpoint_path is not None and not cnn_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"RunManifest.build: cnn_checkpoint declared but not found: {cnn_checkpoint_path}"
            )

        return cls(
            manifest_version=MANIFEST_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=_git_commit(repo_root),
            task=task,
            seed=seed,
            trainer_timesteps=trainer_timesteps,
            training_command=list(training_command),
            env_config_relpath=str(env_config_path.resolve().relative_to(experiment_dir.resolve())),
            env_config_sha256=sha256_of_file(env_config_path),
            agent_config_relpath=str(agent_config_path.resolve().relative_to(experiment_dir.resolve())),
            agent_config_sha256=sha256_of_file(agent_config_path),
            final_checkpoint_relpath=str(final_checkpoint_path.resolve().relative_to(experiment_dir.resolve())),
            final_checkpoint_sha256=sha256_of_file(final_checkpoint_path),
            final_checkpoint_step=final_checkpoint_step,
            cnn_checkpoint_relpath=(
                str(cnn_checkpoint_path.resolve().relative_to(experiment_dir.resolve()))
                if cnn_checkpoint_path is not None
                else None
            ),
            cnn_checkpoint_sha256=(
                sha256_of_file(cnn_checkpoint_path) if cnn_checkpoint_path is not None else None
            ),
            cnn_checkpoint_source=(
                str(cnn_checkpoint_source.resolve()) if cnn_checkpoint_source is not None else None
            ),
            extras=dict(extras or {}),
        )

    # ── persistence ──────────────────────────────────────────────────────────
    def write(self, experiment_dir: Path) -> Path:
        """Write the manifest atomically to ``experiment_dir / MANIFEST_FILENAME``."""
        out_path = experiment_dir / MANIFEST_FILENAME
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        tmp_path.replace(out_path)
        return out_path

    @classmethod
    def load(cls, experiment_dir: Path) -> "RunManifest":
        """Load the manifest from ``experiment_dir``.

        Raises:
            FileNotFoundError: if the manifest does not exist.
            ValueError: if the manifest version is unknown.
        """
        path = experiment_dir / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"RunManifest not found at {path}. The training run that produced "
                f"this experiment dir predates the manifest contract or did not "
                f"complete successfully."
            )
        with open(path) as f:
            data = json.load(f)
        version = data.get("manifest_version")
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported manifest_version {version!r} at {path}. "
                f"This code expects version {MANIFEST_VERSION}."
            )
        # Filter to known fields so older saved manifests with extra fields
        # don't blow up; missing required fields will still raise TypeError.
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    # ── validation ───────────────────────────────────────────────────────────
    def verify_against_disk(self, experiment_dir: Path) -> None:
        """Raise ``ValueError`` if any tracked file is missing or its hash drifted.

        This is the integrity check evaluate.py runs before launching: it
        guarantees that the env_config and CNN checkpoint about to be used are
        bit-for-bit the ones recorded at training time.
        """
        checks: list[tuple[str, str | None, str | None]] = [
            ("env_config", self.env_config_relpath, self.env_config_sha256),
            ("agent_config", self.agent_config_relpath, self.agent_config_sha256),
            ("final_checkpoint", self.final_checkpoint_relpath, self.final_checkpoint_sha256),
            ("cnn_checkpoint", self.cnn_checkpoint_relpath, self.cnn_checkpoint_sha256),
        ]
        for label, relpath, expected_sha in checks:
            if relpath is None:
                continue
            full = (experiment_dir / relpath).resolve()
            if not full.is_file():
                raise ValueError(
                    f"Manifest references {label} at {full} but the file is missing. "
                    f"The experiment directory has been corrupted or partially deleted."
                )
            actual_sha = sha256_of_file(full)
            if expected_sha is None:
                raise ValueError(
                    f"Manifest is missing the recorded SHA256 for {label} but a file "
                    f"is present at {full}; cannot verify integrity."
                )
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Hash mismatch for {label} at {full}.\n"
                    f"  recorded: {expected_sha}\n"
                    f"  on disk:  {actual_sha}\n"
                    f"The file has been modified since training. Refusing to evaluate."
                )

    # ── absolute-path accessors (convenience) ────────────────────────────────
    def env_config_abs(self, experiment_dir: Path) -> Path:
        return (experiment_dir / self.env_config_relpath).resolve()

    def agent_config_abs(self, experiment_dir: Path) -> Path:
        return (experiment_dir / self.agent_config_relpath).resolve()

    def final_checkpoint_abs(self, experiment_dir: Path) -> Path:
        return (experiment_dir / self.final_checkpoint_relpath).resolve()

    def cnn_checkpoint_abs(self, experiment_dir: Path) -> Path | None:
        if self.cnn_checkpoint_relpath is None:
            return None
        return (experiment_dir / self.cnn_checkpoint_relpath).resolve()
