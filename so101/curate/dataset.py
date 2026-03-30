"""PyTorch Dataset for curated CNN pretraining data.

Reads telemetry shards referenced in a curated manifest produced by
``curate.py``.  Each sample is a (image, targets) pair where:

- **image** — ``FloatTensor[C, H, W]`` in ``[0, 1]`` (optionally normalised).
- **targets** — ``dict[str, Tensor]`` with keys:

  - ``is_cube_in_grip_position`` — ``float32`` scalar, 0 or 1.
  - ``cube_quat_gripzone_wxyz``  — ``float32 (4,)`` unit quaternion.
  - ``cube_in_camera_frame``     — ``float32`` scalar, 0 or 1.

Shards are loaded lazily on first access and held in an LRU cache bounded
by ``max_cached_shards``.  When the limit is reached the least-recently-used
shard is evicted.  Pair with :class:`ShardSequentialSampler` so that
consecutive batches draw from the same shard — this keeps the number of
shards live across all DataLoader workers to
``O(num_workers × prefetch_factor)`` regardless of dataset size.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class TelemetryDataset(Dataset):
    """Dataset backed by curated NPZ telemetry shards.

    Args:
        manifest_path: Path to a ``*_manifest.json`` file produced by
            ``curate.py``.
        image_mean: Optional per-channel means for normalisation, shape ``(3,)``.
            If ``None``, images are returned in ``[0, 1]`` without further
            normalisation.
        image_std: Optional per-channel stds for normalisation, shape ``(3,)``.
            Must be provided if *image_mean* is provided.
        max_cached_shards: Maximum number of shards to hold in the LRU cache at
            once.  ``None`` disables eviction (unbounded).  When using
            :class:`ShardSequentialSampler` a value of
            ``2 * num_workers * prefetch_factor`` is sufficient (typically 16).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        max_cached_shards: Optional[int] = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self._manifest_path = manifest_path.resolve()

        with open(manifest_path) as f:
            self._manifest: dict = json.load(f)

        self._telemetry_dir = Path(self._manifest["telemetry_dir"])

        if (image_mean is None) != (image_std is None):
            raise ValueError(
                "image_mean and image_std must both be set or both be None."
            )
        if image_mean is not None:
            self._image_mean: torch.Tensor | None = torch.tensor(
                image_mean, dtype=torch.float32
            ).view(3, 1, 1)
            self._image_std: torch.Tensor | None = torch.tensor(
                image_std, dtype=torch.float32
            ).view(3, 1, 1)
        else:
            self._image_mean = None
            self._image_std = None

        # Flat index: list of (shard_path, row_in_shard)
        self._index: list[tuple[Path, int]] = []
        for shard_entry in self._manifest["shards"]:
            shard_path = self._resolve_shard_path(shard_entry["path"])
            for row in shard_entry["rows"]:
                self._index.append((shard_path, int(row)))

        # LRU shard cache: keyed by resolved shard path, ordered by recency.
        # Only the four label arrays plus the RGB array are stored; other
        # columns (joint_pos, global_step, …) are not read.
        # Eviction occurs in _load_shard when len > max_cached_shards.
        self._max_cached_shards = max_cached_shards
        self._shard_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_shard_path(self, shard_rel_path: str) -> Path:
        shard_path = Path(shard_rel_path)
        candidates: list[Path] = []

        if shard_path.is_absolute():
            candidates.append(shard_path)
        else:
            candidates.append(self._telemetry_dir / shard_path)

            # Support relocated curated datasets where the corresponding raw
            # collection sits in a sibling "collect/" directory.
            manifest_run_dir = self._manifest_path.parent.parent
            candidates.append(manifest_run_dir / "collect" / shard_path)

            # Also allow the shard path to be resolved relative to the manifest.
            candidates.append(self._manifest_path.parent / shard_path)

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists():
                return resolved

        raise FileNotFoundError(
            "Shard not found. Tried: "
            + ", ".join(str(candidate.resolve()) for candidate in candidates)
        )

    def _load_shard(self, shard_path: Path) -> dict[str, np.ndarray]:
        if shard_path in self._shard_cache:
            self._shard_cache.move_to_end(shard_path)  # mark as most-recently-used
            return self._shard_cache[shard_path]
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        data = np.load(shard_path, allow_pickle=False)
        self._shard_cache[shard_path] = {
            "rgb": data["rgb"],
            "is_cube_in_grip_position": data["is_cube_in_grip_position"],
            "cube_quat_gripzone_wxyz": data["cube_quat_gripzone_wxyz"],
            "cube_in_camera_frame": data["cube_in_camera_frame"],
        }
        if self._max_cached_shards is not None:
            while len(self._shard_cache) > self._max_cached_shards:
                self._shard_cache.popitem(last=False)  # evict least-recently-used
        return self._shard_cache[shard_path]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shard_path, row = self._index[idx]
        shard = self._load_shard(shard_path)

        # Image: (H, W, C) uint8 → (C, H, W) float32 in [0, 1]
        img_hwc = shard["rgb"][row]  # (H, W, C) uint8
        img = (
            torch.from_numpy(img_hwc.astype(np.float32)).permute(2, 0, 1).div_(255.0)
        )  # (C, H, W) float32 in [0, 1]

        if self._image_mean is not None:
            img = (img - self._image_mean) / self._image_std  # type: ignore[operator]

        targets: dict[str, torch.Tensor] = {
            "is_cube_in_grip_position": torch.tensor(
                float(shard["is_cube_in_grip_position"][row]),
                dtype=torch.float32,
            ),
            "cube_quat_gripzone_wxyz": torch.from_numpy(
                shard["cube_quat_gripzone_wxyz"][row].copy()
            ).float(),
            "cube_in_camera_frame": torch.tensor(
                float(shard["cube_in_camera_frame"][row]),
                dtype=torch.float32,
            ),
        }
        return img, targets


class ShardSequentialSampler(Sampler):
    """Yields sample indices grouped by shard, with both shard order and
    within-shard order shuffled each epoch.

    With this sampler, consecutive batches draw from the same shard.  Paired
    with :class:`TelemetryDataset`'s LRU cache this limits live shard memory
    to roughly ``num_workers × prefetch_factor`` shards at any one time,
    regardless of how many shards exist in the dataset.

    Compared to fully random shuffling, the i.i.d. property holds within each
    shard but not across shard boundaries.  In practice this is not a
    meaningful difference for SGD convergence.

    Args:
        dataset: The :class:`TelemetryDataset` to sample from.
        generator: Optional :class:`torch.Generator` for reproducibility.
    """

    def __init__(
        self,
        dataset: TelemetryDataset,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self._dataset = dataset
        self._generator = generator

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        # Group flat index positions by shard.
        shard_to_positions: dict[Path, list[int]] = {}
        for i, (shard_path, _) in enumerate(self._dataset._index):
            shard_to_positions.setdefault(shard_path, []).append(i)

        shards = list(shard_to_positions.keys())

        # Shuffle shard order.
        shard_perm = torch.randperm(len(shards), generator=self._generator).tolist()
        shards = [shards[i] for i in shard_perm]

        # Within each shard, shuffle sample positions and emit them.
        indices: list[int] = []
        for shard in shards:
            positions = shard_to_positions[shard]
            row_perm = torch.randperm(
                len(positions), generator=self._generator
            ).tolist()
            indices.extend(positions[p] for p in row_perm)

        return iter(indices)
