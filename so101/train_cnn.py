#!/usr/bin/env python3
"""CNN backbone supervised pretraining entrypoint.

Trains a :class:`PretrainCnn` model on curated telemetry data.  After
training, saves four checkpoints:

- ``checkpoints/best_model.pt``     — full multi-task model at best val metric.
- ``checkpoints/best_backbone.pt``  — backbone-only weights at best val metric.
  **This is the file passed to** ``train.py --cnn-backbone-checkpoint``.
- ``checkpoints/final_model.pt``    — full model after the last epoch.
- ``checkpoints/final_backbone.pt`` — backbone-only weights after the last epoch.

Also writes ``report.json`` (structured run record) and TensorBoard event
files under ``tensorboard/``.

Usage::

    python cnn_trainer/train.py \\
        --curated-dir artifacts/curated/2026-03-24 \\
        --output-dir  artifacts/cnn_pretrain/2026-03-24_<timestamp> \\
        --config      configs/cnn_pretrain.yaml

The --output-dir should be a fresh, timestamped directory per run so that
all configs and checkpoints are self-contained and reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Allow running as a script from the project root without installing the package.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from so101.curate.dataset import TelemetryDataset, ShardSequentialSampler
from so101.model.losses import compute_multitask_loss, compute_multitask_metrics
from so101.model.model import MultiTaskCnn


# ── Config loading and validation ─────────────────────────────────────────────


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _require(d: dict, *keys: str, section: str = "") -> None:
    prefix = f"{section}." if section else ""
    for k in keys:
        if k not in d or d[k] is None:
            raise ValueError(
                f"cfg.{prefix}{k} must be set explicitly; no default is allowed."
            )


def _validate_config(cfg: dict) -> None:
    _require(cfg, "seed")
    bb = cfg.get("backbone")
    if bb is None:
        raise ValueError("cfg.backbone section is required")
    _require(
        bb,
        "in_channels",
        "channels",
        "kernel_sizes",
        "strides",
        "mlp_hidden_dims",
        "output_dim",
        section="backbone",
    )
    heads = cfg.get("heads")
    if heads is None:
        raise ValueError("cfg.heads section is required")
    _require(heads, "grip_position", "orientation", "visibility", section="heads")

    loss = cfg.get("loss")
    if loss is None:
        raise ValueError("cfg.loss section is required")
    _require(
        loss,
        "weight_grip_position",
        "weight_orientation",
        "weight_visibility",
        section="loss",
    )

    train_cfg = cfg.get("training")
    if train_cfg is None:
        raise ValueError("cfg.training section is required")
    _require(
        train_cfg,
        "batch_size",
        "num_epochs",
        "num_workers",
        "pin_memory",
        "best_metric",
        section="training",
    )
    opt_cfg = train_cfg.get("optimizer")
    if opt_cfg is None:
        raise ValueError("cfg.training.optimizer section is required")
    _require(
        opt_cfg, "type", "learning_rate", "weight_decay", section="training.optimizer"
    )

    sched_cfg = train_cfg.get("lr_scheduler")
    if sched_cfg is None:
        raise ValueError("cfg.training.lr_scheduler section is required")
    _require(sched_cfg, "type", section="training.lr_scheduler")


# ── Optimizer and scheduler construction ─────────────────────────────────────


def _build_optimizer(model: MultiTaskCnn, opt_cfg: dict) -> optim.Optimizer:
    opt_type = opt_cfg["type"].lower()
    lr = float(opt_cfg["learning_rate"])
    wd = float(opt_cfg["weight_decay"])
    if opt_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_type == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    raise ValueError(
        f"Unsupported optimizer type: {opt_type!r}. "
        "training.optimizer.type must be one of: adam, adamw, sgd"
    )


def _build_scheduler(
    optimizer: optim.Optimizer,
    sched_cfg: dict,
    num_epochs: int,
) -> optim.lr_scheduler.LRScheduler | None:
    sched_type = sched_cfg["type"].lower()
    if sched_type == "cosine_annealing":
        T_max = int(sched_cfg.get("T_max_epochs", num_epochs))
        eta_min = float(sched_cfg.get("eta_min", 0.0))
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    if sched_type == "step":
        step_size = int(sched_cfg["step_size"])
        gamma = float(sched_cfg.get("gamma", 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched_type == "none":
        return None
    raise ValueError(
        f"Unsupported lr_scheduler type: {sched_type!r}. "
        "training.lr_scheduler.type must be one of: cosine_annealing, step, none"
    )


# ── Training / validation epoch ───────────────────────────────────────────────


def _run_epoch(
    model: MultiTaskCnn,
    loader: DataLoader,
    loss_cfg: dict,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    writer: SummaryWriter | None,
    global_step: int,
    split: str,
) -> tuple[dict[str, float], int]:
    """Run one full epoch over *loader*.

    Args:
        optimizer: If ``None``, runs in eval mode (no gradient computation).
        global_step: Current training step counter (incremented only during
            training, used for TensorBoard x-axis).

    Returns:
        ``(mean_metrics_dict, updated_global_step)``
    """
    is_train = optimizer is not None
    model.train(is_train)

    accum_losses: dict[str, float] = {
        "total": 0.0,
        "grip_position": 0.0,
        "orientation": 0.0,
        "visibility": 0.0,
    }
    accum_metrics: dict[str, float] = {
        "grip_position_acc": 0.0,
        "visibility_acc": 0.0,
        "orientation_mse": 0.0,
    }
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            predictions = model(images)
            losses = compute_multitask_loss(predictions, targets, loss_cfg)

            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

                if writer is not None:
                    writer.add_scalar(
                        "train/loss_total", losses["total"].item(), global_step
                    )
                    writer.add_scalar(
                        "train/loss_grip", losses["grip_position"].item(), global_step
                    )
                    writer.add_scalar(
                        "train/loss_orientation",
                        losses["orientation"].item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/loss_visibility",
                        losses["visibility"].item(),
                        global_step,
                    )
                global_step += 1

            for k in accum_losses:
                accum_losses[k] += losses[k].item()
            metrics = compute_multitask_metrics(predictions, targets)
            for k in accum_metrics:
                accum_metrics[k] += metrics[k]
            n_batches += 1

    denom = max(n_batches, 1)
    mean_losses = {f"loss_{k}": v / denom for k, v in accum_losses.items()}
    mean_metrics = {k: v / denom for k, v in accum_metrics.items()}
    return {**mean_losses, **mean_metrics}, global_step


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN backbone for visual feature pretraining."
    )
    parser.add_argument(
        "--curated-dir",
        required=True,
        help="Curated dataset directory produced by curate.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output directory for checkpoints, TensorBoard logs, and report. "
            "Should be a fresh timestamped directory per run."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="CNN pretrain config YAML (configs/cnn_pretrain.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device string (e.g. cuda:0, cpu).  "
        "Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    _validate_config(cfg)

    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    curated_dir = Path(args.curated_dir).resolve()
    if not curated_dir.exists():
        raise FileNotFoundError(f"Curated directory not found: {curated_dir}")

    output_dir = Path(args.output_dir).resolve()
    ckpt_dir = output_dir / "checkpoints"
    tb_dir = output_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot alongside outputs for reproducibility.
    shutil.copy2(args.config, output_dir / "config.yaml")

    # Device selection.
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[train-cnn] Device: {device}")

    train_cfg = cfg["training"]
    norm_cfg = cfg.get("image_normalization")
    image_mean = norm_cfg["mean"] if norm_cfg else None
    image_std = norm_cfg["std"] if norm_cfg else None

    # ── Datasets ──
    train_manifest = curated_dir / "train_manifest.json"
    val_manifest = curated_dir / "val_manifest.json"
    for p in (train_manifest, val_manifest):
        if not p.exists():
            raise FileNotFoundError(
                f"Manifest not found: {p}. " "Run 'curate' before 'train-cnn'."
            )

    max_cached_shards = train_cfg.get("max_cached_shards")  # None = unbounded
    if max_cached_shards is not None:
        max_cached_shards = int(max_cached_shards)

    train_ds = TelemetryDataset(
        train_manifest,
        image_mean=image_mean,
        image_std=image_std,
        max_cached_shards=max_cached_shards,
    )
    val_ds = TelemetryDataset(val_manifest, image_mean=image_mean, image_std=image_std)
    print(f"[train-cnn] Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}")

    # Reproducible DataLoader workers.
    def _seed_worker(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        sampler=ShardSequentialSampler(train_ds, generator=g),
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
        drop_last=False,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
    )

    # ── Model ──
    model = MultiTaskCnn(
        backbone_cfg=cfg["backbone"],
        heads_cfg=cfg["heads"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    print(
        f"[train-cnn] Parameters — total: {n_params:,}, "
        f"backbone: {n_backbone:,}, heads: {n_params - n_backbone:,}"
    )

    # ── Optimizer and scheduler ──
    optimizer = _build_optimizer(model, train_cfg["optimizer"])
    scheduler = _build_scheduler(
        optimizer,
        train_cfg.get("lr_scheduler", {"type": "none"}),
        int(train_cfg["num_epochs"]),
    )

    num_epochs = int(train_cfg["num_epochs"])
    best_metric_key = str(train_cfg["best_metric"])
    early_stopping_patience = train_cfg.get("early_stopping_patience")

    writer = SummaryWriter(log_dir=str(tb_dir))

    # ── Training loop ──
    best_val_metric: float = float("inf")
    patience_counter = 0
    global_step = 0
    last_epoch = 0
    started_at = datetime.now(timezone.utc)
    epoch_log: list[dict] = []

    print(f"[train-cnn] Starting training for {num_epochs} epochs …")
    for epoch in range(1, num_epochs + 1):
        last_epoch = epoch
        t0 = time.monotonic()

        train_metrics, global_step = _run_epoch(
            model,
            train_loader,
            cfg["loss"],
            optimizer,
            device,
            writer,
            global_step,
            "train",
        )
        val_metrics, _ = _run_epoch(
            model,
            val_loader,
            cfg["loss"],
            None,
            device,
            None,
            0,
            "val",
        )

        if scheduler is not None:
            scheduler.step()

        # Log val metrics to TensorBoard.
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.monotonic() - t0
        val_primary = val_metrics.get(
            best_metric_key, val_metrics.get("loss_total", float("inf"))
        )
        improved = val_primary < best_val_metric

        if improved:
            best_val_metric = val_primary
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        else:
            patience_counter += 1

        epoch_log.append(
            {
                "epoch": epoch,
                "elapsed_s": round(elapsed, 2),
                "train": {k: round(v, 6) for k, v in train_metrics.items()},
                "val": {k: round(v, 6) for k, v in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_metric": round(best_val_metric, 6),
                "improved": improved,
            }
        )

        marker = "*" if improved else " "
        print(
            f"[train-cnn] Epoch {epoch:4d}/{num_epochs} {marker} "
            f"val_loss={val_metrics.get('loss_total', 0):.4f}  "
            f"grip_acc={val_metrics.get('grip_position_acc', 0):.3f}  "
            f"vis_acc={val_metrics.get('visibility_acc', 0):.3f}  "
            f"ori_mse={val_metrics.get('orientation_mse', 0):.4f}  "
            f"{elapsed:.1f}s"
        )

        if early_stopping_patience is not None and patience_counter >= int(
            early_stopping_patience
        ):
            print(
                f"[train-cnn] Early stopping after {patience_counter} epochs "
                "without improvement."
            )
            break

    # ── Final checkpoints ──
    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
    writer.close()

    finished_at = datetime.now(timezone.utc)
    report = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "config": args.config,
        "curated_dir": str(curated_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": seed,
        "num_epochs_planned": num_epochs,
        "num_epochs_ran": last_epoch,
        "best_val_metric": best_val_metric,
        "best_metric_key": best_metric_key,
        "checkpoints": {
            "best_model": str(ckpt_dir / "best_model.pt"),
            "final_model": str(ckpt_dir / "final_model.pt"),
        },
        "epoch_log": epoch_log,
    }
    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[train-cnn] Training complete.")
    print(f"[train-cnn] Report:      {output_dir / 'report.json'}")
    print(f"[train-cnn] Best model:  {ckpt_dir / 'best_model.pt'}")
    print(f"[train-cnn] Final model: {ckpt_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
