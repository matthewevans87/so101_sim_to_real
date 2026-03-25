# CNN Backbone Pretraining — Model Overview

This directory contains the `PretrainCnn` multi-task model and its associated
loss functions, used to pretrain a visual feature extractor on simulator
telemetry before RL training begins.

---

## Motivation

The RL policy uses a trainable CNN (`TrainableCnnFeatureExtractor`) to encode
camera observations into a compact latent vector.  Training this CNN from
random initialisation alongside PPO is unstable: the policy gradient signal is
too sparse to learn useful visual representations quickly.

Supervised pretraining solves this by leveraging privileged state information
that is only available in simulation (e.g., ground-truth cube pose and grip
contact) to pre-shape the CNN weights before RL begins.  After pretraining,
only the backbone weights are transferred; the task heads are discarded.

---

## Architecture

```
image  (N, 3, H, W)   ← float32 RGB in [0, 1]
    │
    ▼
Conv trunk             ← 4 × (Conv2d + ReLU), stride-based downsampling
    channels:     [32, 64, 128, 128]
    kernel_sizes: [ 5,  4,   3,   3]
    strides:      [ 2,  2,   2,   1]
    │
    ▼
SpatialSoftmax         ← per-channel soft argmax → (x, y) keypoints
    output: (N, 2 × last_channel)
    │
    ▼
MLP projection         ← Linear(2×128, 256) + ReLU
    output: (N, 256)   ← shared latent z
    │
    ├──▶ GripPositionHead  MLP [256 → 128 → 1]   → grip_position_logit  (N, 1)
    ├──▶ OrientationHead   MLP [256 → 128 → 4]   → orientation_pred     (N, 4)
    └──▶ VisibilityHead    MLP [256 → 1]          → visibility_logit     (N, 1)
```

The conv trunk + SpatialSoftmax + MLP projection together constitute
`TrainableCnnFeatureExtractor` from `so101_utils`.  The three prediction heads
are MLP branches added solely for the pretraining objective.

> **Architecture compatibility**: `backbone.*` in `configs/cnn_pretrain.yaml`
> must exactly match `models.policy.cnn` in `skrl_ppo_cfg.yaml` so that the
> saved backbone weights load cleanly into the RL actor's `_cnn` module.

---

## Inputs

Each training sample is one observation frame drawn from curated telemetry:

| Field | Shape       | Type    | Description                                                     |
| ----- | ----------- | ------- | --------------------------------------------------------------- |
| `rgb` | `(3, H, W)` | float32 | Camera image, converted from uint8 and transposed to CHW format |

Optional normalisation (mean/std per channel) can be applied via
`image_normalization` in the config.  The default is `null` (no normalisation)
since the backbone is trained from scratch on simulation data, not fine-tuned
from an ImageNet-pretrained model.

---

## Targets (supervised labels)

All labels are extracted from the simulator's privileged state at collection
time and stored in the telemetry NPZ shards:

| Field                      | Shape  | Type    | Description                                                        |
| -------------------------- | ------ | ------- | ------------------------------------------------------------------ |
| `is_cube_in_grip_position` | `(1,)` | float32 | Binary: 1 if the cube is within the gripper's grip zone            |
| `cube_quat_gripzone_wxyz`  | `(4,)` | float32 | Unit quaternion of the cube relative to the grip zone (w, x, y, z) |
| `cube_in_camera_frame`     | `(1,)` | float32 | Binary: 1 if the cube is visible in the camera frame               |

---

## Loss Function

The total loss is a weighted sum of three per-head losses:

$$
\mathcal{L} = w_{\text{grip}} \cdot \mathcal{L}_{\text{grip}}
            + w_{\text{ori}}  \cdot \mathcal{L}_{\text{ori}}
            + w_{\text{vis}}  \cdot \mathcal{L}_{\text{vis}}
$$

| Head          | Loss                             | Details                                                                               |
| ------------- | -------------------------------- | ------------------------------------------------------------------------------------- |
| Grip position | Binary cross-entropy with logits | Target: `is_cube_in_grip_position` ∈ {0, 1}                                           |
| Orientation   | MSE on unit quaternion           | Predicted quaternion is L2-normalised before MSE; target is `cube_quat_gripzone_wxyz` |
| Visibility    | Binary cross-entropy with logits | Target: `cube_in_camera_frame` ∈ {0, 1}                                               |

Default weights are all `1.0`.  All losses are mean-reduced over the batch.

---

## Outputs and Checkpoints

After training, four files are written to `<output-dir>/checkpoints/`:

| File                | Contents                                           | Use                                                     |
| ------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| `best_backbone.pt`  | Backbone-only `state_dict` at best val metric      | **Pass to RL training via `--cnn-backbone-checkpoint`** |
| `best_model.pt`     | Full `PretrainCnn` `state_dict` at best val metric | Inspection / resumption                                 |
| `final_backbone.pt` | Backbone-only `state_dict` at the last epoch       | Fallback if early stopping is too aggressive            |
| `final_model.pt`    | Full `PretrainCnn` `state_dict` at the last epoch  | Inspection / resumption                                 |

`best_backbone.pt` is a valid `state_dict` for `TrainableCnnFeatureExtractor`
and loads directly with `module.load_state_dict(torch.load("best_backbone.pt"))`.

---

## Metrics

Metrics are logged per epoch to TensorBoard (`<output-dir>/tensorboard/`) and
recorded in `<output-dir>/report.json`.

### Tracked during training (per step, train split)

| TensorBoard key          | Description           |
| ------------------------ | --------------------- |
| `train/loss_total`       | Weighted total loss   |
| `train/loss_grip`        | Grip-position BCE     |
| `train/loss_orientation` | Orientation MSE       |
| `train/loss_visibility`  | Visibility BCE        |
| `train/lr`               | Current learning rate |

### Tracked per epoch (val split)

| Key                      | Description                                                          | Target             |
| ------------------------ | -------------------------------------------------------------------- | ------------------ |
| `val/loss_total`         | Weighted total loss                                                  | ↓ lower is better  |
| `val/loss_grip_position` | Grip-position BCE                                                    | ↓                  |
| `val/loss_orientation`   | Orientation MSE (normalised quat)                                    | ↓                  |
| `val/loss_visibility`    | Visibility BCE                                                       | ↓                  |
| `val/grip_position_acc`  | Binary accuracy of grip-position prediction (threshold at logit = 0) | ↑ higher is better |
| `val/visibility_acc`     | Binary accuracy of visibility prediction                             | ↑                  |
| `val/orientation_mse`    | MSE of normalised predicted quat vs. target                          | ↓                  |

The metric used to select the best checkpoint is configured via
`training.best_metric` (default: `loss_total`).

---

## Training Strategy

| Setting        | Default          | Notes                                                                               |
| -------------- | ---------------- | ----------------------------------------------------------------------------------- |
| Optimiser      | Adam             | `learning_rate=1e-3`, `weight_decay=1e-4`                                           |
| LR schedule    | Cosine annealing | `T_max = num_epochs`, `eta_min = 1e-5`                                              |
| Epochs         | 100              | Upper bound; early stopping typically triggers before this                          |
| Early stopping | patience = 20    | Stops if `best_metric` does not improve for 20 consecutive epochs                   |
| Batch size     | 256              |                                                                                     |
| Seed           | from config      | Set globally before any framework init; `torch.backends.cudnn.deterministic = True` |

All stochastic processes (dataset split, DataLoader shuffling, model init) are
seeded from `cfg.seed` in `configs/cnn_pretrain.yaml`, ensuring reproducibility.
DataLoader workers each receive a deterministic per-worker seed (`seed + worker_id`).

---

## Integration with RL Training

```
curate  →  train-cnn  →  train (RL)
                           └── --cnn-backbone-checkpoint <output-dir>/checkpoints/best_backbone.pt
```

The pretrained backbone is loaded into the RL actor's `_cnn` module at the
start of RL training.  From that point it is updated end-to-end by PPO
backpropagation — the pretraining serves only as a warm-start.

See `configs/trainable_cnn.yaml` for the RL environment configuration that
corresponds to this backbone and `scripts/run.sh` for the full pipeline
invocations.
