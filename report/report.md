# Vision-Based Sim-to-Real Manipulation on Consumer Hardware

### Learning wrist-camera-only RL policies for the SO-101 arm in simulation, with a full deployment pipeline to physical hardware

<!-- TODO: Confirm title/subtitle. Honest about scope: a deployment pipeline (built) without claiming validated transfer (depends on the real-robot run). -->

---

## TL;DR

- Trained a **wrist-camera-only** PPO policy to find and lift a cube on the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) 6-DOF arm, reaching **~88% episode success in simulation** under heavy domain randomization (lighting, camera pose, cube appearance, distractors, image augmentation).
- Bootstrapped a **task-specific CNN vision encoder** from synthetic telemetry via iterative self-distillation: it beat a frozen ImageNet ResNet18 baseline by **+4.0 pp** (87.1% vs 83.1%), with a second distillation generation adding a further +0.6 pp (87.7%).
- Ran a **reward-ablation study** showing a **minimal 2-term reward matches a hand-tuned 13-term reward within statistical noise** — and that several individual reward terms were *actively counterproductive* (removing the dense absolute approach reward was the single best configuration tested, at 89.7%).
- Built a **fully reproducible pipeline** (`train -> collect -> curate -> train-cnn`) with typed-YAML configs (no silent defaults), versioned per-run artifacts, and end-to-end provenance tracking.
- Built an **end-to-end deployment path** — OpenCV camera calibration, a self-contained export bundle, and on-hardware inference with no Isaac Lab dependency. <!-- TODO (real-robot run): if you get episodes on the physical arm, add a bullet with the honest result (N episodes, X% success, dominant failure phase). If not, leave as-is. -->

**Code:** [github.com/matthewevans87/so101_sim_to_real](https://github.com/matthewevans87/so101_sim_to_real) · Full data tables in the [Appendix](#appendix-full-results).

---

## Motivation

Inspired by recent work from [Skild AI](https://www.youtube.com/watch?v=JQAfxp-FB0I), [NVIDIA](https://www.youtube.com/watch?v=S4tvirlG8sQ), and [Tesla](https://www.youtube.com/watch?v=g6bOwQdCJrc), this project investigates what's achievable when learning vision-conditioned reinforcement-learning policies on **consumer hardware**, with the eventual goal of zero-shot transfer to a physical SO-101 robotic arm.

It is built around two questions:

1. **Can a policy trained purely from a wrist-mounted RGB camera in simulation learn to find and lift an object?**
2. **What kind of vision encoder works best for this task on consumer hardware** — a frozen, off-the-shelf ImageNet backbone (ResNet18), or a small task-specific CNN pretrained from synthetic telemetry?

---

## Problem Setup

**Task.** The SO-101 6-DOF arm, with a single [wrist-mounted camera](https://github.com/TheRobotStudio/SO-ARM100/blob/main/media/UVC_cam_mount_so101.jpg), must locate a small cube on its work surface and lift it to 10 cm within 10 seconds.

**Observations.** A 1024-D vision feature vector (from the chosen vision encoder) plus the 6-D joint position vector, normalized to `[0, 1]`. Privileged state (e.g. true cube pose) is hidden from the policy — it perceives the world only through the wrist camera.

**Actions.** 6-D joint position commands, normalized to `[0, 1]`.

**Episodes.** 10 s each; physics at 120 Hz, observations every 2 ticks -> 600 policy steps per episode.

**Reward.** A phased reward shapes the policy through three stages — *approach -> grasp -> lift* — with later-phase bonuses gated on earlier phases saturating, plus termination penalties for going out of reach or touching the table. Reward terms are declarative YAML entries, which is what made the ablation study below possible.

<!-- FIGURE 1: a wrist-camera view frame (cube visible), or a simple annotated diagram of arm + cube + camera + work surface. TODO: insert + caption. -->

**Training & evaluation protocol.** All training runs use 256 parallel environments. Reported metrics are measured in a separate post-training evaluation phase of **1,024 episodes across 64 parallel environments**, on the highest-success checkpoint (not necessarily the final one). Reward-ablation and vision-backbone studies are run to **50,000 training steps** and reported as the **mean over 5 RNG seeds**; the exploratory reward-contribution studies (below) use a reduced budget of a single seed to 15,000 steps, and are treated as directional rather than conclusive.

---

## Approach

### Vision encoders and the distillation loop

Two vision encoders are supported; both produce a 1024-D feature vector (via SpatialSoftmax) and are **frozen** during PPO training, so the policy optimizes only the MLP head on top of stable features:

| Encoder           | Description                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `frozen_resnet18` | ImageNet-pretrained ResNet18, truncated before final pooling -> SpatialSoftmax -> 1024-D. Used to **bootstrap** the pipeline. |
| `frozen_cnn`      | Small task-specific CNN (4 conv layers + MLP projection) pretrained from synthetic telemetry -> 1024-D.                       |

The task-specific CNN is **bootstrapped from a working ResNet18 policy** in a self-distillation loop:

1. **`train`** a baseline policy with the ResNet18 encoder.
2. **`collect`** synthetic `(image, telemetry)` samples by playing back the trained policy in sim, sampling uniformly across episode timesteps to avoid early-episode bias.
3. **`curate`** the dataset, rebalancing along two axes — episode timestep and grip-zone distance — so the dominant mid-approach "hover" frames don't drown out the rarer near-contact frames; then split by episode into train/val/test.
4. **`train-cnn`** a small CNN with five heads on a shared backbone latent: cube position (grip-zone frame), gripper-cube alignment, cube orientation (6-D rotation), cube height, and cube visibility.
5. Use the resulting checkpoint as the frozen vision encoder in a fresh PPO run — and optionally repeat the loop, distilling a second generation from the improved policy.

**Result.** The task-specific CNN substantially outperformed the off-the-shelf ResNet18, and a second distillation generation added a smaller further gain:

| Vision encoder                                        | Episode success | Delta       |
| ----------------------------------------------------- | --------------- | ----------- |
| `frozen_resnet18` (bootstrap)                         | 83.1%           | —           |
| `frozen_cnn` — Gen 1 (distilled from ResNet18 policy) | 87.1%           | **+4.0 pp** |
| `frozen_cnn` — Gen 2 (distilled from Gen-1 policy)    | 87.7%           | +0.6 pp     |

The result matches the intuition behind the experiment: ResNet18 was pretrained as an ImageNet classifier on natural images that are well out of domain for a synthetic wrist-camera view, so its features are only loosely task-relevant. A small CNN trained directly on in-domain synthetic telemetry learns features matched to the actual task — cube position, alignment, height, visibility — and the policy built on those features performs better. The diminishing Gen-1 -> Gen-2 gain (+0.6 pp) suggests the first distillation captures most of the available benefit. *(Full metrics, including lift rate, episode length, and time-to-milestone, are in [Appendix A.2](#a2-vision-backbone-studies).)*

### Reward design and the ablation finding

The baseline reward was hand-designed with 13 active terms across the approach/grasp/lift phases. Three campaigns probed how much of that complexity was actually doing work: leave-one-out ablations, single-reward contribution from a minimal base, and a confirmation run of the best minimal configuration. The result was clear, and somewhat humbling:

- **Reward minimality is nearly free.** A **minimal 2-term shaping reward** (`approach_phase_terminal` + `grasp_phase[absolute]`, plus the always-on lift/terminal/action terms) reached **86.9%**, within statistical noise of the full 13-term baseline at **87.7%** (-0.9 pp; the gap is not significant across 5 seeds).
- **Several terms were actively counterproductive.** In the leave-one-out study, *removing* the dense absolute approach reward produced the **single best configuration tested at 89.7%** (+1.9 pp over baseline), and removing the dense absolute grasp reward was second best (89.1%). The hand-tuned reward **over-specified** the task.
- **Curriculum terminals bootstrap learning; dense shaping alone does not.** Starting from the minimal base and adding back exactly one reward at a time, the single fire-once `approach_phase_terminal` reached ~69% success on its own — while every dense approach shaper (distance, alignment, progressive, absolute) produced **0%** within the reduced budget. The sparse phase-completion bonus provides the "bait" that dense rewards can then refine; the dense gradient alone fails to bootstrap exploration from a cold policy.
- **More rewards hurt.** Stacking dense + terminal grasp signals (73.7%) or piling on additional terms consistently underperformed the minimal effective pair, with pathological terminations (cube knocked out of range) rising as reward count grew.

The practical takeaway: across the whole leave-one-out campaign, every single-term removal landed in a tight **83-90%** band at full budget — the bottleneck for this task is no longer reward shaping but training budget and model capacity. *(Full tables, confidence intervals, and the complete baseline reward configuration are in [Appendix A.1](#a1-reward-ablation-studies) and [A.3-A.5](#a3-individual-reward-contribution).)*

<!-- FIGURE 2 (optional): a bar chart of the leave-one-out success band (83-90%), or a training curve of minimal vs full reward. TODO: insert if you make it. -->

### Domain randomization

To support eventual sim-to-real transfer, the environment randomizes per-episode and per-env: scene lighting; camera pose; per-frame image augmentation (Gaussian noise, brightness, contrast, motion blur, JPEG compression); cube color, size, and starting position; arm starting joint positions; and a configurable set of distractor objects of varied geometry, size, and color. The intent is to force the policy to rely on robust, transferable visual structure rather than overfitting to sim-specific appearance.

---

## Engineering and Reproducibility

Every design choice in this project was made in service of *reproducible* experimentation — the discipline that separates a deployable system from a one-off demo:

- **Typed YAML, no silent defaults.** Every configuration value (rewards, seeds, network shapes, augmentations) is set explicitly and validated against a typed dataclass hierarchy at startup; missing or unknown keys fail immediately, before any expensive compute.
- **Self-contained, timestamped artifacts.** Every run writes the fully resolved configs, the explicit seed (set on `torch`/`numpy`/`random` before framework init), checkpoints, evaluation metrics, and TensorBoard logs to its own directory.
- **Provenance tracking.** `run_manifest.json` and `cnn_checkpoint_provenance.json` link every checkpoint back (via SHA256) to the configs and source data that produced it.
- **One-command pipeline and sweep runner.** The full `train -> collect -> curate -> train-cnn` sequence runs from a single command, with each step's output auto-wired to the next; the sweep runner executes Cartesian-product ablations and emits comparison summaries automatically. The reward studies above were all produced by this sweep machinery.

---

## Sim-to-Real: Pipeline and Status

The pipeline is built end-to-end for transfer to the physical SO-101, with **no Isaac Lab dependency at deploy time**.

**Camera calibration.** Sim-to-real transfer requires the simulated wrist camera to match the physical one. Intrinsics (focal length, principal point, distortion) are measured with a standard OpenCV checkerboard calibration and converted to Isaac Sim's pinhole convention. Extrinsics (the camera's mount transform relative to the gripper) are recovered with an **interactive tuning tool** that mirrors live robot joint positions into a sim scene, renders the sim wrist camera, and blends it against a captured real frame in a real-time overlay until they converge.

<!-- FIGURE 3 (STRONG — use regardless of real-robot outcome): the sim-vs-real camera overlay from the calibration tool. Visually proves the calibration work; compelling even without a transfer result. TODO: insert + caption. -->

**Deployment.** An `export` step produces a self-contained bundle (policy weights, frozen CNN backbone, inference-time image-preprocessing spec, joint config, manifest + provenance). The `deploy` command consumes the bundle and runs on-hardware inference directly, with optional live OpenCV overlay and episode recording.

**Status.**

<!-- TODO — WRITE ONE, honestly: -->
<!-- OPTION A (default, no real-robot result yet): "The full deployment path — calibration, export, and on-hardware inference — is implemented and runs end-to-end. Validating zero-shot transfer on the physical arm is the current focus." State plainly, no apology. -->
<!-- OPTION B (if you get episodes on the real arm): "Deployed to the physical SO-101: [N] episodes, [X]% task success. Failures concentrated in the [phase] phase, attributable to [cause]." Include video/frames as Figure 4. -->

<!-- FIGURE 4 (Option B only): a frame or short clip of the real arm executing the task. -->

---

## Limitations and Observations

- **Joint CNN-policy training exhibited representation instability.** Training the vision encoder end-to-end within the PPO loop produced latent-representation drift (abrupt shifts in activation mean and standard deviation) and non-collapsing policy entropy. Decoupling the two stages — pretraining the CNN via supervised regression on synthetic telemetry, then freezing it as a fixed feature extractor — resolved this instability. This decoupled design is the basis for the frozen-encoder architecture used throughout and is a necessary precondition for the controlled backbone comparison in Section A.2.
- **Reward complexity beyond a minimal effective set was counterproductive.** The ablation results indicate that adding reward terms beyond a minimal two-term configuration did not improve episode success and in several cases increased pathological termination rates and reduced convergence speed. The full 13-term reward configuration was over-specified: it matched the minimal configuration within statistical noise at 50,000 steps while introducing redundant shaping that, when isolated, degraded performance.

---

## What's Next

<!-- TODO: 2-4 short forward-looking bullets. Candidates (pick what's true): validate/quantify zero-shot transfer on the physical arm; harder tasks (multi-object, occlusion, non-cube geometry); larger training budget now that reward shaping is shown not to be the bottleneck; additional vision encoders. Keep brief. -->

---

## Links & Credits

- **Code:** [github.com/matthewevans87/so101_sim_to_real](https://github.com/matthewevans87/so101_sim_to_real)
- **SO-101 hardware:** [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- This project traces its origins to a [class project](https://github.com/utd-fall-25-cs-6341-robotics/cs6341-robotics-project-direct) created with Kiran Hegde. 

---

# Appendix: Full Results

*All time metrics (Episode Length, Cube Bump, Time-to-lift, First Approach, First Grasp, First Lift, First Success) are in simulation steps. Bold values mark the best result within each column of a table.*

## A.1 Reward Ablation Studies

**Purpose.** Determine the impact of removing single rewards from the full baseline configuration.

**Setup.** Each study removes exactly one reward; each reported value is the mean over **5 RNG seeds**, run to **50,000 steps**.

**Findings.**
- Removing `approach_phase[absolute]` yielded the best success rate: **0.8967** (+1.9 pp over baseline).
- Removing `grasp_phase[absolute]` was second best: **0.8910** (+1.4 pp).
- Baseline: **0.8773**.
- A `Minimal` configuration with the majority of rewards disabled yielded only slightly degraded performance: **0.8686** (-0.9 pp), within noise of baseline.

**Note.** The grasp phase is arithmetically *gated* by the approach phase — the grasp-phase score is the product of the approach-phase score and a grasp-specific factor — so removing either produces similar effects.

| Study                                | Success    | Lift       | Drop   | Reward      | Ep. Length | Cube Bump | Time-to-lift | First Approach | First Grasp | First Lift  | First Success |
| ------------------------------------ | ---------- | ---------- | ------ | ----------- | ---------- | --------- | ------------ | -------------- | ----------- | ----------- | ------------- |
| Baseline*                            | 0.8773     | 0.9096     | 0.0    | 3165.60     | 103.71     | 7.093     | 73.73        | 14028.8        | 65945.6     | 121088      | 665856        |
| Minimal**                            | 0.8686     | 0.8891     | 0.0002 | 2426.29     | 102.77     | 6.761     | 71.46        | 13670.4        | 110284.8    | 175001.6    | 943206.4      |
| Ablate `approach-distance`           | 0.8592     | 0.9020     | 0.0006 | 3077.74     | 102.18     | 6.287     | 72.64        | 14387.2        | 75724.8     | 158464      | 676761.6      |
| Ablate `approach-alignment`          | 0.8734     | 0.9059     | 0.0    | 3169.21     | 102.35     | 6.584     | 72.97        | 14080          | **42649.6** | 146483.2    | 573184        |
| Ablate `approach-phase[progressive]` | 0.8824     | 0.9041     | 0.0004 | 3169.88     | 102.60     | 6.819     | 71.55        | 13465.6        | 57446.4     | 138240      | 659404.8      |
| Ablate `approach-phase[absolute]`    | **0.8967** | **0.9186** | 0.0    | 3134.01     | 99.39      | 6.895     | 71.29        | 13824          | 67788.8     | 111564.8    | 695142.4      |
| Ablate `approach-phase-terminal`     | 0.8719     | 0.9090     | 0.0    | 2751.39     | 113.07     | **5.694** | 78.60        | 14182.4        | 54937.6     | 108851.2    | 1151692.8     |
| Ablate `grasp-phase[progressive]`    | 0.8629     | 0.9035     | 0.0    | 3156.92     | 104.98     | 6.891     | 72.51        | 14233.6        | 71065.6     | 118528      | 618649.6      |
| Ablate `grasp-phase[absolute]`       | 0.8910     | 0.9135     | 0.0002 | 2981.50     | **95.25**  | 6.381     | **70.92**    | 12953.6        | 47718.4     | 107212.8    | **436889.6**  |
| Ablate `grasp-phase-terminal`        | 0.8732     | 0.9039     | 0.0012 | 2672.95     | 100.91     | 7.233     | 71.66        | 14592          | 62822.4     | 149555.2    | 728678.4      |
| Ablate `wrist-roll-pose`             | 0.8906     | 0.9066     | 0.0008 | **3201.14** | 104.57     | 6.518     | 71.61        | **12185.6**    | 45977.6     | 156979.2    | 532428.8      |
| Ablate `avoid-bumping-cube`          | 0.8885     | 0.9150     | 0.0004 | 3163.61     | 98.60      | 6.681     | 71.16        | 14796.8        | 58624       | **99174.4** | 542412.8      |
| Ablate `time-penalty`                | 0.8686     | 0.9010     | 0.0006 | 3176.74     | 103.69     | 7.017     | 71.29        | 13209.6        | 57190.4     | 102144      | 675328        |

> \* **Baseline** uses the reward configuration in [A.5](#a5-baseline-reward-configuration) and the Gen-2 CNN backbone.
> \*\* **Minimal** is the Baseline with these rewards ablated: `approach_distance`, `approach_alignment`, `approach_phase[progressive]`, `approach_phase[absolute]`, `grasp_phase[progressive]`, `grasp_phase_terminal`, `wrist_roll_pose`, `avoid_bumping_cube`, `safety_touch_table`, `time_penalty`; and these rewards un-gated: `lift_phase[progressive]`, `lift_phase[absolute]`.

## A.2 Vision Backbone Studies

**Setup.** Baseline reward configuration; only the vision backbone changes. Mean over **5 RNG seeds**, **50,000 steps**. All backbones are pretrained and frozen during PPO (no gradient updates through the encoder).

- **ResNet18** — non-domain-specific ImageNet classifier; used to bootstrap.
- **Gen1 CNN** — trained on synthetic data generated by playing back the ResNet18-backbone policy.
- **Gen2 CNN** — trained on synthetic data generated by playing back the Gen1-CNN-backbone policy.

**Findings.** ResNet18 performed worst, as expected for an out-of-domain classifier. Gen1 CNN was a significant leap (+4.0 pp). Gen2 CNN added a minor further gain (+0.6 pp).

| Backbone | Success    | Lift       | Drop    | Reward      | Ep. Length | Cube Bump | Time-to-lift | First Approach | First Grasp | First Lift   | First Success |
| -------- | ---------- | ---------- | ------- | ----------- | ---------- | --------- | ------------ | -------------- | ----------- | ------------ | ------------- |
| ResNet18 | 0.8314     | 0.8742     | 0.0010  | 3176.19     | 120.54     | 7.025     | 79.42        | 13926.4        | **53196.8** | 168908.8     | 872960        |
| Gen1 CNN | 0.8711     | 0.8967     | 0.0002  | 3128.43     | **100.02** | **6.688** | **72.25**    | 14336          | 54476.8     | 129280       | **583987.2**  |
| Gen2 CNN | **0.8773** | **0.9018** | **0.0** | **3194.86** | 105.09     | 6.906     | 74.47        | **11673.6**    | 85196.8     | **108953.6** | 750899.2      |

## A.3 Individual Reward Contribution

**Setup.** Start from the `Minimal` configuration and enable exactly **one** additional reward per study. Reduced budget: **single RNG seed**, **15,000 steps** — directional, not conclusive.

**Findings.** Over the reduced budget, 8 of 13 configurations never reach the success lift height. Two (`approach-distance`, `grasp-phase[progressive]`) achieve ~1%. Two (`grasp-phase[absolute]`, `grasp-phase-terminal`) reach 15-17%. Only `approach-phase-terminal` reaches near-baseline performance (69%) on its own.

| Study                             | Success    | Lift       | Drop       | Reward      | Ep. Length | Cube Bump | Time-to-lift | First Approach | First Grasp | First Lift | First Success |
| --------------------------------- | ---------- | ---------- | ---------- | ----------- | ---------- | --------- | ------------ | -------------- | ----------- | ---------- | ------------- |
| Minimal**                         | 0          | 0          | 0          | -56.70      | **37.26**  | 4.564     | —            | 15104          | 268544      | 378624     | —             |
| Add `approach-distance`           | 0.0010     | 0.0400     | **0.0010** | 766.36      | 370.19     | 52.02     | 319.32       | 14336          | 145152      | 226048     | 3489536       |
| Add `approach-alignment`          | 0          | 0.0010     | 0          | -47.66      | 47.95      | 6.804     | **38.0**     | 16640          | 237312      | 701440     | —             |
| Add `approach-phase[progressive]` | 0          | 0.0391     | **0.0010** | -41.04      | 71.51      | 12.57     | 81.03        | 16128          | 171520      | 431872     | 2350848       |
| Add `approach-phase[absolute]`    | 0          | 0.0078     | 0          | 386.43      | 472.20     | 20.68     | 187.5        | 19712          | 145152      | 336384     | 1099776       |
| Add `approach-phase-terminal`     | **0.6904** | **0.7588** | 0.0039     | **1897.87** | 145.28     | 10.54     | 97.20        | 19200          | 97792       | 51200      | 967936        |
| Add `grasp-phase[progressive]`    | 0.0010     | 0.0996     | 0          | 214.33      | 380.94     | 11.11     | 239.59       | 19712          | 222208      | 222464     | 1441792       |
| Add `grasp-phase[absolute]`       | 0.1514     | 0.2100     | 0          | 386.76      | 107.78     | 7.467     | 103.38       | 15104          | **52736**   | 761344     | 1348096       |
| Add `grasp-phase-terminal`        | 0.1699     | 0.2285     | 0.0020     | 541.54      | 74.62      | 6.106     | 77.29        | 14848          | 272896      | 743168     | 1480448       |
| Add `wrist-roll-pose`             | 0          | 0          | 0          | -46.52      | 61.99      | 10.38     | —            | **13824**      | 312832      | 664832     | —             |
| Add `avoid-bumping-cube`          | 0          | 0.0020     | 0.0020     | -56.03      | 37.34      | **4.509** | 77           | 14848          | 450048      | 1582336    | —             |
| Add `safety-touch-table`          | 0          | 0          | 0          | -53.92      | 46.33      | 6.492     | —            | 22784          | 485888      | 374016     | 2989312       |
| Add `time-penalty`                | 0          | 0          | 0          | -52.84      | 38.61      | 4.895     | —            | 21760          | 433408      | 154880     | —             |

## A.4 Composite Reward Contribution

**Setup.** Start from `Minimal` and add **combinations** of rewards. Single RNG seed, 15,000 steps — directional.

**Findings.** Adding `approach-phase-terminal` + `grasp-phase-terminal` reaches 0.8340. Adding `grasp-phase[absolute]` + `approach-phase-terminal` is better still at 0.8594. Adding all three is slightly degraded (0.7373) — plausibly noise that more seeds would smooth, but consistent with the over-specification seen elsewhere. The combination of the dense `grasp-phase[absolute]` reward with the sparse `approach-phase-terminal` is sufficient to drive training to success on just 1/4 of the training budget.

| Study                                                                          | Success    | Lift       | Drop   | Reward  | Ep. Length | Cube Bump | Time-to-lift | First Approach | First Grasp | First Lift | First Success |
| ------------------------------------------------------------------------------ | ---------- | ---------- | ------ | ------- | ---------- | --------- | ------------ | -------------- | ----------- | ---------- | ------------- |
| Baseline*                                                                      | 0.6416     | 0.8711     | 0      | 3117.05 | 155.96     | 7.698     | 82.48        | **15104**      | **79360**   | 168192     | **638464**    |
| Add `approach-phase-terminal`                                                  | 0.0088     | 0.0566     | 0.0020 | 425.50  | 168.57     | 20.32     | 153.38       | 25088          | 145408      | 77056      | 1093120       |
| Add `approach-phase-terminal`, `grasp-phase-terminal`                          | 0.8340     | 0.8584     | 0.0010 | 2601.71 | **116.72** | 8.053     | **81.07**    | **15104**      | 157696      | 173824     | 1115136       |
| Add `grasp-phase[absolute]`, `approach-phase-terminal`                         | **0.8594** | **0.8838** | 0      | 2496.42 | 131.31     | 8.900     | 90.84        | 19200          | 136960      | **54016**  | 709120        |
| Add `grasp-phase[absolute]`, `approach-phase-terminal`, `grasp-phase-terminal` | 0.7373     | 0.8730     | 0.0029 | 2874.96 | 147.21     | **6.666** | 103.03       | 18176          | 185856      | 172032     | 901376        |

## A.5 Baseline Reward Configuration

The Baseline study used the Gen-2 CNN backbone. Its full reward configuration:

```yaml
rewards:
- type: approach_distance
  enabled: true
  scale: 20.0
  mode: unsigned_progressive
- type: approach_alignment
  enabled: true
  scale: 1.0
  mode: unsigned_progressive
- type: approach_gripper_pose
  enabled: false
  scale: 1.0
- type: approach_phase
  id: progressive
  enabled: true
  scale: 5.0
  mode: unsigned_progressive
- type: approach_phase
  id: absolute
  enabled: true
  scale: 1.0
  mode: absolute
- type: approach_phase_terminal
  enabled: true
  scale: 500.0
  fire_once: true
- type: grasp_phase
  id: progressive
  enabled: true
  scale: 10.0
  mode: unsigned_progressive
  gates:
  - metric: approach_phase
    gte: 0.5
  - metric: grip_zone_cube_distance
    lte: 0.04
- type: grasp_phase
  id: absolute
  enabled: true
  scale: 5.0
  mode: absolute
  gates:
  - metric: approach_phase
    gte: 0.5
  - metric: grip_zone_cube_distance
    lte: 0.04
- type: grasp_phase_terminal
  enabled: true
  scale: 500.0
  fire_once: true
- type: lift_phase
  id: progressive
  enabled: true
  scale: 1000.0
  mode: signed_progressive
  gates:
  - metric: approach_phase
    gte: 0.8
  - metric: grasp_phase
    gte: 0.7
- type: lift_phase
  id: absolute
  enabled: true
  scale: 5.0
  mode: absolute
  gates:
  - metric: approach_phase
    gte: 0.8
  - metric: grasp_phase
    gte: 0.7
- type: static
  id: success_terminal
  enabled: true
  scale: 1000.0
  fire_once: true
  gates:
  - metric: cube_lift_fraction
    gte: 1.0
- type: wrist_roll_pose
  enabled: true
  scale: 1.0
  target_rad: -1.5707963267948966
  pressure: 1.0
  mode: signed_progressive
- type: avoid_bumping_cube
  enabled: true
  scale: -0.1
  cube_widths: 1.0
- type: action
  enabled: true
  scale: -0.02
  joints:
  - shoulder_pan
  - shoulder_lift
  - elbow_flex
  - wrist_flex
  - wrist_roll
  - gripper
  gates:
  - metric: grasp_phase
    lt: 0.7
- type: safety_touch_table
  enabled: true
  scale: -1.0
- type: time_penalty
  enabled: true
  scale: -1.0
- type: cube_out_of_range_terminal
  enabled: true
  scale: 0.0
- type: safety_touch_table_terminal
  enabled: true
  scale: 0.0
terminations:
- id: success_lift_fraction_terminal
  enabled: true
  is_success: true
  gates:
  - metric: cube_lift_fraction
    gte: 1.0
- id: cube_out_of_range_terminal
  enabled: true
  is_success: false
  gates:
  - metric: is_cube_out_of_range
    gte: 0.5
- id: safety_touch_table_terminal
  enabled: true
  is_success: false
  gates:
  - metric: is_table_touched
    gte: 0.5
```

<!-- WRITING CHECKLIST (delete before publishing):
  [ ] Section "Sim-to-Real ... Status": chose Option A or B, written honestly
  [ ] Figure 1 (camera view) + Figure 3 (calibration overlay) inserted — priority
  [ ] Figure 2 (reward chart) + Figure 4 (real arm) — bonus if available
-->