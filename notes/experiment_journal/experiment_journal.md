# Experiment Notes

Experiments are stored in the base dir `/mnt/nas_1/matthew-evans/so101_sim_to_real/experiments/`

## March 19, 2026
Initial attempt at training a CNN in the PPO training loop.
Artifacts located at `2026-03-19_22-15-03/`.

### Results
- `feature_mean`
  - Measures mean and stdev of the latent vector of the CNN (i.e., the output vector of the MLP projection layer)
  - Sudden instability between 400k-500k steps
  - Indicates sudden, unstable shift in the latent space; features that were being learned were suddenly "thrown away", i.e., the representation was learning something useful, but then lost it. The mapping from input -> latent changed abruptly, so previously learned features became invalid. 
- `feature_std`
  - Explodes between 400k-500k steps
- `Policy / Standard deviation`
  - When compared with the ResNet18 approach, we see that the CNN model never gains confidence; it just keeps exploring at the initial level of entropy without settling. 

### Comparison of Frozen Resnet18 approach vs. trainable CNN under PPO
- Trained CNN never settles on stable features
  - why?
    - Big updates of entire network?
    - Not updating frequently enough?
  - Possible fixes
    - Reduce learning rate of just the CNN
    - Freeze the CNN till initial patterns are established?
- The ResNet18 feature extractor doesn't have domain specific features, but, the features it does extract are stable. 
- Non-specific stable features >> domain-specific unstable features. 

## March 20, 2025
Since training a CNN directly in the CNN loop didn't yield positive results, we will try training the CNN in the loop with a **Separate learning rate**.


### Experiment: Using Separate Learning Rates for CNN and Policy Network Layers
**Setup**
- Set LR of CNN to 1/10 of Policy net in. See artifact `2026-03-20_15-35-39`.
- ![Screenshot 2026-03-20 at 4.52.41 PM.png](./assets/Screenshot%202026-03-20%20at%204.52.41 PM.png)

**Results**
- Still seeing high entropy. Latent representation unstable. 


### Experiment: Freeze CNN weights for first 50k steps
**Setup**
- Set RL of CNN to 1/100 of Policy net in `2026-03-20_15-35-39`.
- CNN params all frozen for first 50k steps

**Results**
- It still doesn't seem to be learning any stable representations after 500k steps.


## March 21, 2026

### Experiment: Frozen Ablation Test
**Setup**
- Init PPO trained CNN with frozen random weights.
- Compare perf to semi-frozen and various LRs
**Hypothesis**
  - Its going to be more stable than the when the CNN is being trained, but, much less effective than the frozen ResNet approach. 
  - Controls for drift, but, essentially provides no vision features
**Results**
- `2026-03-21_08-47-55`
- Note, I only let it run for 100k steps. 
- Should re-run entire 1M steps for completeness. 
- Early results show Policy Std drops initially, then stabilizes. 
- Showing slow but steady increase in reward, though _much_ lower than resnet baseline. Need to rerun with same 1M steps.

## March 24, 2026

### Architectural Changes Idea
Use the ResNet18 (RN18) driven policy to boostrap the CNN training. The RN18 policy will drive the actor in sim, where we will capture labeled training examples. We then use those training examples to traint he CNN. Optionally, we then repeat this process iteratively, hopefully producing better and better vision encoders. 

```
-> Train policy using boostrap vision encoder (RN18)
-> Run policy to generate training examples 
-> Train Gen1 Vision Encoder (CNN1) on training examples

-> Train policy using CNN1
-> Run policy to generate training examples 
-> Train Vision Encoder (CNN2) on training examples

-> Train policy using Gen2 CNN2
-> Run policy to generate training examples 
-> Train Vision Encoder CNN3 on training examples
...
```

Train CNN with multiple shallow MLP heads that predict different things.

**Architecture**

```
image
  ↓
CNN (conv layers)
  ↓
Spatial Softmax
  ↓
Projection MLP   ← shared latent (this is the important part)
  ↓
  ├── Head A (position)
  ├── Head B (orientation)
  └── Head C (visibility)
```

### Generating Samples

- "Play" the sim using a model
- Record samples at different times in an episode uniformly so that we don't have a bias for the beginning of the run
  - Input
    - Raw (unmutated) image: `camera_data = self.camera.data.output["rgb"]`
  - Labels
    - Current Timestep: `self.episode_length_buf` (I think); not a feature, but a label so we can create unbiased dataset
    - Joint positions: `self.joint_pos[:, self._dof_idx]`
    - `is_cube_in_grip_position`
    - `q_gz = quat_unique(env.grip_zone_tf.data.target_quat_source[:, 0, :])` (the quat of the cube in the grip_zone's frame)
    - Use `tiled_camera.data.info['instance_segmentation_fast']` to determine if the cube is in frame
      - https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html



## March 25 2026
### Implementation Updates
- **Synthetic training example generator** captures images + sim telemetry as synthetic training examples to train the CNN vision encoder backbone
- **Training example curator** builds proper train/test/validate datasets, normalizing over timesteps

### Fixes needed
- Accidentally retrained RN18 with wrong camera resolution; need to retrain

### In Progress
- As base test (even with wrong camera res), currently running training with frozen pretrained CNN weights

If it looks promising, need to 
1. Retrain with RN18 backbone with corrected camera res
2. Regenerate CNN training examples, also with corrected camera res + distortions
3. Rerun fixed pretrained CNN strat

**Updates**
- There was a bug in the gating logic of the camera feed domain randomization. The bug has been fixed. the RN18 baseline is re-training.
- After 40k steps, looks like this might work. Now going to retrain RN18 correctly.
- Retraining of RN18 baseline policy underway.

## March 26, 2026
- RN18 baseline retraining complete. See `2026-03-25_15-33-39/`. Surprisingly, results not as good as previous run (with higher resolution camera and ungated image domain randomization), though, still workable as a baseline. 
- Refactored all scripts code into proper pipeline, runnable with `scripts/run.py`.
- Now have a full train Policy -> collect -> curate -> train CNN pipeline
- Moved all scripts into `scripts/` and `so101/` dirs
- Ran Collect, Curate, CNN Train based
  - Collected 100 episodes worth of samples
  - Input: `/mnt/nas_1/matthew-evans/so101_sim_to_real/experiments/2026-03-25_15-33-39/skrl/so101_lift_cube/checkpoints/best_agent.pt`
  - Output: `/home/matthew-evans/src/so101_sim_to_real/artifacts/pipeline_20260326_155634`


## March 27, 2026
- The pretrained "Frozen CNN" yielded surprisingly poor results.  `2026-03-26_18-08-27/`
- Seeing somewhat similar performance to the baseline (untrained) frozen CNN.
- Are we 100% sure that it is frozen? This result is really surprising.
- Ideas for improving the CNN's performance as a vision feature encoder:
  - Increase the number of samples `collect`ed
  - Increase the complexity of the CNN's architecture (e.g., more layers)
  - Make the collection more robust in some way (i.e., more randomized positions of the robot rather than training on a "playback" of the RN18 model's behaviors)
  - Give the CNN more/better/different MLP heads to optimize so as to train even better vision features

**Bug Identified and Fixed**
- BatchNorm was _not_ properly frozen on the CNN, even if the CNN weights were.
- Fix: Set `self._monitored_cnn.eval()` when CNN weights are frozen and `self._monitored_cnn.train()` if unfrozen. 


Things to consider
- Normalize images using same strategy as RN18's normalization; use normalized images in CNN and Policy train. 
Don't need this: RN18's images were _trained_ with the normalization, so it is a hard requirement when _using_ it. Our CNN wasn't, so no need. 
- Collect data with randomized arm starting positions
- Collect more data
- Is CNN being trained with same image pipeline as Policy training see?
- Shift CNN training distribution more towards starting positions since a fresh RL run will see a LOT of "early" images where the cube is on the ground. 
- Let's build a "truly frozen" vision feature extractor that is exactly like RN18, except it uses the trained CNN. Absolutely no backprop path. Just feeds straight into the RL network. 

**Considerations**
Our implementation of training the CNN layers with PPO backprop needs to be totally reworked. The previous strategy did silly things like slicing off features from the actor features to be used as observations for the critic rather than propery using the obsevations dict available. We need to read up on how to properly implement custom network architectures in SKRL. 

**Simplifaction**
In the mean time, I've completely removed the code paths for training CNN via PPO backprop and am instead pivoting to purely pre-trained CNN vision extractors as a drop-in replacement for the RN18 vision feature extractor. This will allow us to finally confirm if the frozen pretrained CNN extractor is a valid path. If it is, then we can reapproach fine tuning with PPO backprop. If not, we can abandon it. 

## March 28, 2026
- Large refactor.
  - Rip out ability to fine-tune CNN using PPO backprop. Our implementation was wrong, messy, and a distraction (for now). 
  - Greatly simplify using CNN as vision feature extractor, see `so101_utils/feature_extraction/feature_extraction.py`
- Retraining using (re)trained CNN. So far, results seem to be tracking with RN18 baseline.

**To Explore**
- Heatmap and keypoints to tensorboard during PPO training
- Heatmap and keypoints to tensorboard during CNN training
- Need way to sanity check that the PPO training loop is getting meaningful vision features
- Increase complexity of the CNN (add a layer)
- 10x the training samples
- Increase variety of training examples
- ⭐️ Add a termination condition: if the robot pushes the cube out of reach, terminate episode with high penalty.
- Idea: Train CNN on data feed from env; online training; Not from PPO backprop, but, from samples generated in real time from the env.

**To Fix**
- URDF doesn't include camera or mount, so camera sometimes clips through ground

## March 30, 2026
- Increased output dim of CNN
- Truncating MLP projection before spatial softmax to mirror RN18 approach
- Ran 10k episodes to gather large training set for CNN

## March 31, 2026
- Pretrained CNN
  - Working as vision feature extractor
  - Debug visuals show good features
  - Policy getting stuck
    - Pushes cube away rather than getting into proper grip position
- Implementing phased reward approach
  - Phase 1: Get into position
  - Phase 2: Grip the cube
  - Phase 3: Lift

### Phase 1:
**Observations**

```yaml
rewards:
  approach_phase:
    enabled: true
    scale: 5.0
  action:
    enabled: true
    scale: -0.001
```

*13k steps*
- With just `approach_phase` + `action`, the policy tends to reward hack (RH). If it finds the cube, great, but otherwise, it just camps out with minimal movement.
  - It might be necessary to return the distance to being negative
- `grip_zone_cube_distance` is *increasing*


*30k steps*
- We get good alignment and gripper position, but poor distance. Since

---

*Include distance*
```yaml
rewards:
  approach_phase:
    enabled: true
    scale: 5.0
  distance:
    enabled: true
    scale: -5.0
  action:
    enabled: true
    scale: -0.001
```

*rew_distance adjustment*
- Change `rew_distance` to return value [0, 1] * scale
- Maybe consider making this the case for all rewards

**Observations**
- Seeing immediate improvement with addition of modified `rew_distance`

## April 1, 2026
Previous policy model training complete. 

**Observations**
Good:
- Arm *does* go for cube when it sees it (though does not always see it)

Bad:
- Arm tends to rub the ground aggressively when moving around at first
- Gripper not open soon enough
- Fixed gripper seems to be target zone.
  - Enable debug visuals and record video to verify posisions
- Wristroll orientation causes gripper to push cube rather than straddling it.
- The fixed jaw hits the cube. We need a notion of getting out of the way. It needs to get into position while not hitting the cube.
  - Alignment helps, but only if it is with the right grip zone.
  - The policy needs to learn to identify the optimal grip zone based on the scale, position, and orientation of the cube.
  - The grip zone will be learned implicitly (maybe explicitly give to critic?)
  - We need to compute reward based on ideal grip zone.
    - How do we compute this?
  - Wait a second
    - The goal is *get the cube in the jaw*
    - How can we reward "in the jaw"?

## April 2, 2026

*Reasoning through the ideal grip position*
Trying to compute the width of the jaw isn't the right approach. That's not even what you're trying to achieve. You are trying to get the cube in the grasp zone. That the robot needs to open its jaws is implict. That the fixed jaw is hitting the cube is to be expected: there's not reason not to. We must give it a reason not to, then it will still want to get the cube into position, and will also want to move its gripper out of the way to do so, so as to not touch the cube. 
We might want to compute the ideal position away from the fixed gripper though.

By visual inspection, the `gripper`'s origin falls on the exact center dividing line between the two jaws. `Z=-0.1` is exactly between the two furthest grip pads on the jaws.
The ideal position is between these two pads. "between" depends on the width of the cube. From the `X=-0.0078` from the center line gets you to to the surface of the pad. 

Assume we want to have the surface of the cube "just above" the "tooth". 
The tooth is `a=-0.0078` from the origin. For a cube of length `c`, and buffer gap `g`, the centroid of the cube (assuming it is laying flat on the table, aligned with the gripper), its centroid should be at `a + c/2 + g`.

## April 3, 2026

Domain Randomization (DR) is applied to the cube. As such, the ideal gripping position changes (bigger cube -> further from gripper jaw tooth). We had to change the static grip_zone_offset value to be based on the DR-modified cube size. The DR happens per-env per-episode. This means we need the DR scale available for both the grip_zone_offset calc and then reuse it for the DR scaling of the cube. 
To achieve this, we introduce the concept of `EnvMetricPipeline` and `EnvMetricStep`. `EnvMetricStep`s are compute when a given environment resets in `_reset_idx`.

With these changes in place, we are now positioned to make further refinements to the phased rewards. 

*Update*

With the new EnvMetricPipeline, improved grip zone calculation, and **addition of a *avoid_bumping_cube*** penalty, the policy is learning to straddle the gripper around the cube within `< 50k` steps. 

*Considerations*
- Need to review the camera config settings and ensure we have proper simulated lens
- The policy tends to move around a lot unnecessarily even after it has placed the cube in the grip zone. How can we reduce this extra motion--ideally to zero? Low pass filter?
- We need to add a simulated camera mount rigid body so that the policy doesn't clip the camera through the floor
- The policy tends to touch the tip of the gripper to the cube rather than fully straddling it. I am going to move the grip zone back 1cm and see if this helps.


## April 4, 2026
The Robot Studio provides the stl file for the camera mount we are using: `assets/robots/SO-ARM101_camera_wrist_mount.stl`. We used Isaac Sim to generate a `.usd` file from this. We then add the camera mount usd to the robot's `assets/robots/so101_new_calib/so101_new_calib.usd`. We add an XForm to position the camera on the mount: `/so101_new_calib/gripper/mountscrew/camera_mount/CameraXframe`


Additionally we discovered that there already exists an Xform for the gripper's tooth: `/so101_new_calib/gripper/gripperframe`

*Distance Pressure*
We are working in cm but the world units are meters. This means that distances on the final approach of the gripper to the target position are `<1.0`. By adding a coefficient to the distance value, we can add pressure to push the policy into the final position. Using this technique, we see the policy find the cube within a mere 3000 training steps.

## April 5, 2026
Think about how to start work on the grip phase. 

*Hypothesis*
The `gripper_close_error` isn't needed for the approach phase, the model can learn to open the gripper so as to minimize distance without nudging.

If this holds, then we can easly apply an additional reward to encourage grasping once in position. 

*Results*
After 100k steps, getting similar performance.

## April 7, 2026
The approach phase reward now is accompanied by three component rewards, each with its own backing metric. 
`ApproachPhaseMetricStep` produces 
- `approach_distance`
- `approach_alignment`
- `approach_gripper_open`
- `approach_phase` (the product of the other `approach_*` metrics)

We then have the component rewards
- `ApproachDistanceRewardStep`
- `ApproachAlignmentRewardStep`
- `ApproachGripperOpenRewardStep`
and a terminal reward
- `ApproachPhaseTerminalRewardStep` when `approach_phase` > threshold

The three components are inverse exponential functions, `exp(-kx)` where k is a "pressure" exponential term driving behavior at the "last mile" of the policy's approach. 

*Hypothesis*
I'm going to try introducing a linear reward term to drive initial behaviors, while keeping the inverse exponential to provide sharp signal on the final approach. My hypothesis is that this will cause faster convergence. `Total Reward` will increase, but the test will be if `approach_phase` approaches `1.0` more quickly.


*Possbile Breakthrough: Actions Penalty*
I was incorrectly penalizing "actions". Rather than penalizing large movements, I was penalizing the actions based on their joint position value. Larger numerical positions resulted in larger penalties, which is obviously wrong. The correct behavior is to penalize large movements from the _previous_ position to the _next_.


## April 8, 2026
Ran 1M step training. The policy does a decent job of getting into position. It doesn't get into "perfect" position, but get's close, and hovers there. It's sufficient such that, if a grasp was triggered, it would indeed grab the cube. 

That said, there are two observed issues:
1. It tends to perform a lot of unnecessary movement, even when in a good approach position.
2. The gripper pose reward pushes the policy to focus less on "getting into a good position in which to grasp the cube" and more on "hold this specific pose". 


---

For the unnecessary movement, I'm increasing the rew_action penalty from `-0.001` to `-0.02`. 

For the gripper pose, I'm going to try simply disabling the gripper pose component of the approach phase rewards. The goal is to find find a good approach pose, which includes moving the gripper, so perhaps the gripper pose-specific component is unneeded.

---

*Regarding table touch safety*
Currently, even with a (small) penalty, the robot touches the work surface in sim. This is unacceptable in a physical robot.


---

We're still having a lot of trouble in the approach phase. I think the multiplicative `approach_phase` reward is crushing the learning signal. I'm going to disable it. Also, I'm going to implement a tier approach that unlocks different reward signals at different phases. Also, I want to add a reward for a better approach vector: the axis alignment when grasping the cube matters.
Also, I want to try moving away from a pure "distance" metric and try using a "improvement" metric: you only get more rewards if you next step moves you closer to the cube; no hovering to milk the rewards.

## April 11, 2026

We've changed a lot since last entry. Will have to look at the git logs. 

Now have 
- Gated rewards
- Multiple instances of a reward (e.g., at different gate conditions)
- Ability to set a terminal reward to not terminate
- Ability to set a terminal reward to fire once per episode


We're now working on trying to achieve grasp.

---

Suggested next consideration: Getting gripper to align with cube on axis.

General Align + Distance is good, but `_dbg_approach_dist_exp` is struggling.
Perhaps if we added axis alignment we could get the gripper into a better position. 

Also, noticing that, when nearly in position, the policy still moves around a lot. Perhaps we could gate a stronger action penalty based on distance.

## April 11, 2026 — Iterative CNN Bootstrap Plan

Analysis of the CNN training data revealed that 0% of samples reach `approach_phase >= 0.90` (the terminal threshold), because the policy that collected the data never achieved it — `max(approach_phase) = 0.8999` across all 358k samples. The frozen CNN therefore has no representational coverage of near-contact distances, so the RL policy cannot learn to close the final 5mm gap regardless of reward shaping. The fix is an iterative bootstrap loop: lower the threshold to a value with training data coverage, run RL until the policy reliably hits it, re-collect, re-train the CNN on richer near-contact data, then raise the threshold again.

### Checklist

- [x] Diagnose root cause: confirm 0% training data coverage at `approach_phase >= 0.90` (see run `2026-04-11_10-13-14`)
- [x] Lower `approach_phase_terminal.threshold` to `0.80` (30% training data coverage)
- [x] Lower `grasp_phase` gate from `approach_phase >= 0.95` to `>= 0.80` to match
- [x] Run RL training; confirm `_dbg_approach_dist_exp` now breaks past ~0.5 plateau
- [x] Re-run `collect` step using new RL policy checkpoint to gather near-contact data
- [x] Re-run `curate` + `train-cnn` on new dataset; verify `cube_pos_gz_mae` improves at short distances
- [ ] Raise `approach_phase_terminal.threshold` back toward `0.90` and repeat if needed 


*Side Note — `avoid_bumping_cube` threshold fix*
`cube_widths` was set to `0.25`, which is impossible to satisfy without being *inside* the cube (minimum physical distance from centroid is `cube_width × 0.5 = 0.5`). Raised to `1.0` (= half cube-width radius from the surface). After restarting, `_dbg_approach_dist_exp` improved significantly faster.

```yaml
  - type: avoid_bumping_cube
    enabled: true
    scale: -0.1
    cube_widths: 1.0  # was 0.25 — previously impossible without penetrating the cube
```

*Side Note — `grasp_phase_terminal` enabled*
Enabled `grasp_phase_terminal` mirroring `approach_phase_terminal` (`terminate: false`, `fire_once: true`) to provide a 500-pt bonus for reaching `grasp_phase >= 0.80` without resetting the episode, allowing the lift phase to continue in the same rollout.

```yaml
  - type: grasp_phase_terminal
    enabled: true
    scale: 500.0
    threshold: 0.80
    terminate: false
    fire_once: true
```

## April 12, 2026
Trained CNN Gen2 based on Gen1. Gen2 uses better MLP head features; hoping to see improved perception at final approach. Were no CNN training examples sub-centimeter in Gen1, so, hoping to see improvements with Gen2, since such samples will be in distribution.

## April 13, 2026

With the Gen2 CNN, we're seeing improved performance; see `2026-04-13_08-08-52`. 

- We need to see what is causing termination: timeout? cube out of range? etc.; it's certainly not (yet) lift. The `0` reward terminals need to be StepMetrics for logging purposes. 
- Asymptotic behavior of `_dbg_approach_dist_exp` has moved from 0.5 to 0.6 upper bound. Concrete evidence of improvement. At `pressure=20` this implies distances ~0.025 on a 0.03 width cube. 

Next things to try
- Reduce distance pressure; see if a smoother gradient does better; I don't see why it would; I'd think a sharp gradient would be a very strong signal.
- Include all termination events as metrics so we can see how episodes terminate in tensorboard
- Reduce steps -> 50k, checkpoint interval -> 5k

If reduced pressure shows improvement, do same for grasp phase. 

Continue from April 11, 2026
- [ ] Raise `approach_phase_terminal.threshold` back toward `0.90` and repeat if needed 


*Critical Observation*
The cube width is ~0.03. This means 0.03/2=0.015 is the lowest possible distance. Which means a distance of 0.025 is actually really close, It's only off by 1cm, and is perfectly suitable for grasping.

We need to rework some of the reward computations:
- Set `max_distance = 0.40`
- Set `min_distance = CUBE_WIDTH/2`


---

As expected, reducing the distance pressure yields inferior results. with a distance pressure of `5` yielding distance asymptote at ~0.06. 

Going to now fix the max/min distance described above and rerun. 

---

- Fixed max/min distance; updated metrics and rewards
- Enhanced _get_dones() to log the cause of episode termination
- Increased episode length back to 10 seconds; was seeing anecdotal visual behaviors that suggested a bit more time could result in better discovery.

*Observations*
- Seeing `grip_zone_cube_distance` drop more quickly that before! Which is good because we did **not** normalize this underlying value...
  - Wait: now distance is *increasing*, not good.
  - Changing pressure back to 20.

With the newly normalized distance metrics, we need to fix this:

```yaml
  - type: grasp_phase
    enabled: true
    scale: 10.0
    grip_force_pressure: 20.0
    grip_force_target: 2.2  # 4.448 =~1 lbf in Newtons
    mode: progressive
    gates:
      - metric: approach_phase
        gte: 0.80
      - metric: grip_zone_cube_distance
        lte: 0.005 # <-- impossible
```

changing to

```yaml
  - type: grasp_phase
    enabled: true
    scale: 10.0
    grip_force_pressure: 20.0
    grip_force_target: 2.2  # 4.448 =~1 lbf in Newtons
    mode: progressive
    gates:
      - metric: approach_phase
        gte: 0.50 #lowering threshold
      - metric: normalized_grip_zone_cube_distance
        lte: 0.013 #0.5/(40-1.5) == ~0.5cm from cube edge in normalized units
```


## April 17, 2026

After many changes (see git log), we've achieved consistent lifting behavior. 

There's a few manual things to check before we start a full ablation study.

*Observation:* The policy lifts to ~5cm when `lift_cube.height_threshold = 0.05`. Will it push to 10cm when we up this to `0.10`?


Test: change `lift_cube.height_threshold -> 0.10`, check results after same timestep length.

Result: No! Over the same number of timesteps, the policy achieves approx. the same lift height.

Hypothesis: Between the `static` `cube_height_w gte 0.05` reward and the `progressive` `lift_cube`, the policy gets stuck in a local minima. 


*Observation:* The policy now lifts and lowers the cube repeatedly. This is almost certainly an artifact of the progressive `lift_cube` reward. 

Test: Will using a single `absolute` `lift_cube` reward fix the cyclical lift behavior?


**Potential Refinement:** Change the behavior of `progressive` rewards such that they are only given on improvements over the "current best" value. This means the policy can't reward hack by going up and down. 
Another option is to make `progressive` rewards signed rather than absolute magnitude, which would give both a push (negative) and a pull (positive) reward. 

I think the signed progressive rewards might address the cyclical reward hacking (up -> down -> ...) behavior, enabling it to push past these barriers. 


*Ablation: Remove static cube_height_w rewards*
How does disabling these static "threshold rewards" effect training?
```yaml
  - type: static
    enabled: false
    scale: 500.0
    fire_once: true
    terminate: false
    gates:
      - metric: cube_height_w
        gte: 0.05 # 0.10, 0.15
```

**Result:**
The removal of the static rewards actually yielded significantly better performance.

## April 17, 2026 - Update

We now have a consistently working build that finds, approaches, grasps, and lifts the cube. The lift behavior lifts to the target height, but otherwise is somewhat random.

The stretch goal is now to grasp the cube and move it into a target position. This is arguably much more complex than the lift behavior–at least as complex as the approach phase.


*Physical Cube*
I created a 3D model of a cube similar to the dexterity cube which is printable with a 3D printer. The cube is black with white letters. The cube weighs 14.71 grams when printed with PLA and printed with default settings and when the letters are friction fitted (i.e. no glue). Once things stabilize, I will retrain the entire pipeline with the new cube model. 


## April 18, 2026
Todo
- [x] Goal Zone Task
  - [x] Fix issue with no reward for goal_zone_distance
  - [x] Commit changes
  - [x] Not going to pursue goal zone further for now; just focus on lift task.
- [x] Import new cube 3D model as USD
- Run training on new cube
  - [x] Achieves lifting despite not retraining CNN. CNN will likely improve performance. See run `2026-04-18_12-29-55/`
    - Retrain Steps:
      - [ ] Train RN18 policy
      - [ ] Pipeline: Collect w/ RN18 -> CNN1
      - [ ] Pipeline: Train Policy w/ CNN1 -> CNN2
- Lift Task
  - Re-train with latest changes (but with config set to lift goal only)
  - Commit changes
  - Merge to main
  - Post videos
- Report
  - Design Ablation studies
  - Configure Training pipeline and verify it is working
- Create metrics for these and add to Critic network:
  - [x] Add Cube's angular and linear velocity
  - [x] Computed opening width between the tips of the gripper
- Upgrade to latest version of Isaac Sim (?)


*Contemplations:*
The goal zone, as currently designed, as a known location. This means we're just trying to train the policy to do FK, which isn't novel or needed. There's no great learned capability here. The impressive bit is finding and picking the object. If you want to create a visual target for placement, that's fine, but that's a different task.