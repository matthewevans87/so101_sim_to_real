
## Data
Time metrics (Episode Length, Cube Bump, Time-to-lift, First Approach, First Grasp, First Lift, First Success) are given in simulation steps. 

### Reward Ablation
Each ablation study removes one or more rewards. All metrics are the mean value over **five trials with different RNG seeds**, each run to **50,000 steps**.

| Study                                | Success Rate     | Lift Rate        | Drop Rate    | Reward          | Episode Length  | Cube Bump       | Time-to-lift    | First Approach | First Grasp | First Lift  | First Success |
| ------------------------------------ | ---------------- | ---------------- | ------------ | --------------- | --------------- | --------------- | --------------- | -------------- | ----------- | ----------- | ------------- |
| Baseline*                            | 0.87734375       | 0.9095703125     | 0.0          | 3165.600885     | 103.7085938     | 7.092722162     | 73.73176419     | 14028.8        | 65945.6     | 121088      | 665856        |
| Minimal**                            | 0.8685546875     | 0.8890625        | 0.0001953125 | 2426.288029     | 102.7714844     | 6.760525629     | 71.46077333     | 13670.4        | 110284.8    | 175001.6    | 943206.4      |
| Ablate `approach-distance`           | 0.8591796875     | 0.901953125      | 0.0005859375 | 3077.74492      | 102.1751953     | 6.287181458     | 72.64389923     | 14387.2        | 75724.8     | 158464      | 676761.6      |
| Ablate `approach-alignment`          | 0.8734375        | 0.905859375      | 0            | 3169.207137     | 102.3488281     | 6.583715125     | 72.96950591     | 14080          | **42649.6** | 146483.2    | 573184        |
| Ablate `approach-phase[progressive]` | 0.882421875      | 0.9041015625     | 0.000390625  | 3169.875398     | 102.5953125     | 6.818577693     | 71.55085063     | 13465.6        | 57446.4     | 138240      | 659404.8      |
| Ablate `approach-phase[absolute]`    | **0.8966796875** | **0.9185546875** | 0.0          | 3134.008666     | 99.38789063     | 6.895262796     | 71.29169416     | 13824          | 67788.8     | 111564.8    | 695142.4      |
| Ablate `approach-phase-terminal`     | 0.871875         | 0.908984375      | 0.0          | 2751.39023      | 113.0669922     | **5.693896322** | 78.60159762     | 14182.4        | 54937.6     | 108851.2    | 1151692.8     |
| Ablate `grasp-phase[progressive]`    | 0.862890625      | 0.903515625      | 0.0          | 3156.924531     | 104.9810547     | 6.890873651     | 72.50599761     | 14233.6        | 71065.6     | 118528      | 618649.6      |
| Ablate `grasp-phase[absolute]`       | 0.891015625      | 0.9134765625     | 0.0001953125 | 2981.499725     | **95.24980469** | 6.381201206     | **70.91836072** | 12953.6        | 47718.4     | 107212.8    | **436889.6**  |
| Ablate `grasp-phase-terminal`        | 0.8732421875     | 0.90390625       | 0.001171875  | 2672.94798      | 100.9125        | 7.23261954      | 71.6605283      | 14592          | 62822.4     | 149555.2    | 728678.4      |
| Ablate `wrist-roll-pose`             | 0.890625         | 0.906640625      | 0.00078125   | **3201.138545** | 104.5677734     | 6.51755163      | 71.6091508      | **12185.6**    | 45977.6     | 156979.2    | 532428.8      |
| Ablate `avoid-bumping-cube`          | 0.8884765625     | 0.9150390625     | 0.000390625  | 3163.609642     | 98.59609375     | 6.680919734     | 71.16315117     | 14796.8        | 58624       | **99174.4** | 542412.8      |
| Ablate `time-penalty`                | 0.8685546875     | 0.9009765625     | 0.0005859375 | 3176.744661     | 103.6873047     | 7.016555307     | 71.29361734     | 13209.6        | 57190.4     | 102144      | 675328        |

> *Studies marked as `Baseline` use the reward configuration detailed in [Baseline Reward Configuration](#baseline-reward-configuration).

> **Studies marked as `Minimal` use a minimal reward configuration which is the same as `Baseline` with the following rewards were ablated: `approach_distance`, `approach_alignment`, `aproach_phase[progressive]`, `approach_phase[absolute]`, `grasp-phase[progressive]`, `grasp_phase_terminal`,  `wrist_roll_pose`, `avoid_bumping_cube`, `safety_touch_table`, `time_penalty`, and with the following rewards un-gated `lift_phase[progressive]`, `lift_phase[absolute]`.

### Vision Backbones

The following studies use the Baseline reward configuration, changing only the Vision backbone. All metrics are the mean value over **five trials with different RNG seeds**, each run to **50,000 steps**.

| Backbone | Success Rate   | Lift Rate        | Drop Rate    | Reward          | Episode Length  | Cube Bump       | Time-to-lift    | First Approach | First Grasp | First Lift   | First Success |
| -------- | -------------- | ---------------- | ------------ | --------------- | --------------- | --------------- | --------------- | -------------- | ----------- | ------------ | ------------- |
| ResNet18 | 0.8314453125   | 0.87421875       | 0.0009765625 | 3176.186971     | 120.5423828     | 7.025399015     | 79.42073773     | 13926.4        | **53196.8** | 168908.8     | 872960        |
| Gen1 CNN | 0.87109375     | 0.8966796875     | 0.0001953125 | 3128.433304     | **100.0230469** | **6.688415972** | **72.25066414** | 14336          | 54476.8     | 129280       | **583987.2**  |
| Gen2 CNN | **0.87734375** | **0.9017578125** | **0.0**      | **3194.855945** | 105.0898438     | 6.905817614     | 74.46541223     | **11673.6**    | 85196.8     | **108953.6** | 750899.2      |

### Baseline Reward Configuration
The Baseline study used the Gen 2 CNN. Its reward configuration is given below. 
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

### Individual Reward Contribution
Each reward contribution study starts the with `Minimal` configuration and enables only a single additional reward per study. The Reward Contribution studies were only run over a **single RNG** to **15,000 steps**.

| Study                             | Success Rate     | Lift Rate        | Drop Rate        | Reward          | Episode Length  | Cube Bump       | Time-to-lift | First Approach | First Grasp | First Lift | First Success |
| --------------------------------- | ---------------- | ---------------- | ---------------- | --------------- | --------------- | --------------- | ------------ | -------------- | ----------- | ---------- | ------------- |
| Minimal**                         | 0                | 0                | 0                | -56.69859497    | **37.25585938** | 4.563608208     |              | 15104          | 268544      | 378624     |               |
| Add `approach-distance`           | 0.0009765625     | 0.0400390625     | **0.0009765625** | 766.3568644     | 370.1933594     | 52.02282757     | 319.3170732  | 14336          | 145152      | 226048     | 3489536       |
| Add `approach-alignment`          | 0                | 0.0009765625     | 0                | -47.65941986    | 47.953125       | 6.804180612     | **38.0**     | 16640          | 237312      | 701440     |               |
| Add `approach-phase[progressive]` | 0                | 0.0390625        | **0.0009765625** | -41.04471339    | 71.51269531     | 12.57182435     | 81.025       | 16128          | 171520      | 431872     | 2350848       |
| Add `approach-phase[absolute]`    | 0                | 0.0078125        | 0                | 386.4333235     | 472.2011719     | 20.67633224     | 187.5        | 19712          | 145152      | 336384     | 1099776       |
| Add `approach-phase-terminal`     | **0.6904296875** | **0.7587890625** | 0.00390625       | **1897.867954** | 145.2841797     | 10.53539668     | 97.1956242   | 19200          | 97792       | 51200      | 967936        |
| Add `grasp-phase[progressive]`    | 0.0009765625     | 0.099609375      | 0                | 214.3310782     | 380.9365234     | 11.11078353     | 239.5882353  | 19712          | 222208      | 222464     | 1441792       |
| Add `grasp-phase[absolute]`       | 0.1513671875     | 0.2099609375     | 0                | 386.7596812     | 107.7841797     | 7.467258663     | 103.3767442  | 15104          | **52736**   | 761344     | 1348096       |
| Add `grasp-phase-terminal`        | 0.169921875      | 0.228515625      | 0.001953125      | 541.5431028     | 74.62207031     | 6.10631098      | 77.29059829  | 14848          | 272896      | 743168     | 1480448       |
| Add `wrist-roll-pose`             | 0                | 0                | 0                | -46.52412625    | 61.98535156     | 10.38436544     |              | **13824**      | 312832      | 664832     |               |
| Add `avoid-bumping-cube`          | 0                | 0.001953125      | 0.001953125      | -56.03133264    | 37.33886719     | **4.508705847** | 77           | 14848          | 450048      | 1582336    |               |
| Add `safety-touch-table`          | 0                | 0                | 0                | -53.91918025    | 46.32617188     | 6.492460823     |              | 22784          | 485888      | 374016     | 2989312       |
| Add `time-penalty`                | 0                | 0                | 0                | -52.83921073    | 38.609375       | 4.895316333     |              | 21760          | 433408      | 154880     |               |

### Composite Reward Contribution
Each composite reward contribution study starts the with `Minimal` configuration and enables only a single additional reward per study. The Reward Contribution studies were only run over a **single RNG** to **15,000 steps**.

| Study                                                                          | Success Rate  | Lift Rate        | Drop Rate    | Reward      | Episode Length  | Cube Bump       | Time-to-lift    | First Approach | First Grasp | First Lift | First Success |
| ------------------------------------------------------------------------------ | ------------- | ---------------- | ------------ | ----------- | --------------- | --------------- | --------------- | -------------- | ----------- | ---------- | ------------- |
| Baseline*                                                                      | 0.6416015625  | 0.87109375       | 0            | 3117.053055 | 155.9580078     | 7.698451978     | 82.4764574      | **15104**      | **79360**   | 168192     | **638464**    |
| Add `approach-phase-terminal`                                                  | 0.0087890625  | 0.056640625      | 0.001953125  | 425.4966236 | 168.5722656     | 20.31688515     | 153.3793103     | 25088          | 145408      | 77056      | 1093120       |
| Add `approach-phase-terminal`, `grasp-phase-terminal`                          | 0.833984375   | 0.8583984375     | 0.0009765625 | 2601.714988 | **116.7177734** | 8.052764858     | **81.07394767** | **15104**      | 157696      | 173824     | 1115136       |
| Add `grasp-phase[absolute]`, `approach-phase-terminal`                         | **0.859375 ** | **0.8837890625** | 0            | 2496.416327 | 131.3056641     | 8.900159954     | 90.83756906     | 19200          | 136960      | **54016**  | 709120        |
| Add `grasp-phase[absolute]`, `approach-phase-terminal`, `grasp-phase-terminal` | 0.7373046875  | 0.873046875      | 0.0029296875 | 2874.956084 | 147.2109375     | **6.666361651** | 103.0257271     | 18176          | 185856      | 172032     | 901376        |
