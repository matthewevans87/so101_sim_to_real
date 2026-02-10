# Vision-Based Policy Implementation

This document describes the implementation of `StormVisionPolicy`, a vision-based control policy that uses a frozen ResNet18 encoder for camera data processing.

## Architecture

The `StormVisionPolicy` implements the following pipeline:

```
Camera RGB Image → ResNet18 (frozen) → 512 features → [concat with joint positions] → PPO Network → Actions
```

### Components

1. **Vision Encoder (ResNet18)**
   - Pretrained on ImageNet with default weights
   - Final classification layer removed (outputs 512-dim features)
   - **Frozen** - gradients disabled, used only for inference
   - Applies ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

2. **Policy Network**
   - Input: 512 (vision features) + num_joints (joint positions)
   - Architecture: 2 hidden layers with 32 units each, ELU activation
   - Output: Actions in [-1, 1] range (tanh activation)

## Usage

### 1. Creating the Policy

```python
from so101_controller.JointPositionPolicy import StormVisionPolicy

policy = StormVisionPolicy(
    num_joints=6,      # Number of robot joints
    act_dim=6,         # Action dimension
    device="cuda"      # or "cpu"
)
```

### 2. Running with PolicyController

The `PolicyController` has been updated to support both joint-based and vision-based policies:

```python
from so101_controller.PolicyController import PolicyController
from so101_controller.ControllerConfiguration import ControllerConfiguration
from so101_controller.So101RobotInterface import So101RobotInterface

# Setup robot interface and configuration
robot_interface = So101RobotInterface(...)
controller_config = ControllerConfiguration(...)

# Create controller
controller = PolicyController(
    controller_config=controller_config,
    robot_interface=robot_interface
)

# Define camera source function
def get_camera_frame():
    """Returns camera RGB as torch.Tensor with shape (H, W, 3)"""
    # Your camera capture code here
    return camera_rgb_tensor

# Run the vision policy
controller.run_policy(
    policy=policy,
    camera_source=get_camera_frame  # Required for StormVisionPolicy
)
```

### 3. Camera Source Requirements

The `camera_source` callable must return:
- **Shape**: `(H, W, 3)` or `(N, H, W, 3)` 
- **Format**: RGB (not BGR)
- **Data type**: `torch.uint8` or `torch.float32`
- **Value range**: 
  - `uint8`: [0, 255]
  - `float32`: [0.0, 1.0]

The policy will automatically:
- Handle batched or single images
- Resize to 224×224 for ResNet18
- Convert to NCHW format
- Apply ImageNet normalization

## Implementation Details

### StormVisionPolicy Class

**Constructor Parameters:**
- `num_joints` (int): Number of robot joints
- `act_dim` (int): Action dimension (typically equals num_joints)
- `device` (str): "cuda" or "cpu"

**Forward Method:**
```python
def forward(camera_rgb: torch.Tensor, joint_positions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        camera_rgb: Raw camera RGB, shape (N, H, W, 3) or (H, W, 3)
        joint_positions: Joint positions, shape (N, num_joints) or (num_joints,)
    
    Returns:
        Actions in [-1, 1], shape (N, act_dim) or (act_dim,)
    """
```

### PolicyController Updates

**New Parameter:**
- `camera_source` (Optional[Callable]): Function that returns camera frames
  - Required when using `StormVisionPolicy`
  - Not needed for joint-based policies

**Automatic Detection:**
The controller automatically detects if the policy is vision-based using:
```python
is_vision_policy = isinstance(policy, StormVisionPolicy)
```

## Comparison with Training Environment

The implementation matches the Isaac Lab training environment:

| Component | Training (Isaac Lab) | Deployment (StormVisionPolicy) |
|-----------|---------------------|-------------------------------|
| Vision Encoder | ResNet18 (frozen) | ResNet18 (frozen) |
| Vision Features | 512-dim | 512-dim |
| Preprocessing | ImageNet normalization | ImageNet normalization |
| Input Resize | 224×224 | 224×224 |
| Observation | [512 vision + 6 joints] | [512 vision + 6 joints] |
| Policy Network | 2×[32, ELU] → tanh | 2×[32, ELU] → tanh |

## Example Script

See `run_vision_policy_example.py` for a complete working example.

## Loading Pretrained Weights

To load weights from a trained model:

```python
# Load checkpoint
checkpoint = torch.load("path/to/checkpoint.pt", map_location=device)

# Load only the policy network weights (not ResNet18, as it's pretrained)
policy.fc1.load_state_dict(checkpoint['fc1'])
policy.fc2.load_state_dict(checkpoint['fc2'])
policy.fc3.load_state_dict(checkpoint['fc3'])

# Or if the checkpoint contains the full state dict:
policy.load_state_dict(checkpoint['policy'], strict=False)
```

## Notes

- The ResNet18 backbone is **always frozen** - it only runs inference
- Camera preprocessing happens automatically in `_preprocess_camera()`
- The policy expects joint positions in **radians**
- Actions are output in **[-1, 1]** range and mapped to joint limits by PolicyController
