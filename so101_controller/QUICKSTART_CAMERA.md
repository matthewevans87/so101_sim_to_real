# Quick Start Guide: USB Webcam with Vision Policy

This guide shows how to use a USB webcam with the vision-based robot controller.

## Prerequisites

```bash
pip install opencv-python torch torchvision numpy
```

## Step-by-Step Guide

### 1. Test Your Camera

First, verify your USB webcam is working:

```bash
# List available cameras
cd /home/matthew-evans/src/academics/cs6341-robotics-project-direct
python so101_controller/test_camera.py
```

Expected output:
```
Scanning for available cameras...
  Camera 0: Available (1280x720)
  Camera 1: Available (640x480)
```

### 2. Test Camera Capture

Capture some test frames to verify quality:

```bash
# Capture 30 frames and measure FPS
python so101_controller/test_camera.py --camera 0 --frames 30

# Save frames to inspect image quality
python so101_controller/test_camera.py --camera 0 --frames 10 --save ./test_frames
```

### 3. Run Vision Policy (Example)

```bash
# With checkpoint
python so101_controller/run_vision_policy_example.py \
    --camera 0 \
    --checkpoint /path/to/checkpoint.pt \
    --robot-port /dev/ttyACM0 \
    --urdf-path /path/to/so101_new_calib.urdf

# Without checkpoint (random weights - for testing only)
python so101_controller/run_vision_policy_example.py \
    --camera 0 \
    --robot-port /dev/ttyACM0
```

## Code Example

Here's the minimal code to use a camera with your vision policy:

```python
import torch
from so101_controller.CameraSource import CameraSource
from so101_controller.JointPositionPolicy import StormVisionPolicy
from so101_controller.PolicyController import PolicyController

# Setup camera
camera = CameraSource(camera_id=0)

# Setup vision policy
policy = StormVisionPolicy(
    num_joints=6,
    act_dim=6,
    device="cuda"
)

# Load checkpoint
checkpoint = torch.load("checkpoint.pt")
policy.load_state_dict(checkpoint['policy'])
policy.eval()

# Run policy with camera
controller = PolicyController(config, robot_interface)
controller.run_policy(policy=policy, camera_source=camera)

# Cleanup
camera.release()
```

## Camera Selection Guide

### Linux
- **Camera 0**: Usually `/dev/video0` (first USB camera)
- **Camera 1**: Usually `/dev/video1` (second USB camera)
- Check with: `ls /dev/video*`

### macOS
- **Camera 0**: Usually built-in FaceTime camera
- **Camera 1**: Usually first USB camera
- **Camera 2**: Usually second USB camera

### How to Choose

1. Run the test script to see all available cameras
2. Use `--save` option to inspect image quality
3. Choose the camera with the best view of your workspace

## Troubleshooting

### "No cameras found"

**Linux:**
```bash
# Check if camera is detected
lsusb | grep -i camera

# Check video devices
ls -l /dev/video*

# Add yourself to video group
sudo usermod -a -G video $USER
# Then log out and back in
```

**macOS:**
```
1. Open System Preferences > Security & Privacy > Camera
2. Enable camera access for Terminal (or your IDE)
3. Replug the USB camera
```

### "Camera open failed"

1. Close other applications using the camera (Zoom, Skype, Photo Booth, etc.)
2. Try unplugging and replugging the USB camera
3. Try a different USB port
4. Try a different camera ID

### Poor image quality

1. Adjust camera position and lighting
2. Clean camera lens
3. Try higher resolution: `--width 1280 --height 720`
4. Check if camera supports the resolution you requested

### Low FPS / Lag

1. Lower the resolution in CameraSource
2. Use USB 3.0 port instead of USB 2.0
3. Reduce control frequency (`hz` in ControllerConfiguration)
4. Use GPU if available: `device="cuda"`

## Files Reference

| File | Purpose |
|------|---------|
| `CameraSource.py` | Camera interface implementation |
| `test_camera.py` | Camera testing utility |
| `run_vision_policy_example.py` | Complete working example |
| `CAMERA_README.md` | Detailed camera documentation |
| `VISION_POLICY_README.md` | Vision policy documentation |

## Camera Output Format

The camera provides frames as:
- **Type**: `torch.Tensor`
- **Shape**: `(H, W, 3)` where H=height, W=width, 3=RGB
- **Values**: `uint8` in range [0, 255]
- **Color**: RGB (not BGR)

This is automatically preprocessed by `StormVisionPolicy`:
1. Resized to 224×224 (for ResNet18)
2. Normalized with ImageNet stats
3. Fed through frozen ResNet18 → 512 features
4. Concatenated with joint positions
5. Passed to PPO network → actions

## Performance Tips

1. **Resolution**: 640×480 is usually sufficient and faster than 1280×720
2. **Lighting**: Ensure good, consistent lighting in your workspace
3. **Camera position**: Mount camera in stable position with good view
4. **GPU**: Use CUDA if available for faster inference
5. **Control loop**: 10-20 Hz is reasonable for vision-based control

## Next Steps

1. ✅ Test camera with `test_camera.py`
2. ✅ Verify image quality with `--save` option
3. ✅ Train your vision policy in Isaac Lab
4. ✅ Export the checkpoint
5. ✅ Run on real robot with `run_vision_policy_example.py`

Happy robot controlling! 🤖📷
