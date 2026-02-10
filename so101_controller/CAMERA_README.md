# USB Webcam Camera Source

Cross-platform camera source implementation for vision-based robot control using USB webcams.

## Features

- ✅ **Cross-platform**: Works on both Linux and macOS
- ✅ **Easy to use**: Simple callable interface compatible with `PolicyController`
- ✅ **Auto-reconnect**: Automatically recovers from camera disconnections
- ✅ **Flexible**: Supports custom resolutions and multiple camera devices
- ✅ **Robust**: Built-in error handling and camera detection

## Quick Start

### 1. Install Dependencies

```bash
pip install opencv-python torch numpy
```

### 2. Test Your Camera

```bash
# List available cameras
python so101_controller/test_camera.py

# Test specific camera
python so101_controller/test_camera.py --camera 0 --frames 30

# Save test frames for inspection
python so101_controller/test_camera.py --camera 0 --save ./test_frames
```

### 3. Use with Vision Policy

```bash
# Run vision-based policy with camera 0
python so101_controller/run_vision_policy_example.py \
    --camera 0 \
    --checkpoint path/to/checkpoint.pt \
    --robot-port /dev/ttyACM0
```

## Usage

### Basic Usage

```python
from so101_controller.CameraSource import CameraSource

# Create camera source
camera = CameraSource(camera_id=0)

# Get a frame
frame = camera.get_frame()  # Returns torch.Tensor of shape (H, W, 3)

# Use as callable (for PolicyController)
frame = camera()  # Same as camera.get_frame()

# Clean up
camera.release()
```

### Context Manager (Recommended)

```python
from so101_controller.CameraSource import CameraSource

with CameraSource(camera_id=0) as camera:
    frame = camera.get_frame()
    # Camera automatically released on exit
```

### Custom Resolution

```python
camera = CameraSource(
    camera_id=0,
    width=1280,
    height=720
)
```

### With PolicyController

```python
from so101_controller.CameraSource import CameraSource
from so101_controller.PolicyController import PolicyController

# Initialize camera
camera = CameraSource(camera_id=0)

# Run policy (camera is callable)
controller.run_policy(
    policy=vision_policy,
    camera_source=camera  # Pass camera object directly
)
```

## API Reference

### CameraSource

**Constructor:**
```python
CameraSource(
    camera_id: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    auto_retry: bool = True
)
```

**Parameters:**
- `camera_id`: Camera device index
  - Linux: 0 = `/dev/video0`, 1 = `/dev/video1`, etc.
  - macOS: 0 = FaceTime camera, 1+ = USB cameras
- `width`: Desired frame width (None = camera default)
- `height`: Desired frame height (None = camera default)
- `auto_retry`: Auto-reconnect on read failures

**Methods:**
- `get_frame() -> torch.Tensor`: Capture RGB frame, shape (H, W, 3), dtype uint8
- `get_resolution() -> Tuple[int, int]`: Get current (width, height)
- `release()`: Release camera resource
- `__call__() -> torch.Tensor`: Same as `get_frame()` (for callable interface)

### Utility Functions

**list_available_cameras(max_test: int = 10) -> list[int]:**
```python
from so101_controller.CameraSource import list_available_cameras

available = list_available_cameras()
print(f"Available cameras: {available}")
```

**test_camera_source(camera_id: int = 0, num_frames: int = 10):**
```python
from so101_controller.CameraSource import test_camera_source

test_camera_source(camera_id=0, num_frames=30)
```

## Platform-Specific Notes

### Linux

**Permissions:**
```bash
# Check camera devices
ls -l /dev/video*

# Add user to video group if needed
sudo usermod -a -G video $USER
# Log out and back in for changes to take effect
```

**Backend:**
- Uses V4L2 (Video4Linux2) by default
- Falls back to default OpenCV backend if V4L2 unavailable

### macOS

**Permissions:**
- Grant camera access in: System Preferences > Security & Privacy > Camera
- Allow Terminal or your IDE to access the camera

**Backend:**
- Uses AVFoundation (native macOS framework)
- Typically: Camera 0 = FaceTime, Camera 1+ = USB cameras

## Troubleshooting

### No cameras detected

**Linux:**
1. Check USB connection: `lsusb`
2. Check video devices: `ls -l /dev/video*`
3. Check permissions: `groups` (should include "video")
4. Test with: `ffplay /dev/video0` or `cheese`

**macOS:**
1. Check USB connection: System Information > USB
2. Check permissions: System Preferences > Security & Privacy > Camera
3. Test with Photo Booth or QuickTime

### Camera open fails

1. **Already in use**: Close other applications using the camera (Zoom, Skype, etc.)
2. **Wrong camera ID**: Run `test_camera.py` to list available cameras
3. **Permissions**: See platform-specific notes above

### Low FPS / Performance issues

1. **Reduce resolution**: Use `width` and `height` parameters
2. **Check USB bandwidth**: USB 2.0 may limit high-resolution streams
3. **Try different camera**: Some cameras have better drivers/support

### Frame capture fails intermittently

- `auto_retry=True` (default) will attempt automatic reconnection
- Check USB cable quality/connection
- Try different USB port

## Examples

### Save Camera Frames

```python
from so101_controller.CameraSource import CameraSource
from PIL import Image

with CameraSource(camera_id=0) as camera:
    for i in range(10):
        frame = camera.get_frame()
        # Convert to PIL Image and save
        img = Image.fromarray(frame.cpu().numpy())
        img.save(f"frame_{i:03d}.png")
```

### Monitor FPS

```python
from so101_controller.CameraSource import CameraSource
import time

with CameraSource(camera_id=0) as camera:
    num_frames = 100
    start = time.time()
    
    for _ in range(num_frames):
        frame = camera.get_frame()
    
    elapsed = time.time() - start
    fps = num_frames / elapsed
    print(f"Average FPS: {fps:.1f}")
```

### Multiple Cameras

```python
from so101_controller.CameraSource import CameraSource

# Open multiple cameras
camera_left = CameraSource(camera_id=0)
camera_right = CameraSource(camera_id=1)

try:
    frame_left = camera_left.get_frame()
    frame_right = camera_right.get_frame()
finally:
    camera_left.release()
    camera_right.release()
```

## Output Format

All camera frames are returned as:
- **Type**: `torch.Tensor`
- **Shape**: `(H, W, 3)` - Height × Width × RGB channels
- **Data type**: `uint8`
- **Value range**: `[0, 255]`
- **Color format**: RGB (not BGR)

This format is directly compatible with `StormVisionPolicy.forward()`.
