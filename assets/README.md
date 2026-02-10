# Assets Directory

This directory contains all 3D models, textures, and other assets used in the project.

## Structure

```
assets/
├── robots/          # Robot models (SO-101 arm)
├── tools/           # Tool models (hook, stick, spatula)
├── objects/         # Manipulation objects (cubes, cylinders, etc.)
└── scenes/          # Pre-built scene USD files
```

## Robot Models (`robots/`)

### SO-101 Robotic Arm

**✅ AVAILABLE**: `so101-teleoperate.usd` - SO-101 robot USD file

**Files:**
- `so101-teleoperate.usd` - Main robot USD file ✅
- `so101.urdf` - URDF description (optional, for reference)
- `README.md` - Robot specifications and setup instructions

**Usage in code:**
```python
# Automatically loaded by SO101BaseEnv
env = SO101BaseEnv()
env.setup()  # Loads from assets/robots/so101-teleoperate.usd

# Or specify custom path via environment variable
export SO101_USD_PATH=/path/to/custom/so101.usd
```

**How to add additional robot files:**

1. **From manufacturer:**
   - Request USD/URDF files from SO-101 manufacturer
   - Place in `assets/robots/`

2. **From URDF:**
   ```bash
   # Convert URDF to USD using Isaac Sim
   /opt/isaac-sim/isaac-sim-5.1.0/python.sh -m omni.isaac.asset_converter \
       --input so101.urdf \
       --output assets/robots/so101/so101.usd
   ```

3. **From CAD files:**
   - Import CAD files into Isaac Sim
   - Set up articulation hierarchy
   - Configure joints, collision meshes
   - Export as USD

**Expected properties:**
- 7 DOF articulated arm
- End-effector/gripper
- Collision meshes
- Visual meshes
- Joint limits and dynamics

## Tool Models (`tools/`)

### Hook Tool
- `hook_tool.usd` - Curved hook for pulling objects
- Dimensions: ~20cm length, 5cm hook radius

### Straight Stick
- `straight_stick.usd` - Straight tool for pushing
- Dimensions: ~30cm length, 1cm diameter

### Spatula/Scoop
- `spatula_tool.usd` - Flat tool for scooping/lifting
- Dimensions: ~15cm handle, 10cm blade

**Creating tools:**
```bash
# Create simple primitives or import from CAD
# Tools should be graspable by SO-101 gripper
```

## Manipulation Objects (`objects/`)

### Basic Shapes
- `cube_small.usd` - 5cm cube
- `cube_medium.usd` - 10cm cube
- `cube_large.usd` - 15cm cube
- `cylinder_small.usd` - 5cm diameter, 10cm height
- `cylinder_large.usd` - 10cm diameter, 20cm height
- `sphere_small.usd` - 5cm diameter
- `sphere_large.usd` - 10cm diameter

**Properties:**
- Physics enabled (rigid bodies)
- Collision meshes
- Various masses (0.1-1.0 kg)
- Different materials (wood, metal, plastic)

### Custom Objects
Add task-specific objects here (boxes, bottles, etc.)

## Scenes (`scenes/`)

Pre-built scene files for different training phases:

- `reach_training_scene.usd` - Basic reaching setup
- `manipulation_scene.usd` - Object manipulation setup
- `tool_usage_scene.usd` - Tool-based manipulation setup

## Asset Guidelines

### File Formats
- **Primary**: USD (Universal Scene Description)
- **Source**: URDF, STL, OBJ, FBX (convert to USD)

### Naming Convention
```
<category>_<name>_<variant>.usd
Examples:
- robot_so101_v1.usd
- tool_hook_short.usd
- object_cube_10cm.usd
```

### Physics Properties
All objects should have:
- Mass (kg)
- Friction coefficient
- Restitution (bounciness)
- Collision meshes (simplified)
- Visual meshes (detailed)

### Scale
- Use meters as base unit
- Match Isaac Sim default scale

## Adding New Assets

1. **Create or obtain asset** (CAD, URDF, etc.)
2. **Convert to USD** if needed
3. **Test in Isaac Sim** (load and verify)
4. **Add to appropriate directory**
5. **Update this README** with specifications
6. **Commit to repository**

## Asset Checklist

Before committing assets, verify:
- [ ] File is in USD format
- [ ] Physics properties are set
- [ ] Collision meshes are defined
- [ ] Scale is correct (meters)
- [ ] Joints are properly configured (for articulated objects)
- [ ] File size is reasonable (<50MB per asset)
- [ ] Asset loads without errors in Isaac Sim

## Asset Status

### Available ✅
1. **SO-101 Robot** - `so101-teleoperate.usd` ✅

### Missing Assets (TODO)

Priority assets needed:

1. **Hook Tool** - For pulling objects
2. **Straight Stick** - For pushing objects
3. **Basic Objects** - Cubes, cylinders for manipulation
4. **Spatula Tool** - For scooping/lifting

## References

- [Isaac Sim Asset Converter](https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html#asset-converter)
- [USD Format](https://graphics.pixar.com/usd/docs/index.html)
- [Creating Articulations](https://docs.omniverse.nvidia.com/isaacsim/latest/features/scene_understanding/ext_omni_isaac_articulation.html)
