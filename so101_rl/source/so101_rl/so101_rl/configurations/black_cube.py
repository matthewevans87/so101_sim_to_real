import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

WORKSPACE_PATH = os.environ.get("ISAAC_LAB_WORKSPACE_PATH", "/workspace")

BLACK_CUBE_WIDTH = 0.03  # 3 cm
BLACK_CUBE_RESTING_HEIGHT = BLACK_CUBE_WIDTH / 2  # 0.015 m — half-height

# Mass is baked into the USDA via PhysicsMassAPI (14.71 g = 0.01471 kg).
# Friction and restitution are baked in via PhysicsMaterialAPI.
# The USD is authored at true 3 cm scale; no rescaling is required.
BLACK_CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.2, 0.0, BLACK_CUBE_RESTING_HEIGHT * 2),  # a tiny bit above the ground
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(WORKSPACE_PATH, "assets/props/BlackCube/black_cube.usda"),
        scale=(1.0, 1.0, 1.0),  # USDA is already at true 3 cm scale
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        activate_contact_sensors=True,
        semantic_tags=[("class", "cube")],
    ),
)
