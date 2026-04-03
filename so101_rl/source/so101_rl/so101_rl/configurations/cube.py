import os
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

CUBE_DEFAULT_DIMS = (0.03, 0.03, 0.03)  # 3cm cube
WORKSPACE_PATH = os.environ.get("ISAAC_LAB_WORKSPACE_PATH", "/workspace")
CUBE_WIDTH = 0.03  # 3cm cube
CUBE_RESTING_HEIGHT = CUBE_WIDTH / 2  # half the cube height (0.03m cube)
DEX_CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.2, 0.0, CUBE_RESTING_HEIGHT * 2),  # a tiny bit above the ground
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            WORKSPACE_PATH, "assets/props/DexCube/dex_cube_instanceable.usd"
        ),
        scale=(0.5, 0.5, 0.5),  # original cube is 0.06m, scale down to 0.03m
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

CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.2, 0.0, 0.03),  # a tiny bit above the ground
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.CuboidCfg(
        size=(0.03, 0.03, 0.03),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,  # keep gravity on
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # small but non-zero
        collision_props=sim_utils.CollisionPropertiesCfg(),  # enable collisions
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0),
        ),
        activate_contact_sensors=True,
    ),
)
