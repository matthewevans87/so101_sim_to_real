import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import os
from isaaclab.sensors import ContactSensorCfg

WORKSPACE_PATH = os.environ.get("ISAAC_LAB_WORKSPACE_PATH", "/workspace")

SO101_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            WORKSPACE_PATH, "assets/robots/so101_new_calib/so101_new_calib.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=24,  # more position iterations for better contacts
            solver_velocity_iteration_count=2,  # enable velocity iterations to reduce slip
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(
            0.0,
            0.0,
            -0.03,
        ),  # Spawn position (x, y, z) - adjust as needed for your environment
        rot=(
            0.7071068,
            0.0,
            0.0,
            0.7071068,
        ),  # Spawn orientation (w, x, y, z) - upright orientation
        joint_pos={
            # Add your robot's joint names and default angles
            # e.g., "joint_1": 0.0, "joint_2": -0.5, "gripper_joint": 0.0, ...
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness={
                "shoulder_pan": 200.0,  # Highest - moves all mass
                "shoulder_lift": 170.0,  # Slightly less than rotation
                "elbow_flex": 120.0,  # Reduced based on less mass
                "wrist_flex": 80.0,  # Reduced for less mass
                "wrist_roll": 50.0,  # Low mass to move
            },
            damping={
                "shoulder_pan": 80.0,
                "shoulder_lift": 65.0,
                "elbow_flex": 45.0,
                "wrist_flex": 30.0,
                "wrist_roll": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=1.0,  # was 8.0  (start here)
            velocity_limit_sim=0.5,  # was 1.5
            stiffness=40.0,  # was 120.0
            damping=12.0,  # was 40.0
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

GRIPPER_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper",
    track_air_time=False,
    update_period=0.0,
)


SO101_NUM_JOINTS = 6
