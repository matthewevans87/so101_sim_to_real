import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg

TABLE_SIZE = (2.0, 2.0, 1.0)  # (x, y, z) in meters

TABLE_CFG = RigidObjectCfg(
    # One table per env:
    prim_path="{ENV_REGEX_NS}/Table",
    # Spawn a 1m cube:
    spawn=sim_utils.CuboidCfg(
        size=TABLE_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # table is static/kinematic
            kinematic_enabled=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            density=500.0,  # doesn't matter much if kinematic
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            # default is fine; it's just a collision surface
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.9, 0.9, 0.9),  # white table (matches real workspace)
        ),
        activate_contact_sensors=True,
    ),
    # Place cube center so that env origin = back-center of table top.
    init_state=RigidObjectCfg.InitialStateCfg(
        # center of cube
        pos=(0.45, 0.0, -0.5),
        rot=(1.0, 0.0, 0.0, 0.0),  # identity quaternion (w, x, y, z)
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)


TABLE_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    track_air_time=False,
    update_period=0.0,
)
