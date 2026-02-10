ctx = omni.usd.get_context()
stage = ctx.get_stage()
CAMERA_PRIM_PATH = "/World/workspace_cam"  # your camera
prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)

from omni.usd import get_world_transform_matrix

mat = get_world_transform_matrix(prim)
rot = mat.ExtractRotation()
q = rot.GetQuat()

w = float(q.GetReal())
v = q.GetImaginary()
x, y, z = float(v[0]), float(v[1]), float(v[2])
