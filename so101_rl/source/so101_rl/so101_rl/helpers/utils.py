import omni.usd  # type: ignore
import omni.kit.commands  # type: ignore
from pxr import UsdShade  # type: ignore
import torch


VINYL_MDL_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Materials/Base/Plastics/Vinyl.mdl"
)

VINYL_MATERIAL_PATH = "/World/Looks/VinylMaterial"


def set_material(prim_path):
    stage = omni.usd.get_context().get_stage()

    if not stage.GetPrimAtPath("/World/Looks"):
        omni.kit.commands.execute(
            "CreatePrim",
            prim_path="/World/Looks",
            prim_type="Scope",
            select_new_prim=False,
        )

    if not stage.GetPrimAtPath(VINYL_MATERIAL_PATH):
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url=VINYL_MDL_URL,
            mtl_name="Vinyl",
            mtl_path=VINYL_MATERIAL_PATH,
            select_new_prim=False,
        )

    material_prim = stage.GetPrimAtPath(VINYL_MATERIAL_PATH)
    material = UsdShade.Material(material_prim)

    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI(prim).Bind(
        material, UsdShade.Tokens.strongerThanDescendants
    )


def assert_tensor(tensor: torch.Tensor, shape: tuple, dtype):
    """Utility to assert tensor shape and dtype."""
    assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
    assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
