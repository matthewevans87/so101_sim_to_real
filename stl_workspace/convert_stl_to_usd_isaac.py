"""
convert_stl_to_usd_isaac.py — Convert STL to USD using omni.kit.asset_converter.

Must be run inside the env_isaaclab conda environment:
    conda activate env_isaaclab
    python3 stl_workspace/convert_stl_to_usd_isaac.py

Uses Isaac Sim's built-in converter which:
  - Deduplicates vertices → proper manifold mesh
  - Sets correct USD schemas and metadata
  - Produces a compact, well-formed USD vs the raw flat-vertex USDA we wrote manually

Output: assets/robots/camera_mount.usd  (overwrites any existing file)
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_STL = str(PROJECT_ROOT / "assets/robots/SO-ARM101_camera_wrist_mount.stl")
OUTPUT_USD = str(PROJECT_ROOT / "assets/robots/camera_mount.usd")

# STL files are typically in mm; Isaac Sim works in metres.
# asset_converter does NOT rescale geometry — we apply a uniform scale of 0.001
# via the ScaleFactor in the converter context.
SCALE_MM_TO_M = 0.001


def progress_callback(progress: float, total_steps: int) -> None:
    bar_len = 30
    filled = int(bar_len * progress)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {progress*100:.1f}%", end="", flush=True)


async def convert_asset(input_path: str, output_path: str) -> bool:
    import omni.kit.asset_converter as asset_converter  # type: ignore

    ctx = asset_converter.AssetConverterContext()
    # Merge everything into a single mesh prim
    ctx.merge_all_meshes = True
    # Smooth normals for better visual quality
    ctx.smooth_normals = True
    # We don't need embedded materials from the STL
    ctx.ignore_materials = True
    ctx.ignore_animation = True
    ctx.ignore_cameras = True
    ctx.ignore_lights = True
    # Output USD, not USDA (binary is smaller)
    ctx.export_preview_surface = False
    # Scale mm → m
    ctx.scale = SCALE_MM_TO_M

    task = asset_converter.get_instance().create_converter_task(
        input_path, output_path, progress_callback, ctx
    )
    success = await task.wait_until_finished()
    print()  # newline after progress bar
    if not success:
        detail = task.get_error_message()
        print(f"  ERROR: {detail}", file=sys.stderr)
    return success


def main() -> None:
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_STL
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_USD

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Scale  : {SCALE_MM_TO_M}  (mm → m)")

    from isaacsim import SimulationApp  # type: ignore

    app = SimulationApp({"headless": True, "renderer": "RaytracingLighting"})

    print("Converting …")
    success = asyncio.get_event_loop().run_until_complete(
        convert_asset(input_path, output_path)
    )

    app.close()

    if success:
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Wrote {output_path}  ({size_kb:.0f} KiB)")
        print("Done.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
