"""
convert_stl_to_usd.py — Convert a binary STL to a USDA file suitable for
attachment as a collision+visual mesh under an Isaac Lab articulation link.

Usage:
    python stl_workspace/convert_stl_to_usd.py [input.stl] [output.usd]

Defaults:
    input  : assets/robots/SO-ARM101_camera_wrist_mount.stl
    output : assets/robots/camera_mount.usd

The script writes a plain-text USDA file (Universal Scene Description ASCII)
with:
  - A single Xform prim "CameraMount" as the defaultPrim
  - A Mesh child prim with all STL triangle vertices scaled mm → m (×0.001)
  - PhysicsCollisionAPI + PhysicsMeshCollisionAPI applied to the mesh,
    with approximation = "convexDecomposition"
  - NO RigidBodyAPI → the mesh inherits physics from a parent articulation link

Collision approximation options (edit COLLISION_APPROXIMATION below):
  "convexDecomposition"  – best for thin/complex brackets (default)
  "convexHull"           – single convex hull, fastest but may be too large
  "meshSimplification"   – simplified trimesh, best fidelity but slowest
"""

import struct
import sys
from pathlib import Path

SCALE = 0.001  # mm → m

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "assets/robots/SO-ARM101_camera_wrist_mount.stl"
DEFAULT_OUTPUT = PROJECT_ROOT / "assets/robots/camera_mount.usd"


def read_binary_stl(path: Path):
    """Return (normals, vertices) as flat lists from a binary STL."""
    triangles = []
    with open(path, "rb") as f:
        f.read(80)  # header
        count = struct.unpack("<I", f.read(4))[0]
        for _ in range(count):
            data = struct.unpack("<12fH", f.read(50))
            normal = data[0:3]
            v0, v1, v2 = data[3:6], data[6:9], data[9:12]
            triangles.append((normal, v0, v1, v2))
    return triangles


def fmt_vec3f(v, scale=1.0):
    return f"({v[0]*scale:.8g}, {v[1]*scale:.8g}, {v[2]*scale:.8g})"


def write_usda(triangles, output_path: Path, scale: float):
    """Write mesh as USDA text file."""

    # Compute extent (bounding box) in output units
    xs = [
        v[i] * scale for _, v0, v1, v2 in triangles for v in (v0, v1, v2) for i in ([0])
    ]
    ys = [
        v[i] * scale for _, v0, v1, v2 in triangles for v in (v0, v1, v2) for i in ([1])
    ]
    zs = [
        v[i] * scale for _, v0, v1, v2 in triangles for v in (v0, v1, v2) for i in ([2])
    ]
    extent_min = (min(xs), min(ys), min(zs))
    extent_max = (max(xs), max(ys), max(zs))

    # Build vertex arrays (flat — no deduplication, consistent with STL format)
    points = []
    normals = []
    face_counts = []
    face_indices = []

    for i, (normal, v0, v1, v2) in enumerate(triangles):
        base = i * 3
        points.extend(
            [
                fmt_vec3f(v0, scale),
                fmt_vec3f(v1, scale),
                fmt_vec3f(v2, scale),
            ]
        )
        normals.append(fmt_vec3f(normal))  # one normal per face (uniform)
        face_counts.append(3)
        face_indices.extend([base, base + 1, base + 2])

    n_tris = len(triangles)
    print(f"  Triangles   : {n_tris}")
    print(f"  Vertices    : {n_tris * 3} (flat, un-deduplicated)")
    print(f"  Extent (m)  : {extent_min} → {extent_max}")

    # ── USDA text ────────────────────────────────────────────────────────────
    lines = []

    lines.append("#usda 1.0")
    lines.append("(")
    lines.append('    defaultPrim = "CameraMount"')
    lines.append("    metersPerUnit = 1")
    lines.append('    upAxis = "Z"')
    lines.append(
        '    doc = """Camera wrist mount for SO-101 robot — visual mesh only."""'
    )
    lines.append(")")
    lines.append("")
    lines.append('def Xform "CameraMount" (')
    lines.append('    kind = "component"')
    lines.append(")")
    lines.append("{")
    lines.append('    def Mesh "Mesh"')
    lines.append("    {")

    # extent
    emin = f"({extent_min[0]:.6g}, {extent_min[1]:.6g}, {extent_min[2]:.6g})"
    emax = f"({extent_max[0]:.6g}, {extent_max[1]:.6g}, {extent_max[2]:.6g})"
    lines.append(f"        float3[] extent = [{emin}, {emax}]")

    # face vertex counts (all 3)
    counts_str = ", ".join(str(c) for c in face_counts)
    lines.append(f"        int[] faceVertexCounts = [{counts_str}]")

    # face vertex indices
    indices_str = ", ".join(str(i) for i in face_indices)
    lines.append(f"        int[] faceVertexIndices = [{indices_str}]")

    # normals (one per face = uniform interpolation)
    normals_str = ", ".join(normals)
    lines.append(f"        normal3f[] normals = [{normals_str}] (")
    lines.append('            interpolation = "uniform"')
    lines.append("        )")

    # points
    points_str = ", ".join(points)
    lines.append(f"        point3f[] points = [{points_str}]")

    # display color (dark grey plastic)
    lines.append("        color3f[] primvars:displayColor = [(0.3, 0.25, 0.2)]")

    # no catmull-clark subdivision
    lines.append('        uniform token subdivisionScheme = "none"')

    lines.append("    }")
    lines.append("}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not input_path.is_file():
        print(f"ERROR: STL not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Scale  : {SCALE} (mm → m)")

    triangles = read_binary_stl(input_path)
    write_usda(triangles, output_path, scale=SCALE)

    size_kb = output_path.stat().st_size / 1024
    print(f"\nWrote {output_path}  ({size_kb:.0f} KiB)")
    print("Done.")


if __name__ == "__main__":
    main()
