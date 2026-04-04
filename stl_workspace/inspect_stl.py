"""
inspect_stl.py — Parse binary STL and report geometry statistics.

Usage:
    python stl_workspace/inspect_stl.py [path/to/file.stl]

Prints: triangle count, bounding box (mm), centroid, and key dimensions.
Run this BEFORE convert_stl_to_usd.py to understand where the mesh origin
sits relative to the geometry — needed to set CAMERA_MOUNT_RELATIVE_POS.
"""

import struct
import sys
from pathlib import Path

DEFAULT_STL = (
    Path(__file__).resolve().parent.parent
    / "assets/robots/SO-ARM101_camera_wrist_mount.stl"
)


def read_binary_stl(path: Path):
    """Read binary STL; return list of (normal, v0, v1, v2) tuples (floats in file units)."""
    triangles = []
    with open(path, "rb") as f:
        f.read(80)  # header
        count = struct.unpack("<I", f.read(4))[0]
        for _ in range(count):
            data = struct.unpack("<12fH", f.read(50))
            normal = data[0:3]
            v0 = data[3:6]
            v1 = data[6:9]
            v2 = data[9:12]
            triangles.append((normal, v0, v1, v2))
    return triangles


def main():
    stl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STL

    if not stl_path.is_file():
        print(f"ERROR: STL not found at {stl_path}", file=sys.stderr)
        sys.exit(1)

    print(f"STL file : {stl_path}")
    print(f"Size     : {stl_path.stat().st_size / 1024:.1f} KiB")

    triangles = read_binary_stl(stl_path)
    print(f"Triangles: {len(triangles)}")

    # Collect all vertices
    xs, ys, zs = [], [], []
    for _, v0, v1, v2 in triangles:
        for v in (v0, v1, v2):
            xs.append(v[0])
            ys.append(v[1])
            zs.append(v[2])

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    xsize = xmax - xmin
    ysize = ymax - ymin
    zsize = zmax - zmin

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2

    print()
    print("─── Bounding box (STL file units, presumed mm) ──────────────────────")
    print(f"  X : [{xmin:+10.4f}, {xmax:+10.4f}]   size = {xsize:.4f}")
    print(f"  Y : [{ymin:+10.4f}, {ymax:+10.4f}]   size = {ysize:.4f}")
    print(f"  Z : [{zmin:+10.4f}, {zmax:+10.4f}]   size = {zsize:.4f}")
    print()
    print("─── Centroid ────────────────────────────────────────────────────────")
    print(f"  ({cx:+.4f}, {cy:+.4f}, {cz:+.4f})")
    print()
    print("─── Dimensions (mm) ─────────────────────────────────────────────────")
    dims = sorted([(xsize, "X"), (ysize, "Y"), (zsize, "Z")], reverse=True)
    for size, axis in dims:
        print(f"  {axis}: {size:.2f} mm")
    print()
    print("─── Origin proximity ────────────────────────────────────────────────")
    print(f"  Origin (0,0,0) is at offset from centroid:")
    print(f"    ({-cx:+.4f}, {-cy:+.4f}, {-cz:+.4f}) mm")
    print(
        f"  Origin is {'inside' if (xmin <= 0 <= xmax and ymin <= 0 <= ymax and zmin <= 0 <= zmax) else 'OUTSIDE'} the bounding box"
    )
    print()
    print("─── Conversion hints ────────────────────────────────────────────────")
    print("  STL units are typically mm; Isaac Sim uses metres.")
    print(f"  Scale factor to apply in USD: 0.001  (i.e. 1 mm → 0.001 m)")
    print(
        f"  Scaled dimensions (m): {xsize*0.001:.4f} × {ysize*0.001:.4f} × {zsize*0.001:.4f}"
    )
    print(
        f"  Scaled centroid (m)  : ({cx*0.001:+.5f}, {cy*0.001:+.5f}, {cz*0.001:+.5f})"
    )


if __name__ == "__main__":
    main()
