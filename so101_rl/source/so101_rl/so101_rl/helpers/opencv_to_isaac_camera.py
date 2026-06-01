"""
Convert OpenCV camera calibration output to Isaac Sim FisheyeCameraCfg parameters
for the ``fisheyeRadTanThinPrism`` projection model.

Usage (library)
---------------
>>> from so101_rl.helpers.opencv_to_isaac_camera import load_intrinsics, opencv_to_isaac_fisheye_radtan
>>> intrinsics = load_intrinsics("so101_real/configs/camera_intrinsics.yaml")
>>> params = opencv_to_isaac_fisheye_radtan(intrinsics)
>>> print(params["fisheye_cfg"])       # kwargs for FisheyeCameraCfg
>>> print(params["post_spawn_usd_attrs"])  # extra attrs needing prim.GetAttribute(...).Set(...)

Usage (CLI — run sanity checks)
---------------------------------
    python -m so101_rl.helpers.opencv_to_isaac_camera \
        --intrinsics so101_real/configs/camera_intrinsics.yaml

Isaac Sim camera model notes
-----------------------------
FisheyeCameraCfg (IsaacLab) exposes the following parameters that map to
``fisheyeRadTanThinPrism``:

  focal_length [mm]          — together with horizontal_aperture defines fx:
                               fx_px = focal_length * nominal_width / horizontal_aperture
  horizontal_aperture [mm]   — physical sensor width (or chosen to satisfy above)
  fisheye_nominal_width/height  — rendered image pixel dimensions
  fisheye_optical_centre_x/y    — principal point cx, cy  (pixels)
  fisheye_max_fov               — max full-frame FOV (degrees); set with 20% margin
  fisheye_polynomial_a = k1  — radial distortion coefficients
  fisheye_polynomial_b = k2
  fisheye_polynomial_c = k3
  fisheye_polynomial_d = k4 = 0  (OpenCV 5-coeff calibration yields k4=k5=k6=0)
  fisheye_polynomial_e = k5 = 0
  fisheye_polynomial_f = k6 = 0

Parameters NOT exposed in FisheyeCameraCfg but needed for full accuracy:

  USD attr ``openCVFx``  = fx (pixels) — redundant if focal_length/aperture are set
  USD attr ``openCVFy``  = fy (pixels) — see above
  USD attr ``p0``        = OpenCV p1   — tangential distortion coefficient 1
  USD attr ``p1``        = OpenCV p2   — tangential distortion coefficient 2
  USD attr ``s0``–``s3`` = 0           — thin-prism (zero for standard calibration)

These must be applied via ``prim.GetAttribute(name).Set(value)`` after the camera
prim has been spawned.  A helper ``apply_post_spawn_attrs`` is provided.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Sony IMX291 / Arducam B0200 physical sensor parameters (default).
# Pixel pitch from IMX291 datasheet: 2.9 μm.
# ---------------------------------------------------------------------------
IMX291_PIXEL_PITCH_MM: float = 0.0029


def load_intrinsics(path: str | Path) -> dict[str, Any]:
    """Load ``camera_intrinsics.yaml`` written by ``so101_real.calibrate.run_solve``.

    Required keys: ``fx``, ``fy``, ``cx``, ``cy``, ``k1``, ``k2``, ``p1``,
    ``p2``, ``k3``, ``image_width``, ``image_height``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh)

    required = {
        "fx",
        "fy",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
        "k3",
        "image_width",
        "image_height",
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"camera_intrinsics.yaml is missing required keys: {missing}")
    return data


def opencv_to_isaac_fisheye_radtan(
    intrinsics: dict[str, Any],
    sensor_pixel_pitch_mm: float = IMX291_PIXEL_PITCH_MM,
    fov_margin_factor: float = 1.2,
) -> dict[str, Any]:
    """Convert OpenCV calibration intrinsics to Isaac Sim ``FisheyeCameraCfg``
    parameters for the ``fisheyeRadTanThinPrism`` projection model.

    Args:
        intrinsics: dict produced by :func:`load_intrinsics` (or equivalent).
        sensor_pixel_pitch_mm: Physical size of one pixel in mm.
            Default 0.0029 mm = Sony IMX291 / Arducam B0200.
        fov_margin_factor: ``fisheye_max_fov`` is set to
            ``diagonal_fov_deg * fov_margin_factor``.  Ensure the margin keeps the
            full image within the model's valid cone.

    Returns:
        dict with:

        ``"fisheye_cfg"``
            Keyword arguments accepted by ``sim_utils.FisheyeCameraCfg``.

        ``"post_spawn_usd_attrs"``
            ``{usd_attr_name: value}`` dict of parameters that are *not* exposed
            by ``FisheyeCameraCfg`` and must be applied to the camera prim via
            ``prim.GetAttribute(name).Set(value)`` after spawning.

        ``"diagnostics"``
            Derived quantities useful for sanity-checking (FOV angles, roundtrip
            fx/fy, reprojection error metadata).
    """
    fx: float = float(intrinsics["fx"])
    fy: float = float(intrinsics["fy"])
    cx: float = float(intrinsics["cx"])
    cy: float = float(intrinsics["cy"])
    k1: float = float(intrinsics["k1"])
    k2: float = float(intrinsics["k2"])
    k3: float = float(intrinsics["k3"])
    p1: float = float(intrinsics["p1"])  # OpenCV tangential coeff 1
    p2: float = float(intrinsics["p2"])  # OpenCV tangential coeff 2
    W: int = int(intrinsics["image_width"])
    H: int = int(intrinsics["image_height"])

    # ------------------------------------------------------------------
    # Focal length + horizontal aperture
    # Invariant: fx_px = focal_length_mm * W / horizontal_aperture_mm
    # ------------------------------------------------------------------
    horizontal_aperture_mm: float = W * sensor_pixel_pitch_mm
    focal_length_mm: float = fx * sensor_pixel_pitch_mm  # = fx * ha / W

    # Sanity: roundtrip
    fx_check = focal_length_mm * W / horizontal_aperture_mm
    assert (
        abs(fx_check - fx) < 1e-6
    ), f"focal_length roundtrip failed: got {fx_check:.6f}, expected {fx:.6f}"

    # ------------------------------------------------------------------
    # Diagonal full-frame FOV (using geometric mean focal length for
    # a slight anisotropy in fx/fy)
    # ------------------------------------------------------------------
    f_mean = math.sqrt(fx * fy)
    half_diag_px = math.sqrt((W / 2.0) ** 2 + (H / 2.0) ** 2)
    half_diag_fov_rad = math.atan2(half_diag_px, f_mean)
    diag_fov_deg = math.degrees(2.0 * half_diag_fov_rad)

    fov_x_deg = math.degrees(2.0 * math.atan(W / (2.0 * fx)))
    fov_y_deg = math.degrees(2.0 * math.atan(H / (2.0 * fy)))
    fisheye_max_fov = diag_fov_deg * fov_margin_factor

    # ------------------------------------------------------------------
    # FisheyeCameraCfg kwargs
    # ------------------------------------------------------------------
    fisheye_cfg: dict[str, Any] = dict(
        projection_type="fisheyeRadTanThinPrism",
        focal_length=focal_length_mm,
        horizontal_aperture=horizontal_aperture_mm,
        fisheye_nominal_width=float(W),
        fisheye_nominal_height=float(H),
        fisheye_optical_centre_x=cx,
        fisheye_optical_centre_y=cy,
        fisheye_max_fov=fisheye_max_fov,
        # Radial distortion (k4=k5=k6=0 for standard OpenCV calibrateCamera)
        fisheye_polynomial_a=k1,
        fisheye_polynomial_b=k2,
        fisheye_polynomial_c=k3,
        fisheye_polynomial_d=0.0,
        fisheye_polynomial_e=0.0,
        fisheye_polynomial_f=0.0,
    )

    # ------------------------------------------------------------------
    # Extra USD attributes not exposed in FisheyeCameraCfg.
    # Apply via  prim.GetAttribute(name).Set(value)  after spawning.
    # ------------------------------------------------------------------
    post_spawn_usd_attrs: dict[str, float] = {
        # Focal lengths in pixels (may be read by some Isaac Sim renderers
        # to override the pinhole focal_length/horizontal_aperture values)
        "openCVFx": fx,
        "openCVFy": fy,
        # Tangential distortion (OpenCV p1 → USD p0, OpenCV p2 → USD p1)
        "p0": p1,
        "p1": p2,
        # Thin-prism (zero for standard calibration)
        "s0": 0.0,
        "s1": 0.0,
        "s2": 0.0,
        "s3": 0.0,
    }

    diagnostics: dict[str, Any] = dict(
        fov_x_deg=fov_x_deg,
        fov_y_deg=fov_y_deg,
        diag_fov_deg=diag_fov_deg,
        fisheye_max_fov=fisheye_max_fov,
        horizontal_aperture_mm=horizontal_aperture_mm,
        focal_length_mm=focal_length_mm,
        fx_roundtrip=fx_check,
        n_frames_used=intrinsics.get("n_frames_used"),
        mean_reprojection_error_px=intrinsics.get("mean_reprojection_error_px"),
    )

    return dict(
        fisheye_cfg=fisheye_cfg,
        post_spawn_usd_attrs=post_spawn_usd_attrs,
        diagnostics=diagnostics,
    )


def opencv_to_isaac_pinhole(
    intrinsics: dict[str, Any],
    sensor_pixel_pitch_mm: float = IMX291_PIXEL_PITCH_MM,
) -> dict[str, Any]:
    """Convert OpenCV ``calibrateCamera`` intrinsics to Isaac Sim
    ``PinholeCameraCfg`` parameters.

    Use this for cameras calibrated with :func:`cv2.calibrateCamera` (the
    standard pinhole + Brown-Conrady distortion model).  The rendered image
    fills the full rectangular frame — there is no circular fisheye vignette.

    Note: Isaac Sim's ``PinholeCameraCfg`` does not model distortion (k1/k2/
    k3/p1/p2) in the render path.  For sim-to-real transfer the dominant
    correction is getting the FOV and principal-point offset right; the
    residual distortion can be absorbed by the CNN.

    Args:
        intrinsics: dict produced by :func:`load_intrinsics`.
        sensor_pixel_pitch_mm: Physical size of one pixel in mm.

    Returns:
        dict with:

        ``"pinhole_cfg"``
            Keyword arguments accepted by ``sim_utils.PinholeCameraCfg``.

        ``"diagnostics"``
            Derived quantities for sanity-checking.
    """
    fx: float = float(intrinsics["fx"])
    fy: float = float(intrinsics["fy"])
    cx: float = float(intrinsics["cx"])
    cy: float = float(intrinsics["cy"])
    W: int = int(intrinsics["image_width"])
    H: int = int(intrinsics["image_height"])

    # Sensor dimensions in mm
    horizontal_aperture_mm: float = W * sensor_pixel_pitch_mm
    vertical_aperture_mm: float = H * sensor_pixel_pitch_mm

    # Focal length: use fx. Since fx ≈ fy for a well-calibrated sensor, a
    # single value captures both axes; the slight difference is negligible.
    focal_length_mm: float = fx * sensor_pixel_pitch_mm

    # Principal point offset from sensor centre (positive = right / down).
    # USD convention: horizontal_aperture_offset shifts cx; sign matches OpenCV.
    horizontal_aperture_offset_mm: float = (cx - W / 2.0) * sensor_pixel_pitch_mm
    vertical_aperture_offset_mm: float = (cy - H / 2.0) * sensor_pixel_pitch_mm

    fov_x_deg = math.degrees(2.0 * math.atan(W / (2.0 * fx)))
    fov_y_deg = math.degrees(2.0 * math.atan(H / (2.0 * fy)))
    diag_fov_deg = math.degrees(
        2.0 * math.atan(math.sqrt((W / 2.0) ** 2 + (H / 2.0) ** 2) / math.sqrt(fx * fy))
    )

    pinhole_cfg: dict[str, Any] = dict(
        focal_length=focal_length_mm,
        horizontal_aperture=horizontal_aperture_mm,
        vertical_aperture=vertical_aperture_mm,
        horizontal_aperture_offset=horizontal_aperture_offset_mm,
        vertical_aperture_offset=vertical_aperture_offset_mm,
    )

    diagnostics: dict[str, Any] = dict(
        fov_x_deg=fov_x_deg,
        fov_y_deg=fov_y_deg,
        diag_fov_deg=diag_fov_deg,
        focal_length_mm=focal_length_mm,
        horizontal_aperture_mm=horizontal_aperture_mm,
        vertical_aperture_mm=vertical_aperture_mm,
        horizontal_aperture_offset_mm=horizontal_aperture_offset_mm,
        vertical_aperture_offset_mm=vertical_aperture_offset_mm,
        n_frames_used=intrinsics.get("n_frames_used"),
        mean_reprojection_error_px=intrinsics.get("mean_reprojection_error_px"),
    )

    return dict(pinhole_cfg=pinhole_cfg, diagnostics=diagnostics)


def write_isaac_camera_params(
    params: dict[str, Any],
    out_path: str | Path,
) -> None:
    """Serialise conversion output to YAML for inspection and reproducibility."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        fisheye_cfg=params["fisheye_cfg"],
        post_spawn_usd_attrs=params["post_spawn_usd_attrs"],
        diagnostics=params["diagnostics"],
    )
    with out_path.open("w") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=False)
    print(f"Isaac camera params written to: {out_path}")


def apply_post_spawn_attrs(prim, post_spawn_usd_attrs: dict[str, float]) -> None:
    """Set extra USD attributes on a camera prim after it has been spawned.

    Call this once the camera USD prim exists (e.g. after ``env.reset()`` or
    inside ``post_physics_step``).

    Args:
        prim: ``pxr.Usd.Prim`` of the camera.
        post_spawn_usd_attrs: ``{usd_attr_name: value}`` from
            :func:`opencv_to_isaac_fisheye_radtan`.
    """
    from pxr import Sdf  # import here to avoid hard dep at module load time

    for attr_name, value in post_spawn_usd_attrs.items():
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(float(value))
        else:
            prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float).Set(float(value))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _run_self_test() -> None:
    """Verify the conversion roundtrip with synthetic intrinsics."""

    # Synthetic calibration: 1920×1080, fx=fy=900, cx=960, cy=540, k1=-0.3, k2=0.1
    intrinsics = dict(
        image_width=1920,
        image_height=1080,
        fx=900.0,
        fy=900.0,
        cx=960.0,
        cy=540.0,
        k1=-0.30,
        k2=0.10,
        k3=-0.02,
        p1=0.001,
        p2=-0.0005,
    )

    params = opencv_to_isaac_fisheye_radtan(intrinsics, sensor_pixel_pitch_mm=0.0029)
    cfg = params["fisheye_cfg"]
    usd = params["post_spawn_usd_attrs"]
    diag = params["diagnostics"]

    # 1. Focal-length roundtrip: fx == focal_length * W / h_aperture
    fx_roundtrip = (
        cfg["focal_length"] * intrinsics["image_width"] / cfg["horizontal_aperture"]
    )
    assert (
        abs(fx_roundtrip - intrinsics["fx"]) < 1e-6
    ), f"fx roundtrip: expected {intrinsics['fx']}, got {fx_roundtrip}"

    # 2. Physical horizontal aperture equals image_width * pixel_pitch
    expected_ha = intrinsics["image_width"] * 0.0029
    assert abs(cfg["horizontal_aperture"] - expected_ha) < 1e-9

    # 3. Radial distortion coefficients pass through unchanged
    assert cfg["fisheye_polynomial_a"] == intrinsics["k1"]
    assert cfg["fisheye_polynomial_b"] == intrinsics["k2"]
    assert cfg["fisheye_polynomial_c"] == intrinsics["k3"]
    assert cfg["fisheye_polynomial_d"] == 0.0
    assert cfg["fisheye_polynomial_e"] == 0.0
    assert cfg["fisheye_polynomial_f"] == 0.0

    # 4. Tangential distortion in post-spawn attrs (note index swap: OpenCV p1→USD p0)
    assert usd["p0"] == intrinsics["p1"]
    assert usd["p1"] == intrinsics["p2"]

    # 5. Post-spawn openCVFx/Fy pass through unchanged
    assert usd["openCVFx"] == intrinsics["fx"]
    assert usd["openCVFy"] == intrinsics["fy"]

    # 6. Principal point preserved
    assert cfg["fisheye_optical_centre_x"] == intrinsics["cx"]
    assert cfg["fisheye_optical_centre_y"] == intrinsics["cy"]

    # 7. max_fov > diagonal_fov (margin is applied)
    fov_diag = diag["diag_fov_deg"]
    assert (
        cfg["fisheye_max_fov"] > fov_diag
    ), f"max_fov {cfg['fisheye_max_fov']:.2f}° should exceed diag_fov {fov_diag:.2f}°"

    # 8. FOV is in a sane range for a wide-angle wrist camera (50–150°)
    assert (
        50.0 < fov_diag < 150.0
    ), f"Diagonal FOV {fov_diag:.2f}° outside expected 50–150° range"

    print("All self-tests passed.")
    print(f"  fx roundtrip error: {abs(fx_roundtrip - intrinsics['fx']):.2e} px")
    print(
        f"  Diagonal FOV: {fov_diag:.2f}°  → fisheye_max_fov: {cfg['fisheye_max_fov']:.2f}°"
    )
    print(
        f"  focal_length: {cfg['focal_length']:.4f} mm  horizontal_aperture: {cfg['horizontal_aperture']:.4f} mm"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Convert OpenCV camera intrinsics to Isaac Sim FisheyeCameraCfg params."
    )
    parser.add_argument(
        "--intrinsics",
        required=False,
        help="Path to camera_intrinsics.yaml produced by 'python -m so101_real calibrate-camera --solve'.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional: write converted params to this YAML file.",
    )
    parser.add_argument(
        "--pixel-pitch",
        type=float,
        default=IMX291_PIXEL_PITCH_MM,
        help=f"Sensor pixel pitch in mm (default: {IMX291_PIXEL_PITCH_MM} mm, Sony IMX291 / Arducam B0200).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-tests with synthetic intrinsics and exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        sys.exit(0)

    if args.intrinsics is None:
        parser.error("--intrinsics is required unless --self-test is used.")

    intrinsics = load_intrinsics(args.intrinsics)
    params = opencv_to_isaac_fisheye_radtan(
        intrinsics, sensor_pixel_pitch_mm=args.pixel_pitch
    )

    diag = params["diagnostics"]
    cfg = params["fisheye_cfg"]
    usd = params["post_spawn_usd_attrs"]

    print("\n=== Diagnostics ===")
    print(
        f"  FOV x / y / diag:   {diag['fov_x_deg']:.1f}° / {diag['fov_y_deg']:.1f}° / {diag['diag_fov_deg']:.1f}°"
    )
    print(f"  fisheye_max_fov:     {diag['fisheye_max_fov']:.1f}°")
    print(f"  focal_length:        {diag['focal_length_mm']:.4f} mm")
    print(f"  horizontal_aperture: {diag['horizontal_aperture_mm']:.4f} mm")
    if diag["mean_reprojection_error_px"] is not None:
        print(f"  Reprojection error:  {diag['mean_reprojection_error_px']:.4f} px")
    if diag["n_frames_used"] is not None:
        print(f"  Calibration frames:  {diag['n_frames_used']}")

    print("\n=== FisheyeCameraCfg kwargs ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    print(
        "\n=== Post-spawn USD attributes (set via prim.GetAttribute(name).Set(value)) ==="
    )
    for k, v in usd.items():
        print(f"  {k}: {v}")

    if args.out:
        write_isaac_camera_params(params, args.out)
