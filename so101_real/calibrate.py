"""calibrate.py — Checkerboard camera calibration for the real Arducam wrist camera.

Workflow
--------
Step 1: Capture frames
    python -m so101_real calibrate-camera \\
        --robot-config so101_real/configs/robot.yaml \\
        --out-dir so101_real/calibration/captures \\
        --board-cols 9 --board-rows 6 --square-mm 25

    Opens the camera in a live preview window.  Press SPACE to capture a frame
    (corner detection is run immediately; bad frames are rejected and shown in
    red).  Press Q or Ctrl-C to finish.  Aim for ~25 accepted frames covering
    all areas of the image.

Step 2: Solve intrinsics
    python -m so101_real calibrate-camera --solve \\
        --out-dir so101_real/calibration/captures \\
        --board-cols 9 --board-rows 6 --square-mm 25

    Runs cv2.calibrateCamera and writes:
        so101_real/configs/camera_intrinsics.yaml

    Prints per-image reprojection errors.  Target: < 0.5 px mean error.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# ── Capture ───────────────────────────────────────────────────────────────────


def run_capture(
    device_index: int,
    capture_width: int,
    capture_height: int,
    out_dir: Path,
    board_cols: int,
    board_rows: int,
) -> None:
    """Open camera and interactively capture calibration frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    board_size = (board_cols, board_rows)

    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    # Use MJPG for full-res capture
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {device_index}")

    # Flush startup frames
    for _ in range(5):
        cap.grab()

    accepted = 0
    frame_idx = 0
    print(
        f"[calibrate-camera] Board: {board_cols}×{board_rows} inner corners, "
        f"{board_size} pattern"
    )
    print("[calibrate-camera] SPACE=capture  Q=quit  (press Ctrl-C to abort)")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            if found:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, board_size, refined, found)
                status_color = (0, 220, 0)
                status = f"Board detected  |  accepted: {accepted}"
            else:
                status_color = (0, 60, 220)
                status = f"No board        |  accepted: {accepted}"

            cv2.putText(
                display,
                status,
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                2,
            )
            cv2.putText(
                display,
                "SPACE=capture  Q=quit",
                (12, display.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
            )
            cv2.imshow("calibrate-camera", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if not found:
                    print(
                        f"[calibrate-camera] Frame {frame_idx}: board NOT found — skipped"
                    )
                    # Flash red to indicate rejection
                    red = frame.copy()
                    red[:, :, :2] = 0
                    cv2.imshow("calibrate-camera", red)
                    cv2.waitKey(300)
                else:
                    out_path = out_dir / f"frame_{frame_idx:04d}.png"
                    cv2.imwrite(str(out_path), frame)
                    accepted += 1
                    frame_idx += 1
                    print(
                        f"[calibrate-camera] Frame {frame_idx - 1}: accepted "
                        f"→ {out_path.name}  (total: {accepted})"
                    )
                    # Flash green
                    green = frame.copy()
                    green[:, :, ::2] = 0
                    cv2.imshow("calibrate-camera", green)
                    cv2.waitKey(300)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n[calibrate-camera] Captured {accepted} accepted frames → {out_dir}")
    if accepted < 10:
        print(
            "[calibrate-camera] Warning: fewer than 10 frames captured. "
            "Aim for ~25 for a reliable calibration."
        )


# ── Solve ─────────────────────────────────────────────────────────────────────


def run_solve(
    out_dir: Path,
    board_cols: int,
    board_rows: int,
    square_mm: float,
    intrinsics_path: Path,
) -> None:
    """Load captured frames, run cv2.calibrateCamera, write camera_intrinsics.yaml."""
    board_size = (board_cols, board_rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    frame_paths = sorted(out_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frame_*.png files found in {out_dir}.\n"
            "Run without --solve first to capture frames."
        )

    print(
        f"[calibrate-camera] Solving with {len(frame_paths)} frames, "
        f"board {board_cols}×{board_rows}, square {square_mm} mm"
    )

    # 3-D object points in real-world coordinates (z=0 plane)
    obj_pts_template = np.zeros((board_rows * board_cols, 3), dtype=np.float32)
    obj_pts_template[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    obj_pts_template *= square_mm

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None  # (width, height)

    rejected = 0
    for path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [skip] Could not read {path.name}")
            rejected += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"  [skip] Board not found in {path.name}")
            rejected += 1
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(obj_pts_template)
        image_points.append(refined)

    n_used = len(object_points)
    print(f"  Used: {n_used} frames  |  Rejected: {rejected}")
    if n_used < 6:
        raise RuntimeError(
            f"Only {n_used} usable frames — need at least 6. Capture more frames."
        )

    # Run calibration
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    k1, k2, p1, p2, k3 = (float(v) for v in dist_coeffs.ravel()[:5])

    # Per-image reprojection error
    per_image_errors = []
    for i, (obj_p, img_p, rvec, tvec) in enumerate(
        zip(object_points, image_points, rvecs, tvecs)
    ):
        proj, _ = cv2.projectPoints(obj_p, rvec, tvec, camera_matrix, dist_coeffs)
        err = float(np.sqrt(np.mean((img_p - proj) ** 2)))
        per_image_errors.append(err)
        path_name = frame_paths[
            [p for p in sorted(out_dir.glob("frame_*.png"))].index(
                sorted(out_dir.glob("frame_*.png"))[i]
            )
        ].name
        print(f"  {path_name}: reprojection error = {err:.3f} px")

    mean_err = float(np.mean(per_image_errors))
    print(f"\n[calibrate-camera] Mean reprojection error: {mean_err:.4f} px")
    if mean_err > 0.5:
        print(
            "[calibrate-camera] WARNING: mean error > 0.5 px — calibration quality "
            "may be poor. Consider recapturing with better coverage or a flatter board."
        )

    # Sanity check: diagonal FOV.
    # The Arducam B0200 (IMX291, 1/2.8") lens is rated 100° diagonal, but at
    # 1920×1080 the sensor uses a cropped readout window, giving ~80° effective
    # diagonal FOV.  Accept 70–115° to cover both full-sensor and cropped modes.
    w, h = image_size
    diag_px = float(np.sqrt(w**2 + h**2))
    fov_diag_deg = float(2 * np.degrees(np.arctan(diag_px / (2 * np.sqrt(fx * fy)))))
    print(f"[calibrate-camera] Diagonal FOV (from intrinsics): {fov_diag_deg:.1f}°")
    if not (70.0 <= fov_diag_deg <= 115.0):
        print(
            f"[calibrate-camera] WARNING: diagonal FOV {fov_diag_deg:.1f}° is outside "
            "the expected 70–115° range for the Arducam B0200. "
            "Check board size and square-mm arguments."
        )

    cx_offset_pct = abs(cx - w / 2) / (w / 2) * 100
    cy_offset_pct = abs(cy - h / 2) / (h / 2) * 100
    if cx_offset_pct > 5.0 or cy_offset_pct > 5.0:
        print(
            f"[calibrate-camera] WARNING: principal point ({cx:.1f}, {cy:.1f}) is "
            f"{cx_offset_pct:.1f}% / {cy_offset_pct:.1f}% off image centre "
            f"({w/2:.1f}, {h/2:.1f}). Isaac Sim assumes a centred principal point — "
            "the x/y offset will be dropped during sim conversion, adding some error."
        )

    intrinsics = {
        "image_width": int(w),
        "image_height": int(h),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3,
        "mean_reprojection_error_px": mean_err,
        "n_frames_used": n_used,
        "board_cols": board_cols,
        "board_rows": board_rows,
        "square_mm": float(square_mm),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(intrinsics_path, "w") as f:
        yaml.dump(intrinsics, f, default_flow_style=False, sort_keys=False)

    print(f"\n[calibrate-camera] Intrinsics written → {intrinsics_path}")
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"  k1={k1:.6f}  k2={k2:.6f}  p1={p1:.6f}  p2={p2:.6f}  k3={k3:.6f}")
