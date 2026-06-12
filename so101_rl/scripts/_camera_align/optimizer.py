"""ORB feature matching + Nelder-Mead extrinsic optimizer.

Given a callable that renders the sim wrist camera at a candidate
``CameraXframe`` mount transform (translate, orient_wxyz) and returns a BGR
uint8 image *post-image-pipeline* (i.e. the same view the policy consumes),
this module fits the 6-DoF perturbation that minimizes ORB feature
reprojection error against a reference real frame.

Loss
----
    loss(x) = mean_reprojection_error_px + lambda * (1 - inlier_ratio)

where ``x`` is a 6-vector ``(dtx, dty, dtz, drx, dry, drz)`` of small
perturbations applied around the *initial* transform supplied to
``optimize()``. Translation perturbations are in metres, rotation
perturbations in degrees, applied as an extrinsic XYZ rotation composed with
the initial orientation.

Failure modes are explicit: if fewer than ``min_matches`` ORB matches survive
the Lowe ratio test on the *initial* transform, ``optimize()`` raises
``InsufficientFeaturesError`` rather than running a meaningless search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InsufficientFeaturesError(RuntimeError):
    """Raised when the reference frame or the initial sim render do not yield
    enough ORB matches to drive the optimizer."""


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (no scipy.spatial dependency required)
# ---------------------------------------------------------------------------


def _quat_wxyz_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    n = float(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        raise ValueError("quaternion has zero norm")
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(m: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to (w, x, y, z) — Shoemake's algorithm."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def _euler_xyz_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def perturb_transform(
    base_translate: tuple[float, float, float],
    base_orient_wxyz: tuple[float, float, float, float],
    delta: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Apply ``delta = (dtx, dty, dtz, drx_deg, dry_deg, drz_deg)`` to the base.

    Translations are added in the world frame (matches how the gizmo edits
    the CameraXframe). Rotation perturbation is composed *before* the base
    rotation: ``R_new = R_base @ R_delta`` so small deltas behave as
    intuitive local-frame nudges of the camera.
    """
    dtx, dty, dtz, drx, dry, drz = (float(v) for v in delta)
    tx, ty, tz = base_translate
    new_translate = (tx + dtx, ty + dty, tz + dtz)

    R_base = _quat_wxyz_to_matrix(base_orient_wxyz)
    R_delta = _euler_xyz_to_matrix(drx, dry, drz)
    new_orient = _matrix_to_quat_wxyz(R_base @ R_delta)
    return new_translate, new_orient


# ---------------------------------------------------------------------------
# ORB matcher
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    mean_reprojection_error_px: float
    inlier_ratio: float
    n_matches: int
    n_inliers: int


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


class OrbMatcher:
    """Pre-computes ORB features on a reference image; matches against
    candidate sim renders using BFMatcher + Lowe ratio + homography RANSAC
    inlier filtering. Loss = mean reprojection error among inliers, with a
    soft penalty on (1 - inlier_ratio).

    Resolution: both real and sim must be the same shape (H, W). Caller is
    responsible for ensuring this; the optimizer will assert it.
    """

    def __init__(
        self,
        real_bgr: np.ndarray,
        nfeatures: int = 500,
        ratio: float = 0.75,
        ransac_reproj_threshold_px: float = 3.0,
    ) -> None:
        self.real_bgr = real_bgr
        self.real_shape = real_bgr.shape[:2]
        self.ratio = float(ratio)
        self.ransac_reproj_threshold_px = float(ransac_reproj_threshold_px)
        self._orb = cv2.ORB_create(nfeatures=int(nfeatures))
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        gray = _to_gray(real_bgr)
        self.real_kp, self.real_des = self._orb.detectAndCompute(gray, None)
        if self.real_des is None or len(self.real_kp) < 8:
            raise InsufficientFeaturesError(
                f"Reference real frame yielded only "
                f"{0 if self.real_des is None else len(self.real_kp)} ORB "
                "keypoints (need >=8). Add texture to the calibration scene."
            )

    def match(self, sim_bgr: np.ndarray) -> MatchResult:
        if sim_bgr.shape[:2] != self.real_shape:
            raise ValueError(
                f"sim render shape {sim_bgr.shape[:2]} != real shape "
                f"{self.real_shape}; resize before matching."
            )
        gray = _to_gray(sim_bgr)
        sim_kp, sim_des = self._orb.detectAndCompute(gray, None)
        if sim_des is None or len(sim_kp) < 8:
            return MatchResult(
                mean_reprojection_error_px=float("inf"),
                inlier_ratio=0.0,
                n_matches=0,
                n_inliers=0,
            )

        # KNN match real → sim with Lowe ratio test.
        knn = self._matcher.knnMatch(self.real_des, sim_des, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        if len(good) < 8:
            return MatchResult(
                mean_reprojection_error_px=float("inf"),
                inlier_ratio=0.0,
                n_matches=len(good),
                n_inliers=0,
            )

        src = np.float32([self.real_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([sim_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src, dst, cv2.RANSAC, self.ransac_reproj_threshold_px
        )
        if H is None or mask is None:
            return MatchResult(
                mean_reprojection_error_px=float("inf"),
                inlier_ratio=0.0,
                n_matches=len(good),
                n_inliers=0,
            )

        mask = mask.ravel().astype(bool)
        n_inliers = int(mask.sum())
        if n_inliers < 4:
            return MatchResult(
                mean_reprojection_error_px=float("inf"),
                inlier_ratio=n_inliers / max(1, len(good)),
                n_matches=len(good),
                n_inliers=n_inliers,
            )

        projected = cv2.perspectiveTransform(src[mask], H).reshape(-1, 2)
        target = dst[mask].reshape(-1, 2)
        err = np.linalg.norm(projected - target, axis=1)
        return MatchResult(
            mean_reprojection_error_px=float(err.mean()),
            inlier_ratio=n_inliers / len(good),
            n_matches=len(good),
            n_inliers=n_inliers,
        )


# ---------------------------------------------------------------------------
# Nelder-Mead optimizer
# ---------------------------------------------------------------------------


@dataclass
class OptimizeResult:
    success: bool
    n_evals: int
    final_loss: float
    final_match: MatchResult
    best_delta: np.ndarray  # 6-vector
    best_translate: tuple[float, float, float]
    best_orient_wxyz: tuple[float, float, float, float]


def optimize(
    matcher: OrbMatcher,
    render_fn: Callable[
        [tuple[float, float, float], tuple[float, float, float, float]], np.ndarray
    ],
    base_translate: tuple[float, float, float],
    base_orient_wxyz: tuple[float, float, float, float],
    *,
    translation_bound_m: float = 0.005,
    rotation_bound_deg: float = 5.0,
    max_evals: int = 80,
    inlier_penalty_weight: float = 5.0,
    min_matches: int = 20,
    progress_cb: Callable[[int, float, MatchResult], None] | None = None,
) -> OptimizeResult:
    """Nelder-Mead optimization of the 6-DoF mount perturbation.

    Parameters
    ----------
    matcher:
        Pre-built ORB matcher seeded with the real reference frame.
    render_fn:
        Callable that, given (translate, orient_wxyz), writes the USD prim,
        steps the sim, runs the deploy image pipeline, and returns the
        post-pipeline BGR uint8 image at the real frame's resolution.
    base_translate, base_orient_wxyz:
        Current camera mount transform; the optimizer searches a small
        neighborhood around this.
    translation_bound_m, rotation_bound_deg:
        Hard bounds applied to the perturbation by clipping inside the loss;
        keeps the search local and reversible.
    max_evals:
        Caps the number of ``render_fn`` calls (NM + initial sanity check).
    inlier_penalty_weight:
        ``lambda`` in ``loss = mean_err + lambda * (1 - inlier_ratio)``.
    min_matches:
        Minimum number of Lowe-good matches required at the *initial*
        transform; below this we refuse to run the search.
    progress_cb:
        Optional ``(eval_index, loss, match) -> None`` callback for HUD updates.

    Returns
    -------
    OptimizeResult
    """
    # scipy is part of the Isaac Lab env; import here to keep the module
    # importable in test contexts without scipy.
    from scipy.optimize import minimize

    def _loss_for_delta(delta: np.ndarray) -> tuple[float, MatchResult]:
        # Hard clip inside the bounding box; treat clamped requests as the
        # boundary point rather than rejecting them outright (NM handles
        # this gracefully).
        clipped = delta.copy()
        clipped[:3] = np.clip(clipped[:3], -translation_bound_m, translation_bound_m)
        clipped[3:] = np.clip(clipped[3:], -rotation_bound_deg, rotation_bound_deg)
        translate, orient = perturb_transform(
            base_translate, base_orient_wxyz, clipped
        )
        sim_bgr = render_fn(translate, orient)
        match = matcher.match(sim_bgr)
        if not np.isfinite(match.mean_reprojection_error_px):
            return 1e6, match
        return (
            match.mean_reprojection_error_px
            + inlier_penalty_weight * (1.0 - match.inlier_ratio)
        ), match

    state = {"n_evals": 0, "best_loss": float("inf"), "best_delta": np.zeros(6),
             "best_match": None}

    # Sanity check at delta=0 before launching NM.
    initial_loss, initial_match = _loss_for_delta(np.zeros(6))
    state["n_evals"] += 1
    if initial_match.n_matches < min_matches:
        raise InsufficientFeaturesError(
            f"Only {initial_match.n_matches} ORB matches at initial transform "
            f"(need >= {min_matches}).  "
            f"Real image has {len(matcher.real_kp)} keypoints.  "
            f"Possible causes: (1) sim and real images look too different — "
            f"check the 3-panel display before optimising; "
            f"(2) scene has no texture — add the cube or a checkerboard target; "
            f"(3) image resolution too low — increase --render-width/--render-height."
        )
    state["best_loss"] = initial_loss
    state["best_match"] = initial_match
    if progress_cb is not None:
        progress_cb(state["n_evals"], initial_loss, initial_match)

    def _objective(delta: np.ndarray) -> float:
        if state["n_evals"] >= max_evals:
            # Returning a large value keeps NM from crashing while we
            # short-circuit. We re-check max_evals after .minimize() anyway.
            return state["best_loss"] + 1.0
        loss, match = _loss_for_delta(delta)
        state["n_evals"] += 1
        if loss < state["best_loss"]:
            state["best_loss"] = loss
            state["best_delta"] = delta.copy()
            state["best_match"] = match
        if progress_cb is not None:
            progress_cb(state["n_evals"], loss, match)
        return loss

    # Initial simplex: small steps along each axis. Translation step is 1mm,
    # rotation step is 1deg — well inside the bounds.
    x0 = np.zeros(6)
    initial_simplex = np.vstack([x0, x0 + np.diag([1e-3, 1e-3, 1e-3, 1.0, 1.0, 1.0])])
    minimize(
        _objective,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": initial_simplex,
            "maxiter": max_evals,
            "maxfev": max_evals,
            "xatol": 1e-4,
            "fatol": 1e-3,
            "adaptive": True,
        },
    )

    best_delta = state["best_delta"]
    best_translate, best_orient = perturb_transform(
        base_translate, base_orient_wxyz, best_delta
    )
    return OptimizeResult(
        success=state["best_match"] is not None
        and np.isfinite(state["best_match"].mean_reprojection_error_px),
        n_evals=state["n_evals"],
        final_loss=state["best_loss"],
        final_match=state["best_match"],
        best_delta=best_delta,
        best_translate=best_translate,
        best_orient_wxyz=best_orient,
    )
