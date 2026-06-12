"""calibrate_sim_joints.py — Empirically measure SO-101 sim joint bounds.

Loads the robot USD into a minimal Isaac Sim scene (no table, no cube, no
lights), then for each joint drives it to its PhysX-declared lower and upper
limit and records the settled position.

Three sets of bounds are compared and printed:

  1. **robot.yaml** — the values in ``robot.yaml::joint_limits`` (real-robot
     calibration reference; what ``norm`` conversion uses on the real side).
  2. **PhysX (USD)** — what ``robot.data.joint_pos_limits`` reports after
     scene initialisation (directly from USD physics properties).
  3. **Settled** — what the joint position actually reads after commanding
     the PhysX limit and waiting for the controller to settle.

If settled ≈ PhysX ≈ robot.yaml the bounds source is *not* the cause of the
gripper-clipping issue.  Any column with |Δ| > 0.5° gets a WARNING line.

Output
------
A YAML file (default ``so101_rl/configs/sim_joint_limits.yaml``) in the same
schema as ``robot.yaml::joint_limits`` so it can be used directly for norm
conversion in sim-side tooling.

Usage
-----
    $ISAAC_LAB_PATH/isaaclab.sh -p so101_rl/scripts/calibrate_sim_joints.py --headless

    # Also compare against robot.yaml bounds:
    $ISAAC_LAB_PATH/isaaclab.sh -p so101_rl/scripts/calibrate_sim_joints.py \\
        --headless \\
        --robot-config so101_real/configs/robot.yaml \\
        --out so101_rl/configs/sim_joint_limits.yaml
"""

# ---------------------------------------------------------------------------
# IMPORTANT: AppLauncher must be first.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Empirically measure SO-101 sim joint bounds and compare with "
    "robot.yaml and USD/PhysX values.",
)
parser.add_argument(
    "--robot-config",
    default=None,
    dest="robot_config",
    metavar="PATH",
    help="Path to so101_real robot config YAML.  When supplied, robot.yaml "
    "joint_limits are included as a third comparison column.",
)
parser.add_argument(
    "--out",
    default=None,
    dest="out",
    metavar="PATH",
    help="Write sim_joint_limits.yaml to this path.  "
    "Defaults to so101_rl/configs/sim_joint_limits.yaml next to this script.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=300,
    dest="settle_steps",
    help="Maximum sim steps to wait for a joint to settle at each limit "
    "(default: 300 = 5 s at 60 Hz).",
)
parser.add_argument(
    "--settle-tol-rad",
    type=float,
    default=0.001,
    dest="settle_tol_rad",
    help="Convergence threshold: joint is 'settled' when |q - target| < tol "
    "for --settle-consec consecutive steps (default: 0.001 rad ≈ 0.06°).",
)
parser.add_argument(
    "--settle-consec",
    type=int,
    default=10,
    dest="settle_consec",
    help="Number of consecutive steps within tolerance before declaring "
    "settled (default: 10).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------

import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

_project_root = Path(__file__).resolve().parents[2]
if "ISAAC_LAB_WORKSPACE_PATH" not in os.environ:
    os.environ["ISAAC_LAB_WORKSPACE_PATH"] = str(_project_root)

from so101_rl.configurations.so101 import SO101_CFG  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

_WARN_THRESH_DEG = 0.5  # flag differences larger than this


# ---------------------------------------------------------------------------
# Minimal scene — robot only
# ---------------------------------------------------------------------------


@configclass
class CalibSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore


# ---------------------------------------------------------------------------
# Settle helper
# ---------------------------------------------------------------------------


def _settle_to_target(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    joint_indices,
    target_rad: torch.Tensor,
    max_steps: int,
    tol_rad: float,
    consec_needed: int,
    sim_device: torch.device,
) -> tuple[float, bool]:
    """Command ``target_rad`` and step until settled or timeout.

    Returns (settled_value_rad, did_converge).
    """
    q_target = target_rad.unsqueeze(0).to(sim_device)  # (1,1)
    consec = 0
    q_settled = float(target_rad)

    for _ in range(max_steps):
        robot.write_joint_position_to_sim(q_target, joint_ids=joint_indices)
        robot.write_joint_velocity_to_sim(torch.zeros_like(q_target), joint_ids=joint_indices)
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        q_now = float(robot.data.joint_pos[0, joint_indices[0]])
        q_settled = q_now
        if abs(q_now - float(target_rad)) < tol_rad:
            consec += 1
            if consec >= consec_needed:
                return q_settled, True
        else:
            consec = 0

    return q_settled, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ── Optional robot.yaml bounds ──────────────────────────────────────────
    robot_yaml_limits: dict[str, tuple[float, float]] | None = None
    if args_cli.robot_config is not None:
        try:
            from so101_real.robot import RobotConfig  # noqa: PLC0415

            rcfg = RobotConfig.load(args_cli.robot_config)
            robot_yaml_limits = {
                name: (entry.lower_rad, entry.upper_rad)
                for name, entry in rcfg.joint_limits.items()
                if name in _JOINT_NAMES
            }
            print(f"[calib] Loaded robot.yaml joint_limits from {args_cli.robot_config}")
        except Exception as exc:
            print(f"[calib] WARNING: could not load robot.yaml joint_limits: {exc}")

    # ── Sim setup ───────────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1)
    sim = SimulationContext(sim_cfg)

    scene_cfg = CalibSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[calib] Scene ready.")

    robot: Articulation = scene["robot"]
    sim_device = sim.device

    # ── PhysX-reported limits ───────────────────────────────────────────────
    all_dof_indices, _ = robot.find_joints(_JOINT_NAMES)
    physx_limits = robot.data.joint_pos_limits[0, all_dof_indices, :]  # (6, 2)
    physx_lower = physx_limits[:, 0].tolist()
    physx_upper = physx_limits[:, 1].tolist()

    print("\n[calib] PhysX-reported limits (from USD):")
    for i, name in enumerate(_JOINT_NAMES):
        print(
            f"  {name:<16} lower={math.degrees(physx_lower[i]):+8.3f}°  "
            f"upper={math.degrees(physx_upper[i]):+8.3f}°"
        )

    # ── Sweep each joint to its limits ──────────────────────────────────────
    settled_lower: list[float] = []
    settled_upper: list[float] = []
    converged_lower: list[bool] = []
    converged_upper: list[bool] = []

    print("\n[calib] Sweeping each joint to its PhysX limits …")
    for i, name in enumerate(_JOINT_NAMES):
        joint_idx = [all_dof_indices[i]]
        print(f"  [{i+1}/6] {name}", end="", flush=True)

        # Return to zero first to give a clean starting pose.
        zero = torch.zeros(1, 1, device=sim_device)
        for _ in range(120):
            robot.write_joint_position_to_sim(zero, joint_ids=joint_idx)
            robot.write_joint_velocity_to_sim(torch.zeros_like(zero), joint_ids=joint_idx)
            robot.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        # Lower limit
        target_lo = torch.tensor([[physx_lower[i]]], device=sim_device)
        val_lo, ok_lo = _settle_to_target(
            sim, scene, robot, joint_idx, target_lo.squeeze(),
            args_cli.settle_steps, args_cli.settle_tol_rad, args_cli.settle_consec,
            sim_device,
        )
        settled_lower.append(val_lo)
        converged_lower.append(ok_lo)
        print(f"  lower→{math.degrees(val_lo):+7.2f}°{'✓' if ok_lo else '⚠TIMEOUT'}", end="", flush=True)

        # Return to zero
        for _ in range(120):
            robot.write_joint_position_to_sim(zero, joint_ids=joint_idx)
            robot.write_joint_velocity_to_sim(torch.zeros_like(zero), joint_ids=joint_idx)
            robot.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        # Upper limit
        target_hi = torch.tensor([[physx_upper[i]]], device=sim_device)
        val_hi, ok_hi = _settle_to_target(
            sim, scene, robot, joint_idx, target_hi.squeeze(),
            args_cli.settle_steps, args_cli.settle_tol_rad, args_cli.settle_consec,
            sim_device,
        )
        settled_upper.append(val_hi)
        converged_upper.append(ok_hi)
        print(f"  upper→{math.degrees(val_hi):+7.2f}°{'✓' if ok_hi else '⚠TIMEOUT'}")

    # ── Comparison table ─────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    hdr_extra = "  robot.yaml lower  robot.yaml upper  Δ lower (yaml vs settled)  Δ upper (yaml vs settled)" if robot_yaml_limits else ""
    print(
        f"{'Joint':<16} {'PhysX lower':>12} {'Settled lower':>14} {'Δ lower':>9}"
        f"  {'PhysX upper':>12} {'Settled upper':>14} {'Δ upper':>9}"
        + (f"  {'yaml lower':>10} {'yaml upper':>10} {'Δ lo(y-s)':>10} {'Δ hi(y-s)':>10}" if robot_yaml_limits else "")
    )
    print("─" * 100)

    warnings: list[str] = []
    for i, name in enumerate(_JOINT_NAMES):
        dl = math.degrees(settled_lower[i] - physx_lower[i])
        dh = math.degrees(settled_upper[i] - physx_upper[i])

        extra = ""
        if robot_yaml_limits and name in robot_yaml_limits:
            yl, yu = robot_yaml_limits[name]
            dyl = math.degrees(settled_lower[i] - yl)
            dyh = math.degrees(settled_upper[i] - yu)
            extra = (
                f"  {math.degrees(yl):>10.3f}°"
                f" {math.degrees(yu):>10.3f}°"
                f" {dyl:>+10.3f}°"
                f" {dyh:>+10.3f}°"
            )
            for label, delta in [("lower yaml-vs-settled", dyl), ("upper yaml-vs-settled", dyh)]:
                if abs(delta) > _WARN_THRESH_DEG:
                    warnings.append(
                        f"  WARNING  {name:<16} {label}: |Δ| = {abs(delta):.3f}° > {_WARN_THRESH_DEG}°"
                    )
        for label, delta in [("lower physx-vs-settled", dl), ("upper physx-vs-settled", dh)]:
            if abs(delta) > _WARN_THRESH_DEG:
                warnings.append(
                    f"  WARNING  {name:<16} {label}: |Δ| = {abs(delta):.3f}° > {_WARN_THRESH_DEG}°"
                )

        row = (
            f"{name:<16}"
            f" {math.degrees(physx_lower[i]):>12.3f}°"
            f" {math.degrees(settled_lower[i]):>13.3f}°"
            f" {dl:>+8.3f}°"
            f"  {math.degrees(physx_upper[i]):>12.3f}°"
            f" {math.degrees(settled_upper[i]):>13.3f}°"
            f" {dh:>+8.3f}°"
            + extra
        )
        print(row)

    print("─" * 100)
    if warnings:
        print("\nWARNINGS (|Δ| > 0.5°):")
        for w in warnings:
            print(w)
    else:
        print("\nAll deltas within 0.5° — settled bounds match declared limits.")

    # ── Write YAML ──────────────────────────────────────────────────────────
    out_path = Path(args_cli.out) if args_cli.out else (
        Path(__file__).resolve().parent.parent / "configs" / "sim_joint_limits.yaml"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import datetime as _dt
    today = _dt.date.today().isoformat()

    lines: list[str] = [
        f"# sim_joint_limits.yaml — empirically settled SO-101 sim joint bounds.",
        f"# Generated {today} by so101_rl/scripts/calibrate_sim_joints.py",
        f"# settled = position recorded after commanding each PhysX limit.",
        f"# physx   = USD/PhysX declared limit (what the training env uses for norm).",
        "joint_limits:",
    ]
    for i, name in enumerate(_JOINT_NAMES):
        lo_s = settled_lower[i]
        hi_s = settled_upper[i]
        lo_p = physx_lower[i]
        hi_p = physx_upper[i]
        conv_lo = "✓" if converged_lower[i] else "timeout"
        conv_hi = "✓" if converged_upper[i] else "timeout"
        lines.append(f"  {name}:")
        lines.append(
            f"    # settled: [{lo_s:.6f}, {hi_s:.6f}]  ({conv_lo}/{conv_hi})"
            f"  physx: [{lo_p:.6f}, {hi_p:.6f}]"
        )
        lines.append(f"    sim:  [{lo_p:.6f},  {hi_p:.6f}]")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n[calib] Written → {out_path}")

    # ── Copy-paste block for robot.yaml ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("Copy-paste into robot.yaml::joint_limits  (sim: fields only)")
    print("─" * 60)
    for i, name in enumerate(_JOINT_NAMES):
        lo_p = physx_lower[i]
        hi_p = physx_upper[i]
        lo_s = settled_lower[i]
        hi_s = settled_upper[i]
        conv = ("✓" if converged_lower[i] else "~") + ("✓" if converged_upper[i] else "~")
        print(f"  {name}:")
        print(f"    sim:  [{lo_p:.6f},  {hi_p:.6f}]  # physx (settled: [{lo_s:.6f}, {hi_s:.6f}] {conv})")
    print("─" * 60)
    print("[calib] Done.")



if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
