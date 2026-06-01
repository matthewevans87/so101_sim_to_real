"""CLI command modules — one file per subcommand."""

from . import (
    calibrate_camera,
    camera_test,
    compare_views,
    configure_camera,
    probe,
    robot_test,
    run,
    run_static,
    stream,
)

__all__ = [
    "calibrate_camera",
    "camera_test",
    "compare_views",
    "configure_camera",
    "probe",
    "robot_test",
    "run",
    "run_static",
    "stream",
]
