"""camera-test — display a live camera feed until Ctrl-C."""

from __future__ import annotations


def cmd_camera_test(args) -> None:
    import cv2
    from ..camera import CameraConfig, CameraSource

    config = CameraConfig.load(args.robot_config)
    print(
        f"[camera-test] Opening camera {config.device_index} "
        f"({config.capture_width}x{config.capture_height})..."
    )
    with CameraSource(config) as cam:
        print("[camera-test] Press Ctrl-C to stop.")
        try:
            while True:
                frame = cam.get_frame()
                cv2.imshow("camera-test", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()


def add_parser(sub) -> None:
    p = sub.add_parser("camera-test", help="Display live camera feed")
    p.add_argument("--robot-config", required=True, dest="robot_config", metavar="PATH")
    p.set_defaults(func=cmd_camera_test)
