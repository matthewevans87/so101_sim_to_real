"""compare-views — side-by-side + blend overlay of a real frame and a sim render."""

from __future__ import annotations

import sys
from pathlib import Path


def cmd_compare_views(args) -> None:
    import numpy as np

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required. Run: pip install opencv-python")
        sys.exit(1)

    real_path = Path(args.real)
    sim_path = Path(args.sim)
    if not real_path.exists():
        print(f"ERROR: real image not found: {real_path}")
        sys.exit(1)
    if not sim_path.exists():
        print(f"ERROR: sim image not found: {sim_path}")
        sys.exit(1)

    real = cv2.imread(str(real_path))
    sim = cv2.imread(str(sim_path))
    if real is None:
        print(f"ERROR: could not read real image: {real_path}")
        sys.exit(1)
    if sim is None:
        print(f"ERROR: could not read sim image: {sim_path}")
        sys.exit(1)

    if args.match_size == "real":
        sim = cv2.resize(
            sim, (real.shape[1], real.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    elif args.match_size == "sim":
        real = cv2.resize(
            real, (sim.shape[1], sim.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    else:
        w, h = map(int, args.match_size.split("x"))
        real = cv2.resize(real, (w, h), interpolation=cv2.INTER_LINEAR)
        sim = cv2.resize(sim, (w, h), interpolation=cv2.INTER_LINEAR)

    H, W = real.shape[:2]
    alpha = max(0.0, min(1.0, args.alpha))
    blend = cv2.addWeighted(real, 1.0 - alpha, sim, alpha, 0.0)

    checker = real.copy()
    n = args.checker_size
    for row in range(0, H, 2 * n):
        for col in range(0, W, 2 * n):
            checker[row : row + n, col : col + n] = sim[row : row + n, col : col + n]
            r2 = min(row + 2 * n, H)
            c2 = min(col + 2 * n, W)
            checker[row + n : r2, col + n : c2] = sim[row + n : r2, col + n : c2]

    def _label(img, text):
        out = img.copy()
        cv2.putText(
            out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out,
            text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    panels = [
        _label(real, "real"),
        _label(sim, "sim"),
        _label(blend, f"blend ({int((1 - alpha) * 100)}% real)"),
        _label(checker, f"checker ({n}px)"),
    ]
    composite = np.concatenate(
        [
            np.concatenate(panels[:2], axis=1),
            np.concatenate(panels[2:], axis=1),
        ],
        axis=0,
    )

    out_path = (
        Path(args.output)
        if args.output
        else real_path.with_stem(real_path.stem + "_compare").with_suffix(".jpg")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out_path}  ({composite.shape[1]}x{composite.shape[0]})")

    if args.show:
        max_w = args.display_width
        disp = composite
        ch, cw = composite.shape[:2]
        if cw > max_w:
            scale = max_w / cw
            disp = cv2.resize(
                composite, (max_w, int(ch * scale)), interpolation=cv2.INTER_AREA
            )
        cv2.imshow("compare-views", disp)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def add_parser(sub) -> None:
    p = sub.add_parser(
        "compare-views",
        help="Side-by-side + blend overlay comparison of a real frame and a sim render",
    )
    p.add_argument("--real", required=True, metavar="PATH", help="Real camera image")
    p.add_argument("--sim", required=True, metavar="PATH", help="Sim render image")
    p.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="Output composite image path (default: <real>_compare.jpg)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend weight for sim (0=all-real, 1=all-sim, default 0.5)",
    )
    p.add_argument(
        "--checker-size",
        type=int,
        default=64,
        dest="checker_size",
        metavar="N",
        help="Checkerboard tile size in pixels (default 64)",
    )
    p.add_argument(
        "--match-size",
        default="real",
        dest="match_size",
        metavar="SPEC",
        help="Resize both images to match: 'real', 'sim', or 'WxH' (default: real)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display the composite in a window (requires a display)",
    )
    p.add_argument(
        "--display-width",
        type=int,
        default=1920,
        dest="display_width",
        metavar="PX",
        help="Max pixel width of the displayed window (default: 1920)",
    )
    p.set_defaults(func=cmd_compare_views)
