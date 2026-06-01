#!/usr/bin/env python3
"""Generate a printable checkerboard calibration target as a PDF and PNG.

OpenCV's calibrateCamera uses *inner corners*, so a board with
--cols 9 --rows 6  has a 10×7 square grid (one extra square on each edge).

Usage:
    python so101_real/calibration/gen_checkerboard.py
    python so101_real/calibration/gen_checkerboard.py --cols 9 --rows 6 --square-mm 25
"""

import argparse
from pathlib import Path

import numpy as np


def generate(
    cols: int = 9,
    rows: int = 6,
    square_mm: float = 25.0,
    border_mm: float = 20.0,
    out_dir: Path = Path(__file__).parent,
) -> None:
    try:
        import matplotlib
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")

    # Board has (cols+1) × (rows+1) squares
    n_cols_sq = cols + 1
    n_rows_sq = rows + 1

    board_w_mm = n_cols_sq * square_mm
    board_h_mm = n_rows_sq * square_mm

    # Figure size in inches (at 25.4 mm/inch)
    mm_per_in = 25.4
    border_in = border_mm / mm_per_in
    fig_w_in = board_w_mm / mm_per_in + 2 * border_in
    fig_h_in = board_h_mm / mm_per_in + 2 * border_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    # add_axes([left, bottom, width, height] in figure fractions) — fills entire figure
    # so that 1 data unit == 1 inch, with no hidden subplot margins.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w_in)
    ax.set_ylim(0, fig_h_in)
    ax.set_aspect("equal")
    ax.axis("off")

    # White background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sq_in = square_mm / mm_per_in

    for r in range(n_rows_sq):
        for c in range(n_cols_sq):
            if (r + c) % 2 == 0:
                x = border_in + c * sq_in
                y = border_in + r * sq_in
                rect = mpatches.Rectangle(
                    (x, y), sq_in, sq_in, linewidth=0, facecolor="black"
                )
                ax.add_patch(rect)

    # Label: inner-corner count and square size
    label = (
        f"Checkerboard  {cols}×{rows} inner corners  |  {square_mm:.0f} mm squares  |  "
        f"Board {board_w_mm:.0f}×{board_h_mm:.0f} mm"
    )
    ax.text(
        fig_w_in / 2,
        border_in / 2,
        label,
        ha="center",
        va="center",
        fontsize=7,
        color="black",
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"checkerboard_{cols}x{rows}_{square_mm:.0f}mm"

    pdf_path = out_dir / f"{stem}.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        # No bbox_inches — figure size already encodes exact physical dimensions.
        # bbox_inches="tight" would re-crop and alter the scale.
        pdf.savefig(fig)
    print(f"PDF written: {pdf_path}")

    png_path = out_dir / f"{stem}.png"
    fig.savefig(str(png_path), dpi=300)
    print(f"PNG written: {png_path}")

    plt.close(fig)

    print()
    print("Print instructions:")
    print(
        f"  Print at 100% scale (no 'fit to page').  Each square must measure {square_mm:.0f} mm."
    )
    print(
        f"  Board size: {board_w_mm:.0f} mm × {board_h_mm:.0f} mm "
        f'({board_w_mm/mm_per_in:.2f}" × {board_h_mm/mm_per_in:.2f}")'
    )
    print(f"  Inner corners (for --board-cols / --board-rows): {cols} × {rows}")
    print()
    print(
        "After printing, verify with a ruler that one square = "
        f"{square_mm:.0f} mm before capturing calibration frames."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate printable checkerboard calibration target."
    )
    parser.add_argument(
        "--cols", type=int, default=9, help="Inner corner columns (default 9)"
    )
    parser.add_argument(
        "--rows", type=int, default=6, help="Inner corner rows (default 6)"
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=25.0,
        help="Square side length in mm (default 25)",
    )
    parser.add_argument(
        "--border-mm",
        type=float,
        default=20.0,
        help="White border around board in mm (default 20)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory (default: same dir as this script)",
    )
    args = parser.parse_args()
    generate(
        cols=args.cols,
        rows=args.rows,
        square_mm=args.square_mm,
        border_mm=args.border_mm,
        out_dir=args.out_dir,
    )
