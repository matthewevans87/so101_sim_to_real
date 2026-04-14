"""CLI entry point: ``python -m so101.reward_analysis --config <path>``.

Prints a table of per-reward best-case maxima and detected pathologies.
No Isaac Lab or so101_rl installation required.

Exit code is 1 when pathologies are detected (useful in CI / pre-run checks).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from so101.reward_analysis.analyser import AnalysisResult, RewardRow, analyse


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    if v == 0.0:
        return "0"
    if abs(v) >= 10_000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


# ---------------------------------------------------------------------------
# Plain-text renderer
# ---------------------------------------------------------------------------


def _print_plain(result: AnalysisResult) -> None:
    COLS = [32, 12, 8, 11, 14, 10, 9, 10]
    HEADERS = ["type", "mode", "scale", "max/step", "max/episode", "terminal", "termNow", "reachable"]

    def row_str(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, COLS))

    sep = "  ".join("-" * w for w in COLS)

    print(f"\nConfig : {result.config_path}")
    print(f"Episode: {result.episode_length_s}s  |  Steps: {result.num_steps}\n")
    print(row_str(HEADERS))
    print(sep)

    for r in result.rows:
        term_str = "yes" if r.is_terminal else ""
        term_now_str = "" if not r.is_terminal else ("yes" if r.terminates else "no")
        if r.terminal_reachable is None:
            reach_str = ""
        elif r.terminal_reachable:
            reach_str = "yes"
        else:
            reach_str = "NO <<<"

        label = r.type_name
        if r.gates:
            label += f"  [gated]"

        print(row_str([label, r.mode, _fmt(r.scale), _fmt(r.max_step), _fmt(r.max_episode), term_str, term_now_str, reach_str]))
        for note in r.notes:
            print(f"    NOTE: {note}")

    print()
    if result.pathologies:
        print(f"PATHOLOGIES ({len(result.pathologies)}):")
        for p in result.pathologies:
            print(f"  [!] {p}")
    else:
        print("No pathologies detected.")
    print()


# ---------------------------------------------------------------------------
# Rich renderer
# ---------------------------------------------------------------------------


def _print_rich(result: AnalysisResult) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    table = Table(
        title=f"Reward Analysis — {Path(result.config_path).name}",
        box=box.SIMPLE_HEAD,
        show_lines=False,
        header_style="bold",
    )
    table.add_column("type", no_wrap=True, min_width=28)
    table.add_column("mode", style="dim", min_width=10)
    table.add_column("scale", justify="right", min_width=7)
    table.add_column("max/step", justify="right", min_width=9)
    table.add_column("max/episode", justify="right", min_width=12)
    table.add_column("terminal", justify="center", min_width=8)
    table.add_column("termNow", justify="center", min_width=7)
    table.add_column("reachable", justify="center", min_width=9)
    table.add_column("gates / notes", min_width=30)

    for r in result.rows:
        # Colour terminal rows differently
        if r.is_terminal:
            term_str = "[yellow]yes[/]"
        else:
            term_str = ""

        if not r.is_terminal:
            term_now_str = ""
        elif r.terminates:
            term_now_str = "[yellow]yes[/]"
        else:
            term_now_str = "[magenta]no[/]"

        if r.terminal_reachable is None:
            reach_str = ""
        elif r.terminal_reachable:
            reach_str = "[green]yes[/]"
        else:
            reach_str = "[red bold]NO[/]"

        # Highlight dominant absolute shaping rewards
        ep_val = _fmt(r.max_episode)
        if not r.is_terminal and r.scale > 0 and r.max_episode >= 500:
            ep_str = f"[bold cyan]{ep_val}[/]"
        elif not r.is_terminal and r.scale > 0 and r.max_episode >= 100:
            ep_str = f"[cyan]{ep_val}[/]"
        else:
            ep_str = ep_val

        # Highlight positive terminals
        if r.is_terminal and r.max_episode > 0:
            ep_str = f"[bold yellow]{ep_val}[/]"

        type_label = r.type_name
        if r.gates:
            type_label += " [dim][gated][/]"

        extras: list[str] = list(r.gates)
        for note in r.notes:
            extras.append(f"[yellow]NOTE:[/] {note}")
        extras_str = "; ".join(extras) if extras else ""

        table.add_row(
            type_label,
            r.mode,
            _fmt(r.scale),
            _fmt(r.max_step),
            ep_str,
            term_str,
            term_now_str,
            reach_str,
            extras_str,
        )

    console.print()
    console.print(f"  Config : [dim]{result.config_path}[/]")
    console.print(f"  Episode: {result.episode_length_s}s  |  Steps: {result.num_steps}")
    console.print()
    console.print(table)

    if result.pathologies:
        lines = "\n".join(f"  [red][!][/] {p}" for p in result.pathologies)
        console.print(
            Panel(
                lines,
                title=f"[bold red]Pathologies detected ({len(result.pathologies)})[/]",
                border_style="red",
                expand=False,
            )
        )
    else:
        console.print("  [green]✓ No pathologies detected.[/]\n")
    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m so101.reward_analysis",
        description=(
            "Analytically compute best-case reward maxima and detect pathologies "
            "from a So101EnvParams-compatible YAML. No Isaac Lab required."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to env config YAML.")
    parser.add_argument(
        "--k",
        type=float,
        default=2.0,
        help=(
            "Shaping-dominance threshold: flag when "
            "sum(shaping_max_episode) > k × terminal_scale. Default: 2.0"
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write analysis result as JSON to PATH.",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Use plain-text output even if rich is installed.",
    )
    args = parser.parse_args()

    result = analyse(args.config, k=args.k)

    use_rich = not args.no_rich
    if use_rich:
        try:
            _print_rich(result)
        except ImportError:
            use_rich = False
    if not use_rich:
        _print_plain(result)

    if args.json:
        payload = {
            "config_path": result.config_path,
            "num_steps": result.num_steps,
            "episode_length_s": result.episode_length_s,
            "rewards": [
                {
                    "type_name": r.type_name,
                    "mode": r.mode,
                    "scale": r.scale,
                    "max_step": r.max_step,
                    "max_episode": r.max_episode,
                    "is_terminal": r.is_terminal,
                    "terminates": r.terminates,
                    "terminal_reachable": r.terminal_reachable,
                    "gates": r.gates,
                    "notes": r.notes,
                }
                for r in result.rows
            ],
            "pathologies": result.pathologies,
        }
        out_path = Path(args.json)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"JSON written to {out_path}", file=sys.stderr)

    if result.pathologies:
        sys.exit(1)


if __name__ == "__main__":
    main()
