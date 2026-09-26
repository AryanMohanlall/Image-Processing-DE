#!/usr/bin/env python3
"""Master script: runs every experiment and records what produced the results.

    python run_all.py            # the full study
    python run_all.py --smoke    # troubleshooting run
    python run_all.py --stats    # run for stat generation

Results are cached per cell, so re-running resumes rather than restarts.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments import exp1_bsd500, exp2_chaos, exp3_stats_scaling
from experiments._common import (
    DEFAULT_JOBS,
    SMOKE_EVALUATIONS,
    SMOKE_RUNS,
    tables_dir,
)
from src.algorithms import available
from src.datasets import manifest
from src.experiment import MAX_EVALUATIONS, N_RUNS, RESULTS_ROOT

BANNER = "=" * 70
SWEEPS = (
    ("Experiment 1 (BSD500)", exp1_bsd500),
    ("Experiment 2 (CHAOS MRI)", exp2_chaos),
)
STATS_TITLE = "Experiment 3 (statistics and scalability)"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    algorithms = available()
    if not algorithms:
        print(
            "No algorithms registered. Add an implementation under src/algorithms/;"
            " standard_de.py documents the interface.",
            file=sys.stderr,
        )
        return 1
    print(f"Algorithms: {', '.join(algorithms)}")

    started = time.time()
    manifest().to_csv(tables_dir(args.results) / "image_manifest.csv", index=False)

    sweep_args = _sweep_args(args)
    for title, module in SWEEPS:
        if not _run(title, module, sweep_args):
            return 1

    stats_args = ["--results", str(args.results)]
    if args.stats and not _run(STATS_TITLE, exp3_stats_scaling, stats_args):
        return 1

    elapsed = time.time() - started
    _write_manifest(args, algorithms, elapsed)
    print(f"\nDone in {elapsed / 60:.1f} min. Results under {args.results}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="also run Experiment 3 (statistics, convergence, scalability)",
    )
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--max-evaluations", type=int, default=MAX_EVALUATIONS)
    return parser.parse_args(argv)


def _sweep_args(args: argparse.Namespace) -> list[str]:
    """The command line forwarded to each sweep experiment."""
    forwarded = ["--jobs", str(args.jobs), "--results", str(args.results)]
    if args.overwrite:
        forwarded.append("--overwrite")
    if args.smoke:
        forwarded.append("--smoke")
    else:
        forwarded += [
            "--runs", str(args.runs),
            "--max-evaluations", str(args.max_evaluations),
        ]
    return forwarded


def _run(title: str, module, argv: list[str]) -> bool:
    print(f"\n{BANNER}\n{title}\n{BANNER}")
    return module.main(argv) == 0


def _write_manifest(
    args: argparse.Namespace,
    algorithms: list[str],
    elapsed: float,
) -> None:
    (args.results / "manifest.json").write_text(
        json.dumps(
            {
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(elapsed, 1),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "algorithms": algorithms,
                "runs": SMOKE_RUNS if args.smoke else args.runs,
                "max_evaluations": (
                    SMOKE_EVALUATIONS if args.smoke else args.max_evaluations
                ),
                "smoke": args.smoke,
                "stats": args.stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())

