"""Shared functionality between experiments"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.algorithms import available
from src.datasets import ImageRecord, load_dataset
from src.experiment import K_VALUES, MAX_EVALUATIONS, N_RUNS, RESULTS_ROOT, Plan, run_plan
from src.objectives import DEFAULT_Q, OBJECTIVE_NAMES
from src.stats import summary_table, to_latex

DEFAULT_JOBS = -2

RECONSTRUCTION_METRICS = ("psnr", "ssim", "uniformity")
OVERLAP_METRICS = ("jaccard", "dice")

SMOKE_K_VALUES = (3,)
SMOKE_RUNS = 2
SMOKE_EVALUATIONS = 600
SMOKE_OBJECTIVES = ("otsu",)
SMOKE_IMAGE_COUNT = 1


@dataclass(frozen=True)
class Sweep:

    number: int
    dataset: str
    display_name: str

    @property
    def title(self) -> str:
        return f"Experiment {self.number}"

    @property
    def prefix(self) -> str:
        return f"exp{self.number}"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help="algorithms to run (default: every registered algorithm)",
    )
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVE_NAMES))
    parser.add_argument("--k", nargs="+", type=int, default=list(K_VALUES))
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--max-evaluations", type=int, default=MAX_EVALUATIONS)
    parser.add_argument(
        "--q", type=float, default=DEFAULT_Q, help="Tsallis entropic index"
    )
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="parallel workers")
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute cells already on disk"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny run that exercises the whole pipeline in under a minute",
    )
    return parser


def parse_sweep_args(description: str, argv: list[str] | None) -> argparse.Namespace:
    args = build_parser(description).parse_args(argv)
    if args.smoke:
        apply_smoke(args)
    return args


def apply_smoke(args: argparse.Namespace) -> None:
    args.k = list(SMOKE_K_VALUES)
    args.runs = SMOKE_RUNS
    args.max_evaluations = SMOKE_EVALUATIONS
    args.objectives = list(SMOKE_OBJECTIVES)


def resolve_algorithms(requested: list[str] | None) -> tuple[str, ...]:
    registered = available()
    if not registered:
        raise SystemExit(
            "no algorithms are registered -- add an implementation under "
            "src/algorithms/ (see standard_de.py for the expected interface)"
        )
    if requested is None:
        return tuple(registered)
    unknown = set(requested) - set(registered)
    if unknown:
        raise SystemExit(f"unknown algorithms {sorted(unknown)}; registered: {registered}")
    return tuple(requested)


def load_records(dataset: str, smoke: bool) -> list[ImageRecord]:
    records = load_dataset(dataset)
    return records[:SMOKE_IMAGE_COUNT] if smoke else records


def run_sweep(
    sweep: Sweep, args: argparse.Namespace, records: list[ImageRecord]
) -> pd.DataFrame:
    plan = Plan(
        algorithms=resolve_algorithms(args.algorithms),
        objectives=tuple(args.objectives),
        k_values=tuple(args.k),
        datasets=(sweep.dataset,),
        q=args.q,
        n_runs=args.runs,
        max_evaluations=args.max_evaluations,
        records=records,
    )
    print(f"{sweep.title}: {len(plan.cells())} cells x {args.runs} runs")
    frame = run_plan(plan, args.results, n_jobs=args.jobs, overwrite=args.overwrite)
    return frame[frame.dataset == sweep.dataset]


def tables_dir(results: Path) -> Path:
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    return tables


def write_summary(
    sweep: Sweep, frame: pd.DataFrame, results: Path, metrics: tuple[str, ...]
) -> None:
    """Write the summary CSV plus one LaTeX table per objective and metric."""
    tables = tables_dir(results)
    summary = summary_table(frame, metrics)
    summary.to_csv(tables / f"{sweep.prefix}_{sweep.dataset}_summary.csv", index=False)

    for objective in sorted(summary.objective.unique()):
        subset = summary[summary.objective == objective]
        for metric in metrics:
            latex = to_latex(
                subset, metric,
                caption=(
                    f"{metric.upper()} ($\\mu \\pm \\sigma$) on "
                    f"{sweep.display_name}, {objective}."
                ),
                label=f"tab:{sweep.prefix}-{objective}-{metric}",
            )
            (tables / f"{sweep.prefix}_{objective}_{metric}.tex").write_text(latex)

    print(f"wrote {len(summary)} summary rows to {tables}")
