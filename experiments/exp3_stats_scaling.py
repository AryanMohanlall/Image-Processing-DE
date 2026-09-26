"""Experiment 3 -- statistics and scalability."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from experiments._common import tables_dir
from src import plots
from src.experiment import RESULTS_ROOT, collect
from src.stats import FriedmanResult, compare_all, friedman

MIN_FRIEDMAN_ALGORITHMS = 3


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    frame = collect(args.results)
    if frame.empty:
        print("no results found -- run exp1_bsd500.py and exp2_chaos.py first")
        return 1
    frame = frame[frame.status == "ok"]
    tables = tables_dir(args.results)

    write_wilcoxon(frame, args.metric, tables)
    write_friedman(frame, args.metric, args.results, args.figures)
    if args.figures:
        plot_scalability(frame, args.results)
        plot_convergence(frame, args.results)
    write_timing(frame, tables)

    print(f"wrote statistics and figures under {args.results}")
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--metric", default="best_fitness")
    parser.add_argument(
        "--figures", action=argparse.BooleanOptionalAction, default=True,
        help="render the critical-difference, scalability and convergence figures",
    )
    return parser.parse_args(argv)


def write_wilcoxon(frame: pd.DataFrame, metric: str, tables: Path) -> None:
    comparisons = compare_all(frame, metric)
    if comparisons.empty:
        print("Wilcoxon: need at least two algorithms to compare")
        return

    comparisons.to_csv(tables / "wilcoxon_pairwise.csv", index=False)
    significant = int(comparisons.significant.sum())
    print(
        f"Wilcoxon: {significant}/{len(comparisons)} pairwise comparisons "
        f"significant after Holm correction"
    )


def write_friedman(
    frame: pd.DataFrame, metric: str, results: Path, with_figures: bool
) -> None:
    rankings = []
    for dataset, objective, outcome in friedman_outcomes(frame, metric):
        rankings += ranking_rows(dataset, objective, outcome)
        print(
            f"Friedman {dataset}/{objective}: p={outcome.p_iman_davenport:.3e} "
            f"CD={outcome.critical_difference:.3f} "
            f"best={outcome.mean_ranks.index[0]}"
        )
        if with_figures:
            plots.critical_difference(
                outcome.mean_ranks,
                outcome.critical_difference,
                title=f"{dataset} — {objective}",
                output=figure_path(results, "stats", f"cd_{dataset}_{objective}.png"),
            )

    if rankings:
        pd.DataFrame(rankings).to_csv(
            tables_dir(results) / "friedman_ranks.csv", index=False
        )


def friedman_outcomes(
    frame: pd.DataFrame, metric: str
) -> Iterator[tuple[str, str, FriedmanResult]]:
    for (dataset, objective), group in frame.groupby(["dataset", "objective"]):
        n_algorithms = group.algorithm.nunique()
        if n_algorithms < MIN_FRIEDMAN_ALGORITHMS:
            print(
                f"Friedman skipped for {dataset}/{objective}: needs "
                f"{MIN_FRIEDMAN_ALGORITHMS}+ algorithms, found {n_algorithms}"
            )
            continue
        try:
            yield dataset, objective, friedman(group, metric)
        except ValueError as error:
            print(f"Friedman skipped for {dataset}/{objective}: {error}")


def ranking_rows(
    dataset: str, objective: str, outcome: FriedmanResult
) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "objective": objective,
            "algorithm": algorithm,
            "mean_rank": rank,
            "n_instances": outcome.n_instances,
            "chi_square": outcome.chi_square,
            "iman_davenport": outcome.iman_davenport,
            "p_value": outcome.p_iman_davenport,
            "significant": outcome.significant,
            "critical_difference": outcome.critical_difference,
        }
        for algorithm, rank in outcome.mean_ranks.items()
    ]


def plot_scalability(frame: pd.DataFrame, results: Path) -> None:
    for dataset, group in frame.groupby("dataset"):
        plots.time_vs_k(
            group,
            output=figure_path(results, "scalability", f"time_vs_k_{dataset}.png"),
        )


def plot_convergence(frame: pd.DataFrame, results: Path) -> None:
    algorithms = sorted(frame.algorithm.unique())
    for (dataset, objective), group in frame.groupby(["dataset", "objective"]):
        image_id = min(group.image_id.unique())
        k = int(group.k.max())
        plots.convergence(
            dataset, image_id, objective, k, algorithms, results,
            output=figure_path(results, "convergence", f"{dataset}_{objective}_K{k}.png"),
        )


def write_timing(frame: pd.DataFrame, tables: Path) -> None:
    timing = (
        frame.groupby(["dataset", "algorithm", "k"])
        .agg(
            wall_time_mean=("wall_time_s", "mean"),
            wall_time_std=("wall_time_s", "std"),
            evaluations_per_s=("evaluations_per_s", "mean"),
        )
        .reset_index()
    )
    timing.to_csv(tables / "scalability_timing.csv", index=False)


def figure_path(results: Path, category: str, filename: str) -> Path:
    return results / "figures" / category / filename


if __name__ == "__main__":
    sys.exit(main())
