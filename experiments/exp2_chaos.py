"""Experiment 2 -- Ground truth is optional by design because we still need clarification on this."""

from __future__ import annotations

import sys

import pandas as pd

from experiments._common import (
    OVERLAP_METRICS,
    RECONSTRUCTION_METRICS,
    Sweep,
    load_records,
    parse_sweep_args,
    run_sweep,
    write_summary,
)
from src.datasets import ImageRecord

SWEEP = Sweep(number=2, dataset="chaos", display_name="CHAOS MRI")

MISSING_GROUND_TRUTH_NOTE = (
    "NOTE: no CHAOS ground-truth masks found, so Jaccard and Dice cannot\n"
    "      be computed. Reconstruction metrics still run. Place the masks\n"
    "      in data/CHAOS/Ground/ (same filenames as the slices) to enable\n"
    "      the overlap metrics."
)


def main(argv: list[str] | None = None) -> int:
    args = parse_sweep_args(__doc__.splitlines()[0], argv)
    records = load_records(SWEEP.dataset, args.smoke)
    report_ground_truth(records)

    frame = run_sweep(SWEEP, args, records)
    if frame.empty:
        print("no results produced")
        return 1

    write_summary(SWEEP, frame, args.results, metrics_for(frame))
    return 0


def report_ground_truth(records: list[ImageRecord]) -> None:
    annotated = sum(record.has_ground_truth for record in records)
    if annotated == 0:
        print(MISSING_GROUND_TRUTH_NOTE, file=sys.stderr)
    else:
        print(f"ground truth available for {annotated}/{len(records)} slices")


def metrics_for(frame: pd.DataFrame) -> tuple[str, ...]:
    """Overlap metrics are only reported once some slice has a ground-truth mask."""
    if frame.jaccard.notna().any():
        return RECONSTRUCTION_METRICS + OVERLAP_METRICS
    return RECONSTRUCTION_METRICS


if __name__ == "__main__":
    sys.exit(main())
