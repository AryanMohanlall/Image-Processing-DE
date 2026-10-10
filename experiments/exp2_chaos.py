"""Experiment 2 -- domain application to CHAOS abdominal MRI slices."""

from __future__ import annotations

import sys

from experiments._common import (
    RECONSTRUCTION_METRICS,
    SEPARABILITY_METRICS,
    Sweep,
    load_records,
    parse_sweep_args,
    run_sweep,
    write_summary,
)

SWEEP = Sweep(number=2, dataset="chaos", display_name="CHAOS MRI")

METRICS = RECONSTRUCTION_METRICS + SEPARABILITY_METRICS


def main(argv: list[str] | None = None) -> int:
    args = parse_sweep_args(__doc__.splitlines()[0], argv)
    records = load_records(SWEEP.dataset, args.smoke)

    frame = run_sweep(SWEEP, args, records)
    if frame.empty:
        print("no results produced")
        return 1

    write_summary(SWEEP, frame, args.results, METRICS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
