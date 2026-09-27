"""Experiment 1 -- BSD500 reconstruction metrics."""

from __future__ import annotations

import sys

from experiments._common import (
    RECONSTRUCTION_METRICS,
    Sweep,
    load_records,
    parse_sweep_args,
    run_sweep,
    write_summary,
)

SWEEP = Sweep(number=1, dataset="bsd500", display_name="BSD500")


def main(argv: list[str] | None = None) -> int:
    args = parse_sweep_args(__doc__, argv)
    records = load_records(SWEEP.dataset, args.smoke)

    frame = run_sweep(SWEEP, args, records)
    if frame.empty:
        print("no results produced")
        return 1

    write_summary(SWEEP, frame, args.results, RECONSTRUCTION_METRICS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
