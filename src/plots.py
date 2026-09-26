"""Report figures: convergence, segmentation panels, time vs K, and CD diagrams."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Headless; must precede importing pyplot.
import matplotlib.pyplot as plt  # noqa: E402

from src.experiment import RESULTS_ROOT, CellSpec, load_curves  # noqa: E402
from src.objectives import DEFAULT_Q  # noqa: E402
from src.segmentation import LEVELS, display_image  # noqa: E402

FIGURE_ROOT = RESULTS_ROOT / "figures"
DPI = 200
GRID_STYLE = {"alpha": 0.25, "linewidth": 0.5}
GREY_RANGE = {"cmap": "gray", "vmin": 0, "vmax": LEVELS - 1}
FALLBACK_COLOUR = "#777777"
CD_BAR_COLOUR = "#C44E52"

CD_LEADER_END = 1 - 0.3
CD_LABEL_X = 1 - 0.35

COLOURS = {
    "standard_de": "#4C72B0",
    "jade": "#DD8452",
    "shade": "#55A868",
    "lshade": "#C44E52",
    "late_acceptance_de": "#8172B3",
}
LABELS = {
    "standard_de": "DE",
    "jade": "JADE",
    "shade": "SHADE",
    "lshade": "L-SHADE",
    "late_acceptance_de": "LADE",
}


def _label(algorithm: str) -> str:
    return LABELS.get(algorithm, algorithm)


def _style(algorithm: str) -> dict[str, str]:
    return {
        "color": COLOURS.get(algorithm, FALLBACK_COLOUR),
        "label": _label(algorithm),
    }


def _label_axes(axes: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend(frameon=False, fontsize=9)
    axes.grid(**GRID_STYLE)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def convergence(
    dataset: str,
    image_id: str,
    objective: str,
    k: int,
    algorithms: list[str],
    results_root: Path | None = None,
    q: float = DEFAULT_Q,
    output: Path | None = None,
) -> Path:
    """Median best-so-far with interquartile band, against evaluations"""
    figure, axes = plt.subplots(figsize=(7, 4.5))

    for algorithm in algorithms:
        spec = CellSpec(dataset, image_id, algorithm, objective, k, q=q)
        try:
            curves = load_curves(spec, results_root)
        except FileNotFoundError:
            continue
        evaluations = np.linspace(0, spec.max_evaluations, curves.shape[1])
        median, lower, upper = _median_and_quartiles(curves)
        style = _style(algorithm)
        axes.plot(evaluations, median, linewidth=1.6, **style)
        axes.fill_between(
            evaluations, lower, upper, alpha=0.15, color=style["color"], linewidth=0
        )

    _label_axes(
        axes,
        "Function evaluations",
        f"{objective.title()} fitness (best so far)",
        f"{image_id} — {objective}, $K={k}$",
    )
    filename = f"{dataset}_{image_id}_{objective}_K{k}.png"

    return _save(figure, output or FIGURE_ROOT / "convergence" / filename)


def _median_and_quartiles(
    curves: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Grid points before the first evaluation are all-NaN by design.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            np.nanmedian(curves, axis=0),
            np.nanpercentile(curves, 25, axis=0),
            np.nanpercentile(curves, 75, axis=0),
        )


def segmentation_panel(
    image: np.ndarray,
    thresholds_by_k: dict[int, np.ndarray],
    title: str = "",
    output: Path | None = None,
) -> Path:
    levels = sorted(thresholds_by_k)
    panels = len(levels) + 1
    figure, axes = plt.subplots(1, panels, figsize=(2.4 * panels, 2.8))

    _show_grey(axes[0], image, "Original")
    for axis, k in zip(axes[1:], levels):
        _show_grey(axis, display_image(image, thresholds_by_k[k]), f"$K={k}$")

    if title:
        figure.suptitle(title, fontsize=10)
    name = output or FIGURE_ROOT / "segmentation" / f"{title or 'panel'}.png"
    return _save(figure, name)


def _show_grey(axis: plt.Axes, image: np.ndarray, title: str) -> None:
    axis.imshow(image, **GREY_RANGE)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def time_vs_k(frame: pd.DataFrame, output: Path | None = None) -> Path:
    figure, axes = plt.subplots(figsize=(6.5, 4.2))

    for algorithm, group in frame.groupby("algorithm"):
        by_k = group.groupby("k").wall_time_s.agg(["mean", "std"])
        axes.errorbar(
            by_k.index, by_k["mean"], yerr=by_k["std"],
            marker="o", markersize=4, capsize=3, linewidth=1.4, **_style(algorithm),
        )

    _label_axes(
        axes,
        "Number of thresholds $K$",
        "Wall-clock time per run (s)",
        "Execution time against threshold count",
    )
    return _save(figure, output or FIGURE_ROOT / "scalability" / "time_vs_k.png")


def critical_difference(
    mean_ranks: pd.Series,
    critical_difference_value: float,
    title: str = "",
    output: Path | None = None,
) -> Path:
    """Demsar-style diagram; a bar joins algorithms the test cannot separate."""
    
    ranks = mean_ranks.sort_values()
    n = len(ranks)
    figure, axes = plt.subplots(figsize=(7, 1.1 + 0.34 * n))

    axes.set_xlim(1 - 0.4, n + 0.4)
    axes.set_ylim(0, n + 1.6)
    axes.axis("off")

    _draw_rank_axis(axes, n)
    _draw_algorithm_leaders(axes, ranks)
    _draw_indistinguishable_bars(axes, ranks.to_numpy(), critical_difference_value)

    axes.text(
        n, 0.1, f"CD = {critical_difference_value:.3f}",
        ha="right", va="bottom", fontsize=9,
    )
    if title:
        axes.set_title(title, fontsize=10)

    return _save(figure, output or FIGURE_ROOT / "stats" / "critical_difference.png")


def _draw_rank_axis(axes: plt.Axes, n: int) -> None:
    top = n + 1
    axes.plot([1, n], [top, top], color="black", linewidth=1.1)
    for tick in range(1, n + 1):
        axes.plot([tick, tick], [top, top + 0.16], color="black", linewidth=1.1)
        axes.text(tick, top + 0.3, str(tick), ha="center", va="bottom", fontsize=9)


def _draw_algorithm_leaders(axes: plt.Axes, ranks: pd.Series) -> None:
    """Elbow lines from each algorithm's name up to its mean rank."""
    n = len(ranks)
    for position, (algorithm, rank) in enumerate(ranks.items()):
        height = n - position
        axes.plot([rank, rank], [height, n + 1], color="black", linewidth=0.8)
        axes.plot([rank, CD_LEADER_END], [height, height], color="black", linewidth=0.8)
        axes.text(
            CD_LABEL_X, height, _label(algorithm),
            ha="right", va="center", fontsize=9,
        )


def _draw_indistinguishable_bars(
    axes: plt.Axes, sorted_ranks: np.ndarray, critical_difference_value: float
) -> None:
    bar_height = 0.4
    for first in range(len(sorted_ranks)):
        limit = sorted_ranks[first] + critical_difference_value
        last = int(np.searchsorted(sorted_ranks, limit, "right")) - 1
        if last > first:
            axes.plot(
                [sorted_ranks[first] - 0.03, sorted_ranks[last] + 0.03],
                [bar_height, bar_height],
                color=CD_BAR_COLOUR, linewidth=3, solid_capstyle="butt",
            )
            bar_height += 0.28
