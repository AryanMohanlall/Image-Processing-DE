"""Paired statistical comparison of the algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats as sps

ALPHA = 0.05
CELL_KEYS = ["dataset", "objective", "k"]
SUMMARY_METRICS = ("psnr", "ssim", "uniformity")

HIGHER_IS_BETTER = {
    "best_fitness": True,
    "psnr": True,
    "ssim": True,
    "uniformity": True,
    "jaccard": True,
    "dice": True,
    "gap_to_optimum": False,
    "wall_time_s": False,
}


def _oriented(values, metric: str):
    """Negate lower-is-better metrics so larger is always better."""
    return values if HIGHER_IS_BETTER.get(metric, True) else -values


def holm(p_values: ArrayLike) -> np.ndarray:
    """Holm-Bonferroni (FWER). scipy only offers FDR control, a weaker guarantee"""

    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size

    if n == 0:
        return p_values
    order = np.argsort(p_values)

    scaled = np.maximum.accumulate((n - np.arange(n)) * p_values[order])
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(scaled, 1.0)

    return adjusted


def rank_biserial(differences: ArrayLike) -> float:
    """Wilcoxon effect size in [-1, 1]; positive means the first tends to win"""

    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]

    if nonzero.size == 0:
        return 0.0

    ranks = sps.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()

    return float((positive - negative) / (positive + negative))


@dataclass(frozen=True)
class PairedResult:
    first: str
    second: str
    n: int
    statistic: float
    p_value: float
    effect_size: float
    median_difference: float
    wins: int
    ties: int
    losses: int

    @property
    def winner(self) -> str | None:
        if self.p_value >= ALPHA or self.median_difference == 0:
            return None

        return self.first if self.median_difference > 0 else self.second

    def as_row(self) -> dict[str, object]:
        return {
            "algorithm_a": self.first,
            "algorithm_b": self.second,
            "n_images": self.n,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "median_difference": self.median_difference,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
        }


def _drop_missing(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def paired_test(
    first_values: ArrayLike,
    second_values: ArrayLike,
    first: str = "A",
    second: str = "B",
) -> PairedResult:
    """Wilcoxon signed-rank"""
    a, b = _drop_missing(first_values, second_values)
    differences = a - b
    tally = {
        "first": first,
        "second": second,
        "n": a.size,
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
    }

    if a.size == 0 or np.all(differences == 0):
        return PairedResult(
            statistic=0.0, p_value=1.0, effect_size=0.0, median_difference=0.0,
            **tally,
        )

    statistic, p_value = sps.wilcoxon(
        differences, zero_method="pratt", alternative="two-sided"
    )
    return PairedResult(
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=rank_biserial(differences),
        median_difference=float(np.median(differences)),
        **tally,
    )


def _median_per_image(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return frame.pivot_table(
        index="image_id", columns="algorithm", values=metric, aggfunc="median"
    )


def pairwise(
    frame: pd.DataFrame, metric: str = "best_fitness", correct: bool = True
) -> pd.DataFrame:
    """Holm-corrected comparisons within one (dataset, objective, K) cell"""
    table = _oriented(_median_per_image(frame, metric), metric)
    algorithms = sorted(table.columns)

    rows = pd.DataFrame(
        [
            paired_test(table[first], table[second], first, second).as_row()
            for i, first in enumerate(algorithms)
            for second in algorithms[i + 1 :]
        ]
    )
    if rows.empty:
        return rows
    rows["p_holm"] = holm(rows.p_value) if correct else rows.p_value
    rows["significant"] = rows.p_holm < ALPHA
    return rows


@dataclass(frozen=True)
class FriedmanResult:
    n_instances: int
    n_algorithms: int
    chi_square: float
    p_value: float
    iman_davenport: float
    p_iman_davenport: float
    mean_ranks: pd.Series
    critical_difference: float

    @property
    def significant(self) -> bool:
        return self.p_iman_davenport < ALPHA


def nemenyi_critical_difference(
    n_algorithms: int, n_instances: int, alpha: float = ALPHA
) -> float:
    """Mean-rank gap beyond which two algorithms differ (Demsar, 2006)."""
    q = sps.studentized_range.ppf(1 - alpha, n_algorithms, np.inf) / np.sqrt(2)
    return float(
        q * np.sqrt(n_algorithms * (n_algorithms + 1) / (6.0 * n_instances))
    )


def friedman(frame: pd.DataFrame, metric: str = "best_fitness") -> FriedmanResult:
    """Ranks algorithms over (image, K) instances, with Iman-Davenport's F"""

    table = _median_per_instance(frame, metric)
    algorithms = sorted(table.columns)
    values = table[algorithms].to_numpy()
    n, k = values.shape

    if n < 2 or k < 3:
        raise ValueError(
            f"Friedman needs at least 2 instances and 3 algorithms, got {n} and {k}"
        )

    chi_square, p_value = sps.friedmanchisquare(*values.T)
    f_statistic, p_f = _iman_davenport(chi_square, n, k)

    return FriedmanResult(
        n_instances=n,
        n_algorithms=k,
        chi_square=float(chi_square),
        p_value=float(p_value),
        iman_davenport=float(f_statistic),
        p_iman_davenport=p_f,
        mean_ranks=_mean_ranks(values, algorithms, metric),
        critical_difference=nemenyi_critical_difference(k, n),
    )


def _median_per_instance(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return frame.pivot_table(
        index=["image_id", "k"], columns="algorithm", values=metric, aggfunc="median"
    ).dropna()


def _mean_ranks(values: np.ndarray, algorithms: list[str], metric: str) -> pd.Series:
    """Rank 1 is best"""
    
    ranks = sps.rankdata(-_oriented(values, metric), axis=1)
    return pd.Series(ranks.mean(axis=0), index=algorithms).sort_values()


def _iman_davenport(chi_square: float, n: int, k: int) -> tuple[float, float]:
    denominator = n * (k - 1) - chi_square
    if denominator <= 0:
        return np.inf, 0.0
    f_statistic = (n - 1) * chi_square / denominator
    return f_statistic, float(sps.f.sf(f_statistic, k - 1, (k - 1) * (n - 1)))


def compare_all(
    frame: pd.DataFrame, metric: str = "best_fitness"
) -> pd.DataFrame:
    """:func:`pairwise` for every cell with at least two algorithms"""
    
    blocks = []
    for cell, group in frame.groupby(CELL_KEYS, sort=True):
        if group.algorithm.nunique() < 2:
            continue
        rows = pairwise(group, metric)
        if not rows.empty:
            blocks.append(_tag_cell(rows, cell))
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def _tag_cell(rows: pd.DataFrame, cell: tuple) -> pd.DataFrame:
    for position, (key, value) in enumerate(zip(CELL_KEYS, cell)):
        rows.insert(position, key, value)
    return rows


def summary_table(
    frame: pd.DataFrame, metrics: tuple[str, ...] = SUMMARY_METRICS
) -> pd.DataFrame:
    grouped = frame.groupby([*CELL_KEYS, "algorithm"])
    table = grouped[list(metrics)].agg(["mean", "std"])
    table.columns = [f"{metric}_{statistic}" for metric, statistic in table.columns]
    return table.reset_index()


def to_latex(
    table: pd.DataFrame,
    metric: str,
    caption: str,
    label: str,
    higher_is_better: bool = True,
) -> str:
    """Booktabs ``mean +/- std`` by algorithm and K, best per column in bold"""
    
    means = table.pivot_table(index="algorithm", columns="k", values=f"{metric}_mean")
    stds = table.pivot_table(index="algorithm", columns="k", values=f"{metric}_std")
    best = means.max() if higher_is_better else means.min()

    lines = _latex_header(caption, label, list(means.columns))
    for algorithm in means.index:
        cells = [
            _latex_cell(means.loc[algorithm, k], stds.loc[algorithm, k], best[k])
            for k in means.columns
        ]
        lines.append(_latex_row(algorithm.replace("_", " "), cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _latex_header(caption: str, label: str, k_values: list[int]) -> list[str]:
    return [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(k_values) + "}",
        r"\toprule",
        _latex_row("Algorithm", [f"$K={k}$" for k in k_values]),
        r"\midrule",
    ]


def _latex_row(first: str, cells: list[str]) -> str:
    return first + " & " + " & ".join(cells) + r" \\"


def _latex_cell(mean: float, std: float, best: float) -> str:
    if pd.isna(mean):
        return "--"
    cell = f"{mean:.4f} $\\pm$ {std:.4f}"
    return f"\\textbf{{{cell}}}" if np.isclose(mean, best) else cell
