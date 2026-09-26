"""Otsu, Kapur and Tsallis objectives over histogram tables, plus an exact DP solver."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.segmentation import LEVELS, MAX_THRESHOLD, MIN_THRESHOLD

OBJECTIVE_NAMES: tuple[str, ...] = ("otsu", "kapur", "tsallis")

DEFAULT_Q = 0.8

Objective = Callable[[np.ndarray], np.ndarray]
ClassTerm = Callable[[np.ndarray, np.ndarray], np.ndarray]

SINGULAR_Q_TOLERANCE = 1e-9


def prefix_sum(values: np.ndarray) -> np.ndarray:
    """Leading zero, so ``S[b] - S[a]`` sums ``[a, b)``."""
    return np.concatenate(([0.0], np.cumsum(values)))


def _negative_plogp(p: np.ndarray) -> np.ndarray:
    plogp = np.zeros(p.shape, dtype=np.float64)
    np.log(p, where=p > 0, out=plogp)
    plogp *= -p
    return plogp


def _validated_histogram(histogram: np.ndarray) -> tuple[np.ndarray, float]:
    histogram = np.asarray(histogram, dtype=np.float64)
    if histogram.shape != (LEVELS,):
        raise ValueError(f"histogram must have shape ({LEVELS},)")
    n_pixels = float(histogram.sum())
    if n_pixels <= 0:
        raise ValueError("histogram is empty")
    return histogram, n_pixels


class ImageStats:
    """Prefix-sum histogram tables: any quantity for a class ``[a, b)`` is O(1)."""

    __slots__ = (
        "histogram",
        "p",
        "prob",
        "moment1",
        "moment2",
        "entropy",
        "mean",
        "variance",
        "n_pixels",
        "min_level",
        "max_level",
        "_pq_cache",
    )

    def __init__(self, histogram: np.ndarray) -> None:
        histogram, n_pixels = _validated_histogram(histogram)
        levels = np.arange(LEVELS, dtype=np.float64)
        p = histogram / n_pixels
        occupied = np.flatnonzero(histogram)

        self.histogram = histogram
        self.n_pixels = n_pixels
        self.p = p
        self.prob = prefix_sum(p)
        self.moment1 = prefix_sum(levels * p)
        self.moment2 = prefix_sum(levels * levels * p)
        self.entropy = prefix_sum(_negative_plogp(p))
        self.mean = float(self.moment1[LEVELS])
        self.variance = float(self.moment2[LEVELS] - self.mean**2)
        self.min_level = int(occupied[0])
        self.max_level = int(occupied[-1])
        self._pq_cache: dict[float, np.ndarray] = {}

    @classmethod
    def from_image(cls, image: np.ndarray) -> ImageStats:
        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise TypeError(f"expected a uint8 image, got {image.dtype}")
        return cls(np.bincount(image.ravel(), minlength=LEVELS))

    def power_cumsum(self, q: float) -> np.ndarray:
        cached = self._pq_cache.get(q)
        if cached is None:
            pq = np.zeros(LEVELS, dtype=np.float64)
            np.power(self.p, q, where=self.p > 0, out=pq)
            cached = prefix_sum(pq)
            self._pq_cache[q] = cached
        return cached


def class_boundaries(thresholds: np.ndarray) -> np.ndarray:
    """``(n, K)`` to ``(n, K+2)`` half-open bounds, ``C_0 = [0, t_1 + 1)``."""
    thresholds = np.atleast_2d(np.asarray(thresholds, dtype=np.int64))
    n = thresholds.shape[0]
    return np.concatenate(
        [
            np.zeros((n, 1), dtype=np.int64),
            thresholds + 1,
            np.full((n, 1), LEVELS, dtype=np.int64),
        ],
        axis=1,
    )


def class_sums(table: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return table[bounds[..., 1:]] - table[bounds[..., :-1]]


def class_weights(
    stats: ImageStats, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Non-empty mask, and weights with 1 in empty classes as a safe divisor.

    Masked rather than epsilon-padded: an epsilon biases each objective
    differently, e.g. ln(1e-12) = -27.6 per empty class in Kapur.
    """
    weights = class_sums(stats.prob, bounds)
    non_empty = weights > 0
    return non_empty, np.where(non_empty, weights, 1.0)


def _otsu_term(moment_sum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return moment_sum * moment_sum / weight


def _kapur_term(entropy_sum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return entropy_sum / weight + np.log(weight)


def _sum_class_terms(
    stats: ImageStats, thresholds: np.ndarray, table: np.ndarray, term: ClassTerm
) -> np.ndarray:
    bounds = class_boundaries(thresholds)
    non_empty, safe = class_weights(stats, bounds)
    terms = np.where(non_empty, term(class_sums(table, bounds), safe), 0.0)
    return terms.sum(axis=1)


def otsu(stats: ImageStats) -> Objective:
    """Evaluated as ``sum m1^2 / omega - mu_T^2``: equivalent, and avoids
    subtracting near-equal means."""
    
    mean_square = stats.mean**2

    def evaluate(thresholds: np.ndarray) -> np.ndarray:
        total = _sum_class_terms(stats, thresholds, stats.moment1, _otsu_term)
        return total - mean_square

    return evaluate


def kapur(stats: ImageStats) -> Objective:
    """``H_j = (E[b] - E[a]) / omega_j + ln(omega_j)``: a lookup with no ``0 log 0``."""

    def evaluate(thresholds: np.ndarray) -> np.ndarray:
        return _sum_class_terms(stats, thresholds, stats.entropy, _kapur_term)

    return evaluate


def tsallis(stats: ImageStats, q: float = DEFAULT_Q) -> Objective:
    """``S = sum_j S_j + (1 - q) prod_j S_j``, the single-product convention
    (de Albuquerque et al. 2004), not the recursive pairwise form."""

    if abs(q - 1.0) < SINGULAR_Q_TOLERANCE:
        raise ValueError(
            "Tsallis entropy is singular at q=1. it converges to Kapur's "
            "entropy in the limit, so use the 'kapur' objective instead"
        )
    power = stats.power_cumsum(q)
    scale = 1.0 / (q - 1.0)
    coupling = 1.0 - q

    def evaluate(thresholds: np.ndarray) -> np.ndarray:
        bounds = class_boundaries(thresholds)
        non_empty, safe = class_weights(stats, bounds)
        partial = class_sums(power, bounds)
        entropies = np.where(non_empty, (1.0 - partial / safe**q) * scale, 0.0)
        
        np.maximum(entropies, 0.0, out=entropies)
        return entropies.sum(axis=1) + coupling * entropies.prod(axis=1)

    return evaluate


_FACTORIES: dict[str, Callable[[ImageStats, float], Objective]] = {
    "otsu": lambda stats, q: otsu(stats),
    "kapur": lambda stats, q: kapur(stats),
    "tsallis": tsallis,
}


def build(name: str, stats: ImageStats, q: float = DEFAULT_Q) -> Objective:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise KeyError(
            f"unknown objective {name!r}; available: {OBJECTIVE_NAMES}"
        ) from None
    return factory(stats, q)


_SEPARABLE_TERMS: dict[str, tuple[Callable[[ImageStats], np.ndarray], ClassTerm]] = {
    "otsu": (lambda stats: stats.moment1, _otsu_term),
    "kapur": (lambda stats: stats.entropy, _kapur_term),
}


def _pairwise_sums(table: np.ndarray) -> np.ndarray:
    """``S[a, b]`` = the table's total over ``[a, b)``, for every pair."""
    return table[None, :] - table[:, None]


def _interval_terms(stats: ImageStats, name: str) -> np.ndarray:
    """Objective term for every class ``[a, b)``; -inf where the class is empty."""
    try:
        table_of, term = _SEPARABLE_TERMS[name]
    except KeyError:
        raise ValueError(
            f"{name!r} is not separable over classes, so no dynamic program "
            "exists; Tsallis' product term couples all classes together"
        ) from None
    weights = _pairwise_sums(stats.prob)
    valid = weights > 0
    safe = np.where(valid, weights, 1.0)
    return np.where(valid, term(_pairwise_sums(table_of(stats)), safe), -np.inf)


def exact_optimum(
    stats: ImageStats,
    name: str,
    k: int,
    lower: int = MIN_THRESHOLD,
    upper: int = MAX_THRESHOLD,
) -> tuple[np.ndarray, float]:
    """Globally optimal thresholds and value for Otsu or Kapur, in O(L^2 K)."""
    if k < 1:
        raise ValueError("k must be at least 1")
    terms = _interval_terms(stats, name)

    feasible = np.zeros(LEVELS + 1, dtype=bool)
    feasible[lower + 1 : upper + 2] = True
    best, choices = _best_partial_partitions(terms, feasible, k)

    totals = best + terms[:, LEVELS]
    final = int(np.argmax(totals))
    value = float(totals[final])

    if name == "otsu":
        value -= stats.mean**2
    return _backtrack(final, choices), value


def _best_partial_partitions(
    terms: np.ndarray, feasible: np.ndarray, k: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """``best[b]``: best total of k classes covering ``[0, b)``, plus argmax choices."""

    n_bounds = feasible.size
    best = np.where(feasible, terms[0], -np.inf)
    choices: list[np.ndarray] = []
    for _ in range(1, k):
        candidates = best[:, None] + terms
        previous = np.argmax(candidates, axis=0)
        best = np.where(feasible, candidates[previous, np.arange(n_bounds)], -np.inf)
        choices.append(previous)

    return best, choices


def _backtrack(final: int, choices: list[np.ndarray]) -> np.ndarray:
    bounds = [final]
    
    for previous in reversed(choices):
        bounds.append(int(previous[bounds[-1]]))

    return np.array(sorted(b - 1 for b in bounds), dtype=np.int64)
