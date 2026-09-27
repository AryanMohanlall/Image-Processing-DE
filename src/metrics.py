"""Segmentation quality: reconstruction metrics and ground-truth overlap."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from src.objectives import (
    ImageStats,
    class_boundaries,
    class_sums,
    class_weights,
    prefix_sum,
)
from src.segmentation import class_count, label_map, reconstruct

MAX_INTENSITY = 255.0

SSIM_K1 = 0.01
SSIM_K2 = 0.03
SSIM_SIGMA = 1.5
SSIM_TRUNCATE = 3.5
SSIM_WINDOW = 2 * int(SSIM_TRUNCATE * SSIM_SIGMA + 0.5) + 1
SSIM_BORDER = (SSIM_WINDOW - 1) // 2
SSIM_C1 = (SSIM_K1 * MAX_INTENSITY) ** 2
SSIM_C2 = (SSIM_K2 * MAX_INTENSITY) ** 2
SSIM_BESSEL = SSIM_WINDOW**2 / (SSIM_WINDOW**2 - 1.0)


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """``inf`` for an exact reconstruction, reported rather than capped."""
    error = original.astype(np.float64) - reconstructed
    mse = float(np.mean(error * error))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(MAX_INTENSITY * MAX_INTENSITY / mse)


def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    x = original.astype(np.float64)
    y = reconstructed.astype(np.float64)

    mean_x, mean_y = _smooth(x), _smooth(y)
    variance_x = SSIM_BESSEL * (_smooth(x * x) - mean_x * mean_x)
    variance_y = SSIM_BESSEL * (_smooth(y * y) - mean_y * mean_y)
    covariance = SSIM_BESSEL * (_smooth(x * y) - mean_x * mean_y)

    similarity = ((2 * mean_x * mean_y + SSIM_C1) * (2 * covariance + SSIM_C2)) / (
        (mean_x * mean_x + mean_y * mean_y + SSIM_C1)
        * (variance_x + variance_y + SSIM_C2)
    )
    return float(_crop_border(similarity).mean())


def _smooth(values: np.ndarray) -> np.ndarray:
    return gaussian_filter(values, SSIM_SIGMA, truncate=SSIM_TRUNCATE, mode="reflect")


def _crop_border(values: np.ndarray) -> np.ndarray:
    return values[SSIM_BORDER:-SSIM_BORDER, SSIM_BORDER:-SSIM_BORDER]


def uniformity(stats: ImageStats, thresholds: np.ndarray) -> float:
    """Uses the ``2K`` convention (Sathya & Kayalvizhi 2011), not ``2(K+1)``."""
    spread = float(stats.max_level - stats.min_level)
    if spread <= 0:
        return 1.0

    thresholds = np.asarray(thresholds, dtype=np.int64)
    bounds = class_boundaries(thresholds)
    non_empty, safe = class_weights(stats, bounds)
    first = class_sums(stats.moment1, bounds)
    second = class_sums(stats.moment2, bounds)
    # Clamped: cancellation gives ~-1e-13 for a single-level class.
    scatter = np.maximum(np.where(non_empty, second - first * first / safe, 0.0), 0.0)

    return float(1.0 - 2.0 * thresholds.size * scatter.sum() / (spread * spread))


@dataclass(frozen=True)
class BandMatch:
    organ: str
    first_class: int
    last_class: int
    jaccard: float
    dice: float


def match_band(
    labels: np.ndarray, mask: np.ndarray, n_classes: int, organ: str = ""
) -> BandMatch:
    """Best-Jaccard contiguous run of classes, with the Dice of that same run.

    Only contiguous runs: thresholding partitions intensity, so an organ is an
    intensity band. Arbitrary class subsets would make the metric supervised.
    """
    count_prefix, hit_prefix = _run_totals(labels, mask, n_classes)
    truth = float(mask.sum())

    best = BandMatch(organ, 0, 0, -1.0, 0.0)
    for first, last in _contiguous_runs(n_classes):
        intersection = hit_prefix[last + 1] - hit_prefix[first]
        size = count_prefix[last + 1] - count_prefix[first]
        jaccard = _ratio_or_zero(intersection, truth + size - intersection)
        if jaccard > best.jaccard:
            dice = _ratio_or_zero(2.0 * intersection, truth + size)
            best = BandMatch(organ, first, last, jaccard, dice)
    return best


def _run_totals(
    labels: np.ndarray, mask: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Prefix sums of pixels and mask hits per class, for O(1) run lookups."""
    counts = np.bincount(labels.ravel(), minlength=n_classes).astype(np.float64)
    hits = np.bincount(
        labels.ravel(), weights=mask.ravel().astype(np.float64), minlength=n_classes
    )
    return prefix_sum(counts), prefix_sum(hits)


def _contiguous_runs(n_classes: int) -> Iterator[tuple[int, int]]:
    for first in range(n_classes):
        for last in range(first, n_classes):
            yield first, last


def _ratio_or_zero(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def overlap_scores(
    image: np.ndarray,
    thresholds: np.ndarray,
    masks: dict[str, np.ndarray] | None,
) -> dict[str, object] | None:
    """Organs are matched independently, so their bands may overlap."""
    if not masks:
        return None

    thresholds = np.asarray(thresholds, dtype=np.int64)
    labels = label_map(image, thresholds)
    n_classes = class_count(thresholds)

    matches = []
    for organ, mask in masks.items():
        _require_aligned(organ, mask, image)
        matches.append(match_band(labels, mask, n_classes, organ))
    return _summarise_matches(matches)


def _require_aligned(organ: str, mask: np.ndarray, image: np.ndarray) -> None:
    if mask.shape != image.shape:
        raise ValueError(
            f"ground-truth mask for {organ!r} is {mask.shape} but the image "
            f"is {image.shape}; masks must be pixel-aligned"
        )


def _summarise_matches(matches: list[BandMatch]) -> dict[str, object]:
    return {
        "jaccard": float(np.mean([m.jaccard for m in matches])),
        "dice": float(np.mean([m.dice for m in matches])),
        "per_organ": {
            m.organ: {
                "jaccard": m.jaccard,
                "dice": m.dice,
                "band": (m.first_class, m.last_class),
            }
            for m in matches
        },
    }


# Overlap columns stay present without ground truth, keeping the schema stable
_NO_OVERLAP: dict[str, object] = {"jaccard": None, "dice": None, "per_organ": None}


def evaluate(
    image: np.ndarray,
    thresholds: np.ndarray,
    stats: ImageStats,
    masks: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    rebuilt = reconstruct(image, thresholds)
    return {
        "psnr": psnr(image, rebuilt),
        "ssim": ssim(image, rebuilt),
        "uniformity": uniformity(stats, thresholds),
        **(overlap_scores(image, thresholds, masks) or _NO_OVERLAP),
    }
