"""Segmentation quality metrics. All unsupervised."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from src.objectives import ImageStats, class_boundaries, class_sums, class_weights
from src.segmentation import reconstruct

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


def class_separability(stats: ImageStats, thresholds: np.ndarray) -> float:
    """Class separability"""
    if stats.variance <= 0:

        return 0.0

    thresholds = np.asarray(thresholds, dtype=np.int64)
    bounds = class_boundaries(thresholds)
    non_empty, safe = class_weights(stats, bounds)
    first = class_sums(stats.moment1, bounds)
    between = np.where(non_empty, first * first / safe, 0.0).sum() - stats.mean**2

    return float(np.clip(between / stats.variance, 0.0, 1.0))


def evaluate(
    image: np.ndarray, thresholds: np.ndarray, stats: ImageStats
) -> dict[str, object]:
    rebuilt = reconstruct(image, thresholds)
    
    return {
        "psnr": psnr(image, rebuilt),
        "ssim": ssim(image, rebuilt),
        "uniformity": uniformity(stats, thresholds),
        "class_separability": class_separability(stats, thresholds),
    }
