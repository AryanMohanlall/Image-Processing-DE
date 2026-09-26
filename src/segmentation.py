"""Threshold decoding and segmented-image reconstruction."""

from __future__ import annotations

import numpy as np

LEVELS = 256

MIN_THRESHOLD = 1
MAX_THRESHOLD = LEVELS - 2


def decode_thresholds(
    population: np.ndarray,
    lower: int = MIN_THRESHOLD,
    upper: int = MAX_THRESHOLD,
) -> np.ndarray:
    """Real DE vectors to strictly increasing integer thresholds in bounds"""

    single = population.ndim == 1

    if single:
        population = population[None, :]

    decoded = np.sort(np.rint(population).astype(np.int64), axis=1)
    decoded = np.clip(decoded, lower, upper)
    _separate_upwards(decoded, lower)
    _separate_downwards(decoded, upper)

    return decoded[0] if single else decoded


def _separate_upwards(decoded: np.ndarray, lower: int) -> None:
    for column in range(decoded.shape[1]):
        minimum = lower + column
        
        if column:
            minimum = np.maximum(minimum, decoded[:, column - 1] + 1)
        decoded[:, column] = np.maximum(decoded[:, column], minimum)


def _separate_downwards(decoded: np.ndarray, upper: int) -> None:
    last = decoded.shape[1] - 1

    for column in range(last, -1, -1):
        maximum = upper - (last - column)

        if column < last:
            maximum = np.minimum(maximum, decoded[:, column + 1] - 1)
        decoded[:, column] = np.minimum(decoded[:, column], maximum)


def class_lookup(thresholds: np.ndarray) -> np.ndarray:
    """Class index of each grey level, so an image maps to classes in one gather"""

    thresholds = np.asarray(thresholds, dtype=np.int64)
    return np.searchsorted(thresholds, np.arange(LEVELS), side="left")


def class_count(thresholds: np.ndarray) -> int:
    return np.asarray(thresholds).size + 1


def class_means(thresholds: np.ndarray, histogram: np.ndarray) -> np.ndarray:
    """Empty classes get a mean of 0; no pixel ever indexes them"""

    thresholds = np.asarray(thresholds, dtype=np.int64)
    histogram = np.asarray(histogram, dtype=np.float64)
    levels = np.arange(LEVELS, dtype=np.float64)

    classes = class_lookup(thresholds)
    n_classes = class_count(thresholds)
    counts = np.bincount(classes, weights=histogram, minlength=n_classes)
    totals = np.bincount(classes, weights=histogram * levels, minlength=n_classes)

    means = np.zeros(n_classes, dtype=np.float64)
    np.divide(totals, counts, out=means, where=counts > 0)
    return means


def _paint(
    image: np.ndarray, thresholds: np.ndarray, class_values: np.ndarray
) -> np.ndarray:
    """Every pixel replaced by its class's entry in class_values"""

    return class_values[class_lookup(thresholds)][np.asarray(image)]


def reconstruct(image: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Class-mean image for the metrics, as unrounded float64"""

    image = np.asarray(image)
    histogram = np.bincount(image.ravel(), minlength=LEVELS).astype(np.float64)
    return _paint(image, thresholds, class_means(thresholds, histogram))


def label_map(image: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return class_lookup(thresholds)[np.asarray(image)].astype(np.uint8)


def display_image(image: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Classes spread evenly over the grey range for figures only"""

    thresholds = np.asarray(thresholds, dtype=np.int64)
    spread = np.rint(
        np.linspace(0, LEVELS - 1, class_count(thresholds))
    ).astype(np.uint8)
    return _paint(image, thresholds, spread)
