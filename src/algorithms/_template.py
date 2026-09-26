"""Copy-me template for a new DE variant; rules in the README."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.segmentation import decode_thresholds

FitnessFunction = Callable[[np.ndarray], np.ndarray]


def template(
    objective: FitnessFunction,
    dimensions: int,
    max_evaluations: int,
    *,
    population_size: int = 50,
    differential_weight: float = 0.5,
    crossover_rate: float = 0.9,
    lower_bound: int = 1,
    upper_bound: int = 254,
    seed: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return ``(best_thresholds, best_fitness, history_fes, history_best)``."""
    rng = np.random.default_rng(seed)

    population = rng.uniform(
        lower_bound, upper_bound, size=(population_size, dimensions)
    )

    fitness = objective(decode_thresholds(population, lower_bound, upper_bound))
    evaluations = population_size

    best_index = int(np.argmax(fitness))
    best_vector = population[best_index].copy()
    best_fitness = float(fitness[best_index])
    history_fes, history_best = [evaluations], [best_fitness]

    while evaluations < max_evaluations:
        count = min(population_size, max_evaluations - evaluations)

        trials = population[:count].copy()
        for row in range(count):
            choices = rng.choice(
                np.delete(np.arange(population_size), row), size=3, replace=False
            )
            r1, r2, r3 = (population[i] for i in choices)
            mutant = np.clip(
                r1 + differential_weight * (r2 - r3), lower_bound, upper_bound
            )
            crossover = rng.random(dimensions) < crossover_rate
            crossover[rng.integers(dimensions)] = True
            trials[row] = np.where(crossover, mutant, population[row])

        trial_fitness = objective(
            decode_thresholds(trials, lower_bound, upper_bound)
        )
        evaluations += count

        improved = trial_fitness >= fitness[:count]
        population[:count][improved] = trials[improved]
        fitness[:count][improved] = trial_fitness[improved]

        generation_best = int(np.argmax(fitness))
        if fitness[generation_best] > best_fitness:
            best_fitness = float(fitness[generation_best])
            best_vector = population[generation_best].copy()
        history_fes.append(evaluations)
        history_best.append(best_fitness)

    return (
        decode_thresholds(best_vector, lower_bound, upper_bound),
        best_fitness,
        np.asarray(history_fes, dtype=np.int64),
        np.asarray(history_best, dtype=float),
    )
