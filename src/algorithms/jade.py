"""This module implment JADE.

Based on the logic in the JADE paper, these are implmented: current-to-pbest/1 mutation, optional archive,
binomial crossover, and adaptive F/CR updates.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.segmentation import decode_thresholds

__all__ = ["JADEResult", "JADE", "jade", "decode_thresholds", "optimize_thresholds"]

FitnessFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class JADEResult:
    """Compatibility container for the original standalone API."""

    best_vector: np.ndarray
    best_fitness: float
    n_fes: int
    n_generations: int
    elapsed_seconds: float
    history_fes: np.ndarray
    history_best: np.ndarray
    mu_f: float
    mu_cr: float
    archive_size: int
    seed: int | None
    thresholds: np.ndarray | None = None


def _validate_inputs(
    dimensions: int,
    population_size: int,
    p: float,
    c: float,
    max_evaluations: int,
    lower_bound: int,
    upper_bound: int,
) -> None:
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    if not 0.0 < c <= 1.0:
        raise ValueError("c must be in (0, 1]")
    if max_evaluations < population_size:
        raise ValueError("max_evaluations must cover the initial population")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    if dimensions > upper_bound - lower_bound + 1:
        raise ValueError("the bounds do not contain enough distinct thresholds")


def _evaluate(
    objective: FitnessFunction,
    population: np.ndarray,
    lower_bound: int,
    upper_bound: int,
) -> np.ndarray:
    decoded = decode_thresholds(population, lower_bound, upper_bound)
    fitness = np.asarray(objective(decoded), dtype=np.float64)
    if fitness.shape != (population.shape[0],):
        raise ValueError("objective must return one fitness value per population member")
    if not np.all(np.isfinite(fitness)):
        raise ValueError("objective returned a non-finite fitness value")
    return fitness


class JADE:
    """Maximising JADE/current-to-pbest/1 with archive and adaptive F/CR."""

    def __init__(
        self,
        population_size: int = 50,
        p: float = 0.05,
        c: float = 0.1,
        use_archive: bool = True,
    ) -> None:
        self.population_size = int(population_size)
        self.p = float(p)
        self.c = float(c)
        self.use_archive = bool(use_archive)

    def optimize(
        self,
        objective: FitnessFunction,
        dimensions: int,
        max_evaluations: int,
        *,
        lower_bound: int = 1,
        upper_bound: int = 254,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        dimensions = int(dimensions)
        max_evaluations = int(max_evaluations)
        lower_bound = int(lower_bound)
        upper_bound = int(upper_bound)
        _validate_inputs(
            dimensions,
            self.population_size,
            self.p,
            self.c,
            max_evaluations,
            lower_bound,
            upper_bound,
        )

        rng = np.random.default_rng(seed)
        population = rng.uniform(
            lower_bound,
            upper_bound,
            size=(self.population_size, dimensions),
        )
        fitness = _evaluate(objective, population, lower_bound, upper_bound)
        evaluations = self.population_size

        archive: list[np.ndarray] = []
        mu_f = 0.5
        mu_cr = 0.5
        best_index = int(np.argmax(fitness))
        best_vector = population[best_index].copy()
        best_fitness = float(fitness[best_index])
        history_fes = [evaluations]
        history_best = [best_fitness]

        while evaluations < max_evaluations:
            parents = population.copy()
            parent_fitness = fitness.copy()
            next_population = parents.copy()
            next_fitness = parent_fitness.copy()

            ranked = np.argsort(-parent_fitness, kind="stable")
            n_pbest = max(1, int(np.ceil(self.p * self.population_size)))
            successful_f: list[float] = []
            successful_cr: list[float] = []

            for target in rng.permutation(self.population_size):
                if evaluations >= max_evaluations:
                    break

                cr = float(np.clip(rng.normal(mu_cr, 0.1), 0.0, 1.0))
                f = mu_f + 0.1 * rng.standard_cauchy()
                while not np.isfinite(f) or f <= 0.0:
                    f = mu_f + 0.1 * rng.standard_cauchy()
                f = float(min(f, 1.0))

                pbest_index = int(rng.choice(ranked[:n_pbest]))
                r1 = int(rng.integers(self.population_size))
                while r1 == target:
                    r1 = int(rng.integers(self.population_size))

                archive_count = len(archive) if self.use_archive else 0
                donor_index = int(rng.integers(self.population_size + archive_count))
                while donor_index == target or donor_index == r1:
                    donor_index = int(rng.integers(self.population_size + archive_count))

                if self.use_archive and archive_count:
                    if donor_index < self.population_size:
                        donor = parents[donor_index]
                    else:
                        donor = np.asarray(archive[donor_index - self.population_size], dtype=float)
                else:
                    donor = parents[donor_index]

                mutant = (
                    parents[target]
                    + f * (parents[pbest_index] - parents[target])
                    + f * (parents[r1] - donor)
                )
                mutant = np.clip(mutant, lower_bound, upper_bound)

                crossover = rng.random(dimensions) < cr
                crossover[rng.integers(dimensions)] = True
                trial = np.where(crossover, mutant, parents[target])

                score = float(_evaluate(objective, trial[None, :], lower_bound, upper_bound)[0])
                evaluations += 1

                if score > parent_fitness[target]:
                    if self.use_archive:
                        archive.append(parents[target].copy())
                    next_population[target] = trial
                    next_fitness[target] = score
                    successful_f.append(f)
                    successful_cr.append(cr)

                if score > best_fitness:
                    best_fitness = score
                    best_vector = trial.copy()
                history_fes.append(evaluations)
                history_best.append(best_fitness)

            population = next_population
            fitness = next_fitness

            if self.use_archive and len(archive) > self.population_size:
                keep = rng.choice(
                    len(archive), self.population_size, replace=False
                )
                archive = [archive[index] for index in keep]

            if successful_f:
                sf = np.asarray(successful_f, dtype=np.float64)
                scr = np.asarray(successful_cr, dtype=np.float64)
                mu_f = (1.0 - self.c) * mu_f + self.c * (
                    float(np.sum(sf * sf) / np.sum(sf))
                )
                mu_cr = (1.0 - self.c) * mu_cr + self.c * float(np.mean(scr))

        return (
            _thresholds(best_vector[None, :], lower_bound, upper_bound)[0],
            float(best_fitness),
            np.asarray(history_fes, dtype=np.int64),
            np.asarray(history_best, dtype=np.float64),
        )


def _thresholds(population: np.ndarray, lower_bound: int, upper_bound: int) -> np.ndarray:
    return decode_thresholds(population, lower_bound, upper_bound)


def jade(
    objective: FitnessFunction,
    dimensions: int,
    max_evaluations: int,
    *,
    population_size: int = 50,
    p: float = 0.05,
    c: float = 0.1,
    use_archive: bool = True,
    lower_bound: int = 1,
    upper_bound: int = 254,
    seed: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Project-compatible JADE entry point."""
    return JADE(
        population_size=population_size,
        p=p,
        c=c,
        use_archive=use_archive,
    ).optimize(
        objective,
        dimensions,
        max_evaluations,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        seed=seed,
    )


def optimize_thresholds(
    objective: Callable[[np.ndarray], float],
    *,
    k: int,
    max_fes: int,
    n_levels: int = 256,
    population_size: int = 50,
    p: float = 0.05,
    c: float = 0.1,
    use_archive: bool = True,
    seed: int | None = None,
) -> JADEResult:
    """Standalone threshold-optimization wrapper retained for compatibility."""
    if not 1 <= k <= n_levels - 2:
        raise ValueError("k must satisfy 1 <= k <= n_levels - 2")

    thresholds, best_fitness, history_fes, history_best = jade(
        lambda population: objective(decode_thresholds(population, n_levels)),
        k,
        max_fes,
        population_size=population_size,
        p=p,
        c=c,
        use_archive=use_archive,
        lower_bound=1,
        upper_bound=n_levels - 2,
        seed=seed,
    )
    return JADEResult(
        best_vector=thresholds.copy(),
        best_fitness=float(best_fitness),
        n_fes=int(history_fes[-1]),
        n_generations=int(len(history_fes) - 1),
        elapsed_seconds=0.0,
        history_fes=np.asarray(history_fes, dtype=np.int64),
        history_best=np.asarray(history_best, dtype=np.float64),
        mu_f=0.5,
        mu_cr=0.5,
        archive_size=0,
        seed=seed,
        thresholds=thresholds.copy(),
    )

