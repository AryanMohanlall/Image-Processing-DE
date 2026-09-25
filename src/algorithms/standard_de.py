"""Standard Differential Evolution using the DE/rand/1/bin strategy."""

# DE      = Differential Evolution
# rand    = base vector is chosen randomly
# 1       = one difference vector is used
# bin     = binomial crossover

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.segmentation import decode_thresholds


# A supplied objective accepts an (n, K) array of threshold sets and returns
# n fitness scores, one per candidate; larger scores are better.
# TODO (dataset/objective integration): convert image I to grayscale, count
# h(i) for all L gray levels, and compute p_i = h(i) / (M * N). Prepare this
# histogram once per image and make it available to the supplied objective.
FitnessFunction = Callable[[np.ndarray], np.ndarray]


def _validate_inputs(
    dimensions: int,
    population_size: int,
    differential_weight: float,
    crossover_rate: float,
    max_evaluations: int,
    lower_bound: int,
    upper_bound: int,
) -> None:
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")
    if population_size < 4:
        raise ValueError("population_size must be at least 4 for DE/rand/1")
    if not 0.0 <= differential_weight <= 2.0:
        raise ValueError("differential_weight must be in [0, 2]")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1]")
    if max_evaluations < population_size:
        raise ValueError("max_evaluations must cover the initial population")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    if dimensions > upper_bound - lower_bound + 1:
        raise ValueError("the bounds do not contain enough distinct thresholds")


def _thresholds(population: np.ndarray, lower: int, upper: int) -> np.ndarray:
    """Convert real DE vectors to sorted, distinct integer thresholds."""
    return decode_thresholds(population, lower, upper)


def _evaluate(
    objective: FitnessFunction,
    population: np.ndarray,
    lower: int,
    upper: int,
) -> np.ndarray:
    # TODO (objective integration): calculate Otsu, Kapur, or Tsallis fitness
    fitness = np.asarray(objective(_thresholds(population, lower, upper)), dtype=float)
    if fitness.shape != (population.shape[0],):
        raise ValueError(
            "objective must return one fitness value per population member"
        )
    if not np.all(np.isfinite(fitness)):
        raise ValueError("objective returned a non-finite fitness value")
    return fitness


class StandardDE:
    """Maximising DE/rand/1/bin for multilevel image thresholds."""

    def __init__(
        self,
        population_size: int = 50,
        differential_weight: float = 0.5,
        crossover_rate: float = 0.9,
    ) -> None:
        self.population_size = int(population_size)
        self.differential_weight = float(differential_weight)
        self.crossover_rate = float(crossover_rate)

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
        """Run DE and return thresholds, fitness, FE checkpoints and convergence.

        ``objective`` must accept an ``(n, dimensions)`` integer array and return
        an ``(n,)`` array. Fitness is maximised. The final two arrays record the
        best-so-far fitness after initialisation and after each generation.
        """
        # dimensions is the assignment's K, the number of thresholds (not
        # classes). Experiments should use K in {3, 5, 7, 9, 11, 12}.
        # Bounds default to [1,254] for L=256. For another grayscale depth,
        # the caller must supply lower_bound=1 and upper_bound=L-2;
        # this algorithm does not receive the image or infer L.
        dimensions = int(dimensions)
        max_evaluations = int(max_evaluations)
        lower_bound = int(lower_bound)
        upper_bound = int(upper_bound)
        _validate_inputs(
            dimensions,
            self.population_size,
            self.differential_weight,
            self.crossover_rate,
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

        best_index = int(np.argmax(fitness))
        best_vector = population[best_index].copy()
        best_fitness = float(fitness[best_index])
        history_fes = [evaluations]
        history_best = [best_fitness]

        all_indices = np.arange(self.population_size)
        while evaluations < max_evaluations:
            count = min(self.population_size, max_evaluations - evaluations)
            targets = rng.choice(self.population_size, size=count, replace=False)
            trials = population[targets].copy()

            for row, target in enumerate(targets):
                candidates = all_indices[all_indices != target]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                mutant = population[r1] + self.differential_weight * (
                    population[r2] - population[r3]
                )
                mutant = np.clip(mutant, lower_bound, upper_bound)

                crossover = rng.random(dimensions) < self.crossover_rate
                crossover[rng.integers(dimensions)] = True
                trials[row] = np.where(crossover, mutant, population[target])

            trial_fitness = _evaluate(
                objective, trials, lower_bound, upper_bound
            )
            evaluations += count
            improved = trial_fitness >= fitness[targets]
            accepted_targets = targets[improved]
            population[accepted_targets] = trials[improved]
            fitness[accepted_targets] = trial_fitness[improved]

            generation_best = int(np.argmax(fitness))
            if fitness[generation_best] > best_fitness:
                best_fitness = float(fitness[generation_best])
                best_vector = population[generation_best].copy()
            history_fes.append(evaluations)
            history_best.append(best_fitness)

        # TODO (segmentation integration): use these K thresholds to assign
        # image pixels to K+1 classes, with the objective's boundary convention.
        best_thresholds = _thresholds(
            best_vector[None, :], lower_bound, upper_bound
        )[0]
        return (
            best_thresholds,
            best_fitness,
            np.asarray(history_fes, dtype=np.int64),
            np.asarray(history_best, dtype=float),
        )


def standard_de(
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
    """Convenience wrapper around :class:`StandardDE`."""
    return StandardDE(
        population_size, differential_weight, crossover_rate
    ).optimize(
        objective,
        dimensions,
        max_evaluations,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        seed=seed,
    )
