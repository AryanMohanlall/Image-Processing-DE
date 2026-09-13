"""Late Acceptance Differential Evolution using DE/rand/1/bin."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


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
    history_length: int,
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
    if history_length < 1:
        raise ValueError("history_length must be at least 1")
    if max_evaluations < population_size:
        raise ValueError("max_evaluations must cover the initial population")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    if dimensions > upper_bound - lower_bound + 1:
        raise ValueError("the bounds do not contain enough distinct thresholds")


def _thresholds(population: np.ndarray, lower: int, upper: int) -> np.ndarray:
    """Convert real DE vectors to sorted, distinct integer thresholds."""
    # DE searches with real-valued vectors internally. Decode before every
    # evaluation and at return so the objective always receives K integers.
    decoded = np.sort(np.rint(population).astype(np.int64), axis=1)
    decoded = np.clip(decoded, lower, upper)
    dimensions = decoded.shape[1]

    # Sorting alone allows duplicates. These passes enforce strict ordering
    # while retaining K thresholds within the bounds. With lower=1 and
    # upper=L-2, this satisfies 0 < t_1 < ... < t_K < L-1.
    for column in range(dimensions):
        minimum = lower + column
        if column:
            minimum = np.maximum(minimum, decoded[:, column - 1] + 1)
        decoded[:, column] = np.maximum(decoded[:, column], minimum)
    for column in range(dimensions - 1, -1, -1):
        maximum = upper - (dimensions - 1 - column)
        if column < dimensions - 1:
            maximum = np.minimum(maximum, decoded[:, column + 1] - 1)
        decoded[:, column] = np.minimum(decoded[:, column], maximum)

    return decoded


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


class LateAcceptanceDE:
    """Maximising DE/rand/1/bin with per-individual late acceptance."""

    def __init__(
        self,
        population_size: int = 50,
        differential_weight: float = 0.5,
        crossover_rate: float = 0.9,
        history_length: int = 20,
    ) -> None:
        self.population_size = int(population_size)
        self.differential_weight = float(differential_weight)
        self.crossover_rate = float(crossover_rate)
        self.history_length = int(history_length)

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
        """Run LADE and return thresholds, fitness, FE checkpoints and convergence.

        ``objective`` must accept an ``(n, dimensions)`` integer array and return
        an ``(n,)`` array. A trial is accepted if it is no worse than its parent
        or no worse than that individual's value in the current history slot.
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
            self.history_length,
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

        # Section 4, step 1 -- Fitness History Array H:
        # Adapt the single history buffer to a population by giving each
        # individual i its own row H_i of length L = history_length.
        # This L is a history length, not the number of grayscale levels.
        # Initially every slot contains that individual's initial fitness.
        fitness_history = np.repeat(
            fitness[:, None], self.history_length, axis=1
        )
        # Start at generation G=0; this index represents G mod L.
        history_slot = 0

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
            # Section 4, step 2 -- Read the historical benchmark H_i[G mod L]
            # before overwriting it. A slot is revisited every L generations;
            # see the update below for the benchmark-retention policy.
            late_fitness = fitness_history[targets, history_slot]
            # Section 4.1 -- Selection formula:
            # trial_fitness = f(u_i^G); fitness[targets] = f(x_i^G).
            # The PDF uses <= for minimization. Our objectives are maximized,
            # so use >= in BOTH comparisons; | is the elementwise logical OR.
            # A trial worse than its parent can pass the historical comparison.
            accepted = (trial_fitness >= fitness[targets]) | (
                trial_fitness >= late_fitness
            )
            # Accepted: x_i^(G+1) = u_i^G, with the corresponding trial fitness.
            # Rejected: no assignment, so x_i^(G+1) = x_i^G and fitness is kept.
            accepted_targets = targets[accepted]
            population[accepted_targets] = trials[accepted]
            fitness[accepted_targets] = trial_fitness[accepted]

            # Section 4.1 -- Write the benchmark back AFTER selection.
            # This implementation retains the better of the old slot benchmark
            # and the resulting current fitness: H_i[slot] = max(H_i[slot], f_i).
            # This is a best-per-slot history policy, rather than unconditionally
            # storing current fitness. Therefore H_i[slot] can retain a value
            # older than L generations; it is not an exact rolling fitness log.
            fitness_history[:, history_slot] = np.maximum(
                fitness_history[:, history_slot], fitness
            )
            # Advance to (G+1) mod L; modulo wraps around the fixed-size buffer.
            history_slot = (history_slot + 1) % self.history_length

            # Preserve the best solution seen throughout the run separately:
            # late acceptance can replace a parent with a worse trial.
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


LADE = LateAcceptanceDE


def late_acceptance_de(
    objective: FitnessFunction,
    dimensions: int,
    max_evaluations: int,
    *,
    population_size: int = 50,
    differential_weight: float = 0.5,
    crossover_rate: float = 0.9,
    history_length: int = 20,
    lower_bound: int = 1,
    upper_bound: int = 254,
    seed: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Convenience wrapper around :class:`LateAcceptanceDE`."""
    return LateAcceptanceDE(
        population_size,
        differential_weight,
        crossover_rate,
        history_length,
    ).optimize(
        objective,
        dimensions,
        max_evaluations,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        seed=seed,
    )
