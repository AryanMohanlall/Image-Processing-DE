import numpy as np
import pytest

from src.algorithms.jade import jade
from src.algorithms import available, get


def test_jade_matches_project_contract() -> None:
    objective = lambda population: population.sum(axis=1)

    thresholds, best_fitness, fes, best_history = jade(
        objective,
        3,
        40,
        population_size=10,
        lower_bound=1,
        upper_bound=10,
        seed=7,
    )

    assert thresholds.shape == (3,)
    assert thresholds.dtype.kind in {"i", "u"}
    assert np.all(1 <= thresholds) and np.all(thresholds <= 10)
    assert np.isfinite(best_fitness)
    assert len(fes) >= 1
    assert len(best_history) == len(fes)
    assert fes[-1] <= 40


@pytest.mark.parametrize("name", ["shade", "lshade"])
@pytest.mark.parametrize("budget", [10, 11, 137])
def test_success_history_contract(name, budget):
    assert name in available()
    evaluated = []

    def objective(population):
        assert population.ndim == 2
        assert np.all(np.diff(population, axis=1) > 0)
        assert np.all((population >= 1) & (population <= 10))
        evaluated.extend(population.copy())
        return -np.square(population - [2, 5, 8]).sum(axis=1)

    optimize = get(name)
    result = optimize(objective, 3, budget, population_size=10,
                      lower_bound=1, upper_bound=10, seed=7)
    thresholds, fitness, fes, history = result
    assert len(evaluated) == budget == fes[-1]
    assert np.all(np.diff(fes) > 0)
    assert np.all(np.diff(history) >= 0)
    assert fitness == -np.square(thresholds - [2, 5, 8]).sum()
    assert fitness == max(-np.square(row - [2, 5, 8]).sum() for row in evaluated)
    repeated = optimize(objective, 3, budget, population_size=10,
                        lower_bound=1, upper_bound=10, seed=7)
    for first, second in zip(result, repeated):
        np.testing.assert_array_equal(first, second)
