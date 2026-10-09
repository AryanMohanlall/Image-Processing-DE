import numpy as np

from src.algorithms.jade import jade


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
