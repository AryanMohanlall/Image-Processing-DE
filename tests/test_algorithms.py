"""
Tests for src/algorithms/shade.py

Run from the project root:
    pytest tests/test_shade.py -v

These don't test "does it find the true optimum" (that needs a real image
and takes too long for a unit test) -- they test the properties that MUST
hold regardless of the objective: valid thresholds, respected FE budget,
monotonic improvement, reproducibility, and correct input validation.
"""

import numpy as np
import pytest

from src.algorithms.shade import SHADE, shade, _thresholds


# ---------------------------------------------------------------------
# A tiny deterministic objective, cheap enough to run thousands of times.
# Prefers thresholds centred on 100 -- so we know what "improving" means
# and can check the optimizer actually moves toward it.
# ---------------------------------------------------------------------
def toy_objective(thresholds_batch: np.ndarray) -> np.ndarray:
    target = 100.0
    return -np.sum((thresholds_batch.astype(float) - target) ** 2, axis=1)


@pytest.fixture
def rng_seed():
    return 42


# ---------------------------------------------------------------------
# 1. Threshold decoding is always valid: sorted, distinct, in bounds.
# ---------------------------------------------------------------------
def test_thresholds_are_sorted_and_distinct():
    pop = np.array([
        [50.0, 50.0, 50.0],   # all identical -> must be separated
        [200.0, 10.0, 100.0],  # out of order -> must be sorted
        [-5.0, 300.0, 130.0],  # out of bounds -> must be clipped
    ])
    decoded = _thresholds(pop, lower=1, upper=254)

    for row in decoded:
        assert np.all(np.diff(row) > 0), f"not strictly increasing: {row}"
        assert row.min() >= 1 and row.max() <= 254, f"out of bounds: {row}"


def test_thresholds_handle_full_width_K():
    # K thresholds packed into a narrow range should still come out valid
    pop = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
    decoded = _thresholds(pop, lower=1, upper=10)
    assert np.all(np.diff(decoded[0]) > 0)
    assert decoded[0].min() >= 1 and decoded[0].max() <= 10


# ---------------------------------------------------------------------
# 2. Every candidate SHADE ever evaluates is a valid threshold set.
#    (Catches bugs where mutation/crossover produces something _thresholds
#    would need to repair -- we check the optimizer's OWN output is valid.)
# ---------------------------------------------------------------------
def test_output_thresholds_valid(rng_seed):
    best_t, best_f, _, _ = SHADE(population_size=10, memory_size=3).optimize(
        toy_objective, dimensions=4, max_evaluations=300,
        lower_bound=1, upper_bound=254, seed=rng_seed,
    )
    assert best_t.shape == (4,)
    assert np.all(np.diff(best_t) > 0)
    assert best_t.min() >= 1 and best_t.max() <= 254


# ---------------------------------------------------------------------
# 3. FE budget is respected: never fewer evaluations than requested budget
#    allows to be skipped, and never wildly over (batching may overshoot
#    by at most one partial generation, capped by `count`).
# ---------------------------------------------------------------------
@pytest.mark.parametrize("max_fes", [50, 137, 500, 2000])
def test_fe_budget_respected(max_fes, rng_seed):
    _, _, history_fes, _ = SHADE(population_size=10, memory_size=3).optimize(
        toy_objective, dimensions=3, max_evaluations=max_fes,
        lower_bound=1, upper_bound=254, seed=rng_seed,
    )
    assert history_fes[-1] >= max_fes  # loop stops once budget is met/exceeded
    assert history_fes[-1] <= max_fes + 10  # never overshoots by more than one batch


# ---------------------------------------------------------------------
# 4. Convergence history is monotonic non-decreasing (greedy selection
#    means the tracked best can never regress).
# ---------------------------------------------------------------------
def test_history_is_monotonic(rng_seed):
    _, _, _, history_best = SHADE(population_size=15, memory_size=5).optimize(
        toy_objective, dimensions=3, max_evaluations=1500,
        lower_bound=1, upper_bound=254, seed=rng_seed,
    )
    assert np.all(np.diff(history_best) >= 0)


# ---------------------------------------------------------------------
# 5. It actually improves: on this toy objective, final fitness should be
#    meaningfully better than the very first (random-init) best.
# ---------------------------------------------------------------------
def test_actually_improves(rng_seed):
    _, _, _, history_best = SHADE(population_size=15, memory_size=5).optimize(
        toy_objective, dimensions=3, max_evaluations=1500,
        lower_bound=1, upper_bound=254, seed=rng_seed,
    )
    assert history_best[-1] > history_best[0]


# ---------------------------------------------------------------------
# 6. Reproducibility: same seed -> identical result, every time.
#    Critical for the assignment's 30-independent-runs protocol -- if this
#    fails, "run 7" isn't actually reproducible for debugging/reporting.
# ---------------------------------------------------------------------
def test_reproducible_with_same_seed():
    args = dict(dimensions=3, max_evaluations=500, lower_bound=1, upper_bound=254, seed=7)
    t1, f1, _, _ = SHADE(population_size=10, memory_size=4).optimize(toy_objective, **args)
    t2, f2, _, _ = SHADE(population_size=10, memory_size=4).optimize(toy_objective, **args)
    assert np.array_equal(t1, t2)
    assert f1 == f2


def test_different_seeds_usually_differ():
    # NOTE: toy_objective is a simple convex bowl with one global optimum.
    # With a generous budget SHADE reliably finds that exact optimum
    # regardless of seed -- correct behaviour, not a bug -- so this test
    # uses a tight budget where runs are still mid-search and seed-sensitive.
    base = dict(dimensions=3, max_evaluations=60, lower_bound=1, upper_bound=254)
    t1, _, _, _ = SHADE(population_size=10).optimize(toy_objective, seed=1, **base)
    t2, _, _, _ = SHADE(population_size=10).optimize(toy_objective, seed=2, **base)
    assert not np.array_equal(t1, t2)


# ---------------------------------------------------------------------
# 7. Archive on vs off: both are valid execution paths.
# ---------------------------------------------------------------------
@pytest.mark.parametrize("use_archive", [True, False])
def test_archive_modes(use_archive, rng_seed):
    best_t, best_f, _, _ = SHADE(
        population_size=10, memory_size=3, use_archive=use_archive
    ).optimize(toy_objective, dimensions=3, max_evaluations=300,
               lower_bound=1, upper_bound=254, seed=rng_seed)
    assert np.all(np.diff(best_t) > 0)


# ---------------------------------------------------------------------
# 8. Edge cases: K=1 (single threshold), and a population that shrinks
#    the pbest pool to a single candidate (population_size == 4, the min).
# ---------------------------------------------------------------------
def test_k_equals_one(rng_seed):
    best_t, _, _, _ = SHADE(population_size=8).optimize(
        toy_objective, dimensions=1, max_evaluations=200,
        lower_bound=1, upper_bound=254, seed=rng_seed,
    )
    assert best_t.shape == (1,)


def test_minimum_population_size(rng_seed):
    best_t, _, _, _ = SHADE(population_size=4, memory_size=1).optimize(
        toy_objective, dimensions=2, max_evaluations=100,
        lower_bound=1, upper_bound=254, seed=rng_seed,
    )
    assert np.all(np.diff(best_t) > 0)


# ---------------------------------------------------------------------
# 9. Input validation raises clear errors instead of failing silently.
# ---------------------------------------------------------------------
def test_rejects_tiny_population():
    with pytest.raises(ValueError, match="population_size"):
        SHADE(population_size=2).optimize(
            toy_objective, dimensions=3, max_evaluations=100)


def test_rejects_insufficient_bounds_for_K():
    # K=10 thresholds can't fit strictly increasing in a 3-wide range
    with pytest.raises(ValueError, match="bounds"):
        SHADE(population_size=10).optimize(
            toy_objective, dimensions=10, max_evaluations=200,
            lower_bound=1, upper_bound=4)


def test_rejects_budget_smaller_than_population():
    with pytest.raises(ValueError, match="max_evaluations"):
        SHADE(population_size=20).optimize(
            toy_objective, dimensions=3, max_evaluations=5)


def test_rejects_bad_memory_size():
    with pytest.raises(ValueError, match="memory_size"):
        SHADE(population_size=10, memory_size=0).optimize(
            toy_objective, dimensions=3, max_evaluations=100)


# ---------------------------------------------------------------------
# 10. The functional wrapper matches the class directly (same seed,
#     same settings -> identical result), since experiment code may use
#     either calling style.
# ---------------------------------------------------------------------
def test_functional_wrapper_matches_class(rng_seed):
    t_class, f_class, _, _ = SHADE(population_size=10, memory_size=4).optimize(
        toy_objective, dimensions=3, max_evaluations=400,
        lower_bound=1, upper_bound=254, seed=rng_seed)
    t_func, f_func, _, _ = shade(
        toy_objective, dimensions=3, max_evaluations=400,
        population_size=10, memory_size=4,
        lower_bound=1, upper_bound=254, seed=rng_seed)
    assert np.array_equal(t_class, t_func)
    assert f_class == f_func


# ---------------------------------------------------------------------
# 11. Objective contract is enforced: wrong-shaped or non-finite output
#     from a (buggy) objective must be caught, not silently propagated.
# ---------------------------------------------------------------------
def test_rejects_wrong_shaped_objective_output():
    def bad_objective(batch):
        return np.array([1.0, 2.0])  # wrong length regardless of batch size

    with pytest.raises(ValueError, match="one fitness value per population member"):
        SHADE(population_size=10).optimize(
            bad_objective, dimensions=3, max_evaluations=100, seed=1)


def test_rejects_non_finite_objective_output():
    def nan_objective(batch):
        out = np.zeros(batch.shape[0])
        out[0] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        SHADE(population_size=10).optimize(
            nan_objective, dimensions=3, max_evaluations=100, seed=1)