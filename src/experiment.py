"""Experiment harness: metered budgets, paired seeds, traces and resumable results."""

from __future__ import annotations

import hashlib
import os
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

from src import metrics
from src.algorithms import get as get_algorithm
from src.datasets import DATASETS, ImageRecord, load_all
from src.objectives import (
    DEFAULT_Q,
    OBJECTIVE_NAMES,
    ImageStats,
    Objective,
    build,
    exact_optimum,
)
from src.segmentation import MAX_THRESHOLD, MIN_THRESHOLD, decode_thresholds

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"

# Budget is fixed across K so time-vs-K measures cost per evaluation.
K_VALUES: tuple[int, ...] = (3, 5, 7, 9, 11, 12)
N_RUNS = 30
MAX_EVALUATIONS = 30_000

TRACE_POINTS = 201

SEED_BASE = 0xC05791

SEPARABLE_OBJECTIVES = ("otsu", "kapur")

CONTRACT_TOLERANCE = 1e-9

RESULT_ORDER = ["dataset", "image_id", "objective", "k", "algorithm", "run"]


class BudgetExceeded(Exception):
    """Treated as normal termination: the run's work so far still counts."""


def _missing_trace(points: int = TRACE_POINTS) -> np.ndarray:
    return np.full(points, np.nan, dtype=np.float32)


class CountingObjective:
    """Drop-in objective wrapper that meters, caps and traces evaluations."""

    __slots__ = ("_objective", "max_evaluations", "used", "best_fitness",
                 "best_thresholds", "_checkpoints", "_values")

    def __init__(self, objective: Objective, max_evaluations: int) -> None:
        self._objective = objective
        self.max_evaluations = int(max_evaluations)
        self.used = 0
        self.best_fitness = -np.inf
        self.best_thresholds: np.ndarray | None = None
        self._checkpoints: list[int] = []
        self._values: list[float] = []

    @property
    def remaining(self) -> int:
        return self.max_evaluations - self.used

    def __call__(self, thresholds: np.ndarray) -> np.ndarray:
        thresholds = np.atleast_2d(thresholds)
        count = thresholds.shape[0]
        if count > self.remaining:
            raise BudgetExceeded(
                f"requested {count} evaluations with {self.remaining} of "
                f"{self.max_evaluations} remaining"
            )

        fitness = np.asarray(self._objective(thresholds), dtype=np.float64)
        self.used += count
        self._record_best(thresholds, fitness)
        return fitness

    def _record_best(self, thresholds: np.ndarray, fitness: np.ndarray) -> None:
        best = int(np.argmax(fitness))
        if fitness[best] > self.best_fitness:
            self.best_fitness = float(fitness[best])
            self.best_thresholds = thresholds[best].copy()
        self._checkpoints.append(self.used)
        self._values.append(self.best_fitness)

    def trace(self, points: int = TRACE_POINTS) -> np.ndarray:
        if not self._checkpoints:
            return _missing_trace(points)
        grid = np.linspace(0, self.max_evaluations, points)
        checkpoints = np.asarray(self._checkpoints)
        values = np.asarray(self._values)
        index = np.searchsorted(checkpoints, grid, side="right") - 1
        trace = np.where(index >= 0, values[np.clip(index, 0, None)], np.nan)
        return trace.astype(np.float32)


def make_seed(image_id: str, objective: str, k: int, run: int) -> int:
    # No algorithm argument: every algorithm starts run r from the same
    # population, which the paired statistical tests rely on.

    key = f"{SEED_BASE}|{image_id}|{objective}|{k}|{run}"
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _resolve_root(results_root: Path | None) -> Path:
    return results_root or RESULTS_ROOT


@dataclass(frozen=True)
class CellSpec:
    dataset: str
    image_id: str
    algorithm: str
    objective: str
    k: int
    q: float = DEFAULT_Q
    n_runs: int = N_RUNS
    max_evaluations: int = MAX_EVALUATIONS

    @property
    def uses_q(self) -> bool:
        return self.objective == "tsallis"

    @property
    def objective_key(self) -> str:
        if self.uses_q:
            return f"tsallis_q{self.q:g}"
        return self.objective

    def relative_path(self, suffix: str) -> Path:
        return Path(
            self.dataset,
            self.algorithm,
            self.objective_key,
            f"K{self.k}",
            f"{self.image_id}{suffix}",
        )

    def table_path(self, results_root: Path) -> Path:
        return results_root / "raw" / self.relative_path(".parquet")

    def curve_path(self, results_root: Path) -> Path:
        return results_root / "curves" / self.relative_path(".npy")

    def seed(self, run: int) -> int:
        return make_seed(self.image_id, self.objective, self.k, run)


def _write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    write(temporary)
    os.replace(temporary, path)


@dataclass(frozen=True)
class CellInputs:
    image: np.ndarray
    stats: ImageStats
    masks: dict[str, np.ndarray] | None
    reference: float | None

    @classmethod
    def load(cls, spec: CellSpec, record: ImageRecord) -> CellInputs:
        image = record.load()
        stats = ImageStats.from_image(image)
        return cls(
            image=image,
            stats=stats,
            masks=record.load_ground_truth(),
            reference=_reference_optimum(spec, stats),
        )


@dataclass(frozen=True)
class _Outcome:
    status: str = "ok"
    error: str | None = None
    traceback: str | None = None
    reported_fitness: float | None = None

    def columns(self) -> dict[str, object]:
        columns: dict[str, object] = {"status": self.status, "error": self.error}
        if self.traceback is not None:
            columns["traceback"] = self.traceback
        return columns


def run_once(
    spec: CellSpec, run: int, inputs: CellInputs
) -> tuple[dict[str, object], np.ndarray]:
    """Never raises: a failed run becomes a row."""
    seed = spec.seed(run)
    objective = build(spec.objective, inputs.stats, spec.q)
    counter = CountingObjective(objective, spec.max_evaluations)

    start = time.perf_counter()
    outcome = _execute(spec, counter, seed)
    elapsed = time.perf_counter() - start

    evaluated = counter.best_thresholds is not None
    if not evaluated and outcome.status == "ok":
        outcome = replace(outcome, status="no_evaluations")
    row = {**_identity_columns(spec, run, seed), **outcome.columns()}

    if not evaluated:
        row.update({"wall_time_s": elapsed, "used_evaluations": counter.used})
        return row, _missing_trace()

    row.update(_score(inputs, counter, elapsed))
    row.update(_verification(counter.best_fitness, outcome, inputs.reference))
    return row, counter.trace()


def _identity_columns(spec: CellSpec, run: int, seed: int) -> dict[str, object]:
    return {
        "dataset": spec.dataset,
        "image_id": spec.image_id,
        "algorithm": spec.algorithm,
        "objective": spec.objective,
        "q": spec.q if spec.uses_q else None,
        "k": spec.k,
        "run": run,
        "seed": seed,
        "max_evaluations": spec.max_evaluations,
    }


def _execute(spec: CellSpec, counter: CountingObjective, seed: int) -> _Outcome:
    try:
        optimize = get_algorithm(spec.algorithm)
        _, reported_fitness, _, _ = optimize(
            counter,
            spec.k,
            spec.max_evaluations,
            lower_bound=MIN_THRESHOLD,
            upper_bound=MAX_THRESHOLD,
            seed=seed,
        )
        return _Outcome(reported_fitness=float(reported_fitness))
    except BudgetExceeded:
        return _Outcome(status="budget_exceeded")
    except Exception as error:  # one bad run must not stop a sweep
        return _Outcome(
            status="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(limit=3),
        )


def _score(
    inputs: CellInputs, counter: CountingObjective, elapsed: float
) -> dict[str, object]:
    thresholds = decode_thresholds(counter.best_thresholds)
    scores = metrics.evaluate(inputs.image, thresholds, inputs.stats, inputs.masks)
    return {
        "best_fitness": counter.best_fitness,
        "thresholds": ",".join(str(int(t)) for t in thresholds),
        "used_evaluations": counter.used,
        "wall_time_s": elapsed,
        "evaluations_per_s": counter.used / elapsed if elapsed > 0 else np.nan,
        "psnr": scores["psnr"],
        "ssim": scores["ssim"],
        "uniformity": scores["uniformity"],
        "jaccard": scores["jaccard"],
        "dice": scores["dice"],
    }


def _verification(
    best_fitness: float, outcome: _Outcome, reference: float | None
) -> dict[str, object]:
    reported = outcome.reported_fitness
    return {

        "contract_ok": reported is None
        or reported <= best_fitness + CONTRACT_TOLERANCE,
        "dp_optimum": reference,
        "gap_to_optimum": _gap_to_optimum(best_fitness, reference),
    }


def _gap_to_optimum(best_fitness: float, reference: float | None) -> float | None:
    if reference in (None, 0.0):
        return None
    return (reference - best_fitness) / abs(reference)


def run_cell(
    spec: CellSpec,
    record: ImageRecord | None = None,
    results_root: Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    root = _resolve_root(results_root)
    if not overwrite:
        cached = _load_complete_cell(spec.table_path(root), spec.n_runs)
        if cached is not None:
            return cached

    inputs = CellInputs.load(spec, record or _find_record(spec))
    frame, curves = _run_repetitions(spec, inputs)
    _save_cell(spec, root, frame, curves)
    return frame


def _run_repetitions(
    spec: CellSpec, inputs: CellInputs
) -> tuple[pd.DataFrame, np.ndarray]:
    rows, traces = zip(*(run_once(spec, run, inputs) for run in range(spec.n_runs)))
    return pd.DataFrame(list(rows)), np.vstack(traces)


def _save_cell(
    spec: CellSpec, root: Path, frame: pd.DataFrame, curves: np.ndarray
) -> None:
    _write_atomic(spec.table_path(root), lambda p: frame.to_parquet(p, index=False))
    _write_atomic(spec.curve_path(root), lambda p: np.save(p, curves))


def _load_complete_cell(table_path: Path, n_runs: int) -> pd.DataFrame | None:
    if not table_path.is_file():
        return None
    existing = pd.read_parquet(table_path)
    return existing if len(existing) == n_runs else None


def _reference_optimum(spec: CellSpec, stats: ImageStats) -> float | None:
    if spec.objective not in SEPARABLE_OBJECTIVES:
        return None
    _, value = exact_optimum(stats, spec.objective, spec.k)
    return value


def _find_record(spec: CellSpec) -> ImageRecord:
    for record in load_all():
        if record.image_id == spec.image_id and record.dataset == spec.dataset:
            return record
    raise KeyError(f"no image {spec.image_id!r} in dataset {spec.dataset!r}")


@dataclass
class Plan:
    algorithms: tuple[str, ...]
    objectives: tuple[str, ...] = OBJECTIVE_NAMES
    k_values: tuple[int, ...] = K_VALUES
    datasets: tuple[str, ...] = DATASETS
    q: float = DEFAULT_Q
    n_runs: int = N_RUNS
    max_evaluations: int = MAX_EVALUATIONS
    records: list[ImageRecord] = field(default_factory=list)

    def image_records(self) -> list[ImageRecord]:
        records = self.records or load_all()
        return [record for record in records if record.dataset in self.datasets]

    def cells(self) -> list[CellSpec]:
        return [
            CellSpec(
                dataset=record.dataset,
                image_id=record.image_id,
                algorithm=algorithm,
                objective=objective,
                k=k,
                q=self.q,
                n_runs=self.n_runs,
                max_evaluations=self.max_evaluations,
            )
            for record in self.image_records()
            for algorithm in self.algorithms
            for objective in self.objectives
            for k in self.k_values
        ]


def run_plan(
    plan: Plan,
    results_root: Path | None = None,
    n_jobs: int = -2,
    overwrite: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    root = _resolve_root(results_root)
    cells = plan.cells()
    if not overwrite:
        cells = [cell for cell in cells if not cell.table_path(root).is_file()]

    if cells:
        runner = _CellRunner(_records_by_key(plan), root, overwrite)
        _run_in_parallel(_with_progress(cells, progress), runner, n_jobs)
    return collect(root)


def _records_by_key(plan: Plan) -> dict[tuple[str, str], ImageRecord]:
    return {(r.dataset, r.image_id): r for r in plan.image_records()}


def _with_progress(cells: list[CellSpec], progress: bool) -> Iterable[CellSpec]:
    return tqdm(cells, desc="cells", unit="cell") if progress else cells


@dataclass(frozen=True)
class _CellRunner:
    records: dict[tuple[str, str], ImageRecord]
    root: Path
    overwrite: bool

    def __call__(self, cell: CellSpec) -> pd.DataFrame:
        record = self.records[(cell.dataset, cell.image_id)]
        return run_cell(cell, record, self.root, self.overwrite)


def _run_in_parallel(
    cells: Iterable[CellSpec], runner: _CellRunner, n_jobs: int
) -> None:
    with parallel_config(backend="loky", inner_max_num_threads=1):
        Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(runner)(cell) for cell in cells
        )


def collect(results_root: Path | None = None) -> pd.DataFrame:
    files = sorted((_resolve_root(results_root) / "raw").rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frame = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return frame.sort_values(RESULT_ORDER, ignore_index=True)


def load_curves(spec: CellSpec, results_root: Path | None = None) -> np.ndarray:
    return np.load(spec.curve_path(_resolve_root(results_root)))
