# """
# Test SHADE & L-SHADE with Otsu, Kapur, and Tsallis objectives.

# Usage:
#     python test_objectives.py

# Place objectives.py in the same folder as this script.
# No src/ package needed — objectives.py is self-contained.
# """

# # import sys, os
# # # from tkinter import Image
# # sys.path.insert(0, os.path.dirname(__file__))   # find objectives.py next to this file

# import numpy as np
# import matplotlib.pyplot as plt
# # from objectives import ImageStats, build, OBJECTIVE_NAMES

# import sys, os



# # project root (parent of both src/ and tests/)
# ROOT = os.path.join(os.path.dirname(__file__), "..")
# sys.path.insert(0, ROOT)               # so "from src.segmentation import ..." works
# sys.path.insert(0, os.path.join(ROOT, "src"))  # so "from objectives import ..." works

# from objectives import ImageStats, build, OBJECTIVE_NAMES
# from segmentation import (
#     decode_thresholds,
#     reconstruct,
#     label_map,
#     display_image,
# )
# # from segmentation import decode_thresholds

# from metrics import evaluate

# class DEResult:
#     def __init__(self, best_x, best_f, history, fes_used, algo_name="DE"):
#         self.best_x    = best_x
#         self.best_f    = best_f
#         self.history   = history
#         self.fes_used  = fes_used
#         self.algo_name = algo_name

#     @property
#     def thresholds(self):
#         return decode_thresholds(self.best_x)

#     # @property
#     # def thresholds(self):
#     #     return np.sort(np.clip(np.round(self.best_x), 1, 254)).astype(int)


# class BaseDE:
#     name = "BaseDE"
#     def __init__(self, objective, K, bounds=(1, 254), pop_size=30, max_fes=10000, seed=None):
#         self.objective = objective
#         self.K = K;  self.lo, self.hi = bounds
#         self.pop_size = pop_size;  self.max_fes = max_fes;  self.seed = seed
#         self.pop = None;  self.fitness = None;  self.fes = 0;  self.history = []

#     def _clip(self, x):  return np.clip(x, self.lo, self.hi)
#     def _eval(self, x):
#         self.fes += 1
#         result = self.objective(np.atleast_2d(np.clip(np.round(x), 1, 254).astype(np.int64)))
#         return float(result[0] if hasattr(result, '__len__') else result)

#     def _init_population(self, rng):
#         self.pop     = rng.uniform(self.lo, self.hi, size=(self.pop_size, self.K))
#         self.fitness = np.array([self._eval(ind) for ind in self.pop])

#     def run(self):
#         rng = np.random.default_rng(self.seed)
#         self._init_population(rng)
#         self.history = [self.fitness.max()]
#         while self.fes < self.max_fes:
#             self._generation(rng)
#             self.history.append(self.fitness.max())
#         best_idx = int(np.argmax(self.fitness))
#         return DEResult(self.pop[best_idx].copy(), float(self.fitness[best_idx]),
#                         self.history, self.fes, self.name)

#     def _generation(self, rng): raise NotImplementedError


# class SHADE(BaseDE):
#     name = "SHADE"
#     def __init__(self, *args, H=10, use_archive=True, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.H = H;  self.use_archive = use_archive
#         self.M_F = np.full(H, 0.5);  self.M_CR = np.full(H, 0.5)
#         self.mem_idx = 0;  self.archive = []

#     def _sample_F(self, rng, r):
#         while True:
#             f = rng.standard_cauchy() * 0.1 + self.M_F[r]
#             if f > 0: return min(f, 1.0)

#     def _sample_CR(self, rng, r):
#         return float(np.clip(rng.normal(self.M_CR[r], 0.1), 0, 1))

#     def _generation(self, rng):
#         NP    = self.pop_size
#         order = np.argsort(-self.fitness)
#         S_F, S_CR, weights = [], [], []
#         for i in range(NP):
#             if self.fes >= self.max_fes: break
#             r    = rng.integers(self.H)
#             F_i  = self._sample_F(rng, r);  CR_i = self._sample_CR(rng, r)
#             p    = rng.uniform(min(2/NP,0.2), max(2/NP,0.2))
#             n_pb = max(1, int(p * NP))
#             x_pb = self.pop[order[rng.integers(n_pb)]]
#             idxs = [j for j in range(NP) if j != i]
#             r1   = rng.choice(idxs)
#             if self.use_archive and self.archive:
#                 combined = self.pop.tolist() + self.archive
#                 r2_vec   = combined[rng.integers(len(combined))]
#             else:
#                 r2 = rng.choice([j for j in idxs if j != r1])
#                 r2_vec = self.pop[r2]
#             mutant = self._clip(self.pop[i] + F_i*(x_pb - self.pop[i]) + F_i*(self.pop[r1] - r2_vec))
#             mask   = rng.random(self.K) < CR_i
#             if not mask.any(): mask[rng.integers(self.K)] = True
#             trial  = np.where(mask, mutant, self.pop[i])
#             f_t    = self._eval(trial)
#             if f_t >= self.fitness[i]:
#                 if self.use_archive:
#                     self.archive.append(self.pop[i].copy())
#                     if len(self.archive) > NP:
#                         self.archive.pop(rng.integers(len(self.archive)))
#                 delta = f_t - self.fitness[i]
#                 self.pop[i] = trial;  self.fitness[i] = f_t
#                 S_F.append(F_i);  S_CR.append(CR_i);  weights.append(delta)
#         if S_F:
#             w   = np.array(weights); w = w/w.sum() if w.sum()>0 else np.ones_like(w)/len(w)
#             sF  = np.array(S_F);  sCR = np.array(S_CR)
#             mF  = np.sum(w*sF**2) / np.sum(w*sF)
#             mCR = np.sum(w*sCR**2)/np.sum(w*sCR) if np.sum(w*sCR)>0 else np.mean(sCR)
#             self.M_F[self.mem_idx] = mF;  self.M_CR[self.mem_idx] = mCR
#             self.mem_idx = (self.mem_idx + 1) % self.H


# class LSHADE(SHADE):
#     name = "L-SHADE"
#     def __init__(self, *args, N_init=None, N_min=4, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.N_init = N_init if N_init is not None else self.pop_size
#         self.N_min  = N_min

#     def _shrink_population(self):
#         ratio = self.fes / self.max_fes
#         N_new = max(self.N_min, int(round(self.N_init + (self.N_min - self.N_init) * ratio)))
#         if N_new < len(self.pop):
#             n_drop = len(self.pop) - N_new
#             worst  = np.argsort(self.fitness)[:n_drop]
#             keep   = np.ones(len(self.pop), dtype=bool);  keep[worst] = False
#             self.pop = self.pop[keep];  self.fitness = self.fitness[keep]
#         self.pop_size = len(self.pop)
#         while len(self.archive) > self.pop_size: self.archive.pop(0)

#     def _generation(self, rng):
#         super()._generation(rng)
#         self._shrink_population()


# # ═══════════════════════════════════════════════════════════════
# #  TEST DATASET (10 IMAGES)
# # ═══════════════════════════════════════════════════════════════

# from pathlib import Path
# from PIL import Image

# # Set this to the folder containing your 10 test images.
# # Supported formats: PNG, JPG/JPEG, BMP, TIFF.
# TEST_DATASET_DIR = Path(r"C:\Users\MolokoManthata\Documents\dev_2\Image-Processing-DE\data\BDS500")
# IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# NUM_TEST_IMAGES = 10

# image_paths = sorted(
#     p for p in TEST_DATASET_DIR.iterdir()
#     if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
# )[:NUM_TEST_IMAGES] if TEST_DATASET_DIR.exists() else []

# if len(image_paths) < NUM_TEST_IMAGES:
#     raise ValueError(
#         f"Expected at least {NUM_TEST_IMAGES} images in {TEST_DATASET_DIR}, "
#         f"but found {len(image_paths)}. Update TEST_DATASET_DIR to your dataset folder."
#     )

# # Store each image's results separately so images do not share statistics.
# all_results = {}

# # ═══════════════════════════════════════════════════════════════
# #  RUN SETTINGS
# # ═══════════════════════════════════════════════════════════════

# K_VALUES   = [3, 5, 7, 9, 11, 12]
# ALGOS      = ["SHADE", "L-SHADE"]
# OBJECTIVES = list(OBJECTIVE_NAMES)   # otsu, kapur, tsallis
# MAX_FES    = 8000
# SEED       = 42

# for image_index, image_path in enumerate(image_paths, start=1):
#     img_array = np.array(Image.open(image_path).convert("L"))
#     stats = ImageStats.from_image(img_array)

#     print("\\n" + "=" * 88)
#     print(f"IMAGE {image_index}/{NUM_TEST_IMAGES}: {image_path.name}")
#     print(f"  Shape: {img_array.shape} | range [{img_array.min()}, {img_array.max()}]")
#     print("=" * 88)

#     results = {obj: {algo: {} for algo in ALGOS} for obj in OBJECTIVES}
#     print(f"  {'Obj':8} {'Algo':8} {'K':>4}  {'Fitness':>12}  Thresholds")
#     print("-" * 72)

#     for obj_name in OBJECTIVES:
#         objective_fn = build(obj_name, stats)

#         for K in K_VALUES:
#             N_init = max(30, 18 * K)

#             shade = SHADE(
#                 objective_fn, K=K, bounds=(1, 254), pop_size=30,
#                 max_fes=MAX_FES, seed=SEED + image_index
#             )
#             r_s = shade.run()
#             results[obj_name]["SHADE"][K] = r_s

#             lshade = LSHADE(
#                 objective_fn, K=K, bounds=(1, 254),
#                 pop_size=N_init, N_init=N_init, N_min=4,
#                 max_fes=MAX_FES, seed=SEED + image_index
#             )
#             r_l = lshade.run()
#             results[obj_name]["L-SHADE"][K] = r_l

#             # Decode the optimized thresholds
#             shade_thresholds = decode_thresholds(r_s.best_x)
#             lshade_thresholds = decode_thresholds(r_l.best_x)

#             # Reconstruct each segmented image using class means
#             shade_segmented = reconstruct(img_array, shade_thresholds)
#             lshade_segmented = reconstruct(img_array, lshade_thresholds)

#             # Create class-label maps
#             shade_labels = label_map(img_array, shade_thresholds)
#             lshade_labels = label_map(img_array, lshade_thresholds)

#             # Create display-friendly segmentation images
#             shade_display = display_image(img_array, shade_thresholds)
#             lshade_display = display_image(img_array, lshade_thresholds)

#             for algo, result in [("SHADE", r_s), ("L-SHADE", r_l)]:
#                 thresholds = str(list(map(int, result.thresholds)))
#                 print(
#                     f"  {obj_name:8} {algo:8} {K:>4}  "
#                     f"{result.best_f:>12.6f}  {thresholds}"
#                 )

#     all_results[image_path.name] = results

#     # Save convergence plots for this image and objective.
#     colors = plt.cm.tab10.colors
#     safe_stem = image_path.stem.replace(" ", "_")

#     for obj_name in OBJECTIVES:
#         fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
#         fig.suptitle(
#             f"{image_path.name} — Convergence ({obj_name.upper()} objective)",
#             fontsize=12
#         )

#         for ax, algo in zip(axes, ALGOS):
#             for i, K in enumerate(K_VALUES):
#                 history = results[obj_name][algo][K].history
#                 ax.plot(history, color=colors[i], lw=1.5, label=f"K={K}")
#             ax.set_title(algo)
#             ax.set_xlabel("Generation")
#             ax.set_ylabel("Best fitness")
#             ax.legend(ncol=3, fontsize=8)
#             ax.grid(alpha=0.3)

#         plt.tight_layout()
#         fname = f"convergence_{safe_stem}_{obj_name}.png"
#         plt.savefig(fname, dpi=120)
#         plt.close()
#         print(f"  Saved -> {fname}")

#     # Save a fitness comparison chart for this image.
#     fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(6 * len(OBJECTIVES), 5))
#     if len(OBJECTIVES) == 1:
#         axes = [axes]

#     for ax, obj_name in zip(axes, OBJECTIVES):
#         x = np.arange(len(K_VALUES))
#         width = 0.35
#         for j, algo in enumerate(ALGOS):
#             values = [results[obj_name][algo][K].best_f for K in K_VALUES]
#             ax.bar(
#                 x + j * width, values, width, label=algo,
#                 color=["steelblue", "tomato"][j], alpha=0.85
#             )
#         ax.set_title(obj_name.upper())
#         ax.set_xticks(x + width / 2)
#         ax.set_xticklabels([f"K={k}" for k in K_VALUES], rotation=30)
#         ax.set_ylabel("Best fitness")
#         ax.legend()
#         ax.grid(axis="y", alpha=0.3)

#     plt.suptitle(f"{image_path.name} — SHADE vs L-SHADE Best Fitness", fontsize=12)
#     plt.tight_layout()
#     fitness_fname = f"fitness_comparison_{safe_stem}.png"
#     plt.savefig(fitness_fname, dpi=120)
#     plt.close()
#     print(f"  Saved -> {fitness_fname}")

# shade_thresholds = decode_thresholds(r_s.best_x)
# lshade_thresholds = decode_thresholds(r_l.best_x)

# shade_metrics = evaluate(img_array, shade_thresholds, stats)
# lshade_metrics = evaluate(img_array, lshade_thresholds, stats)

# print("SHADE metrics:", shade_metrics)
# print("L-SHADE metrics:", lshade_metrics)

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(img_array, cmap="gray", vmin=0, vmax=255)
# axes[0].set_title("Original image")

# axes[1].imshow(shade_display, cmap="gray", vmin=0, vmax=255)
# axes[1].set_title("SHADE segmentation")

# axes[2].imshow(lshade_display, cmap="gray", vmin=0, vmax=255)
# axes[2].set_title("L-SHADE segmentation")

# for ax in axes:
#     ax.axis("off")

# plt.tight_layout()
# plt.savefig("segmentation_comparison.png", dpi=150)
# plt.close()

# print("\\nCompleted test run for", len(all_results), "images.")

"""
Master script: reproduces all experiments, statistics, tables and plots.

Protocol
--------
* 30 independent runs per algorithm for every (Image, Objective, K)
* Equal budget: every algorithm stops at the same MAX_FES
* Wilcoxon signed-rank (alpha = 0.05, paired by run) + Friedman rank tests
* Outputs: PSNR / SSIM / Uniformity mu +- sigma tables, execution-time table,
  convergence curves, segmentation figures, statistical-test CSVs

Layout: de_algorithms.py, objectives.py, segmentation.py, metrics.py in src/;
this script in tests/ (or the project root).

Usage:

    python run_experiments.py                    # full protocol
    python run_experiments.py --quick            # smoke test (2 imgs, 3 runs)
    python run_experiments.py --analyze-only     # re-do tables/plots from saved raw data
    python run_experiments.py --workers 8

Raw results are cached per (image, objective, K) in <out>/raw/*.pkl, so an
interrupted run resumes where it stopped.
"""
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
 
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)   # project root, so `src.*` imports resolve (as in metrics.py)
 
from src.objectives import ImageStats, build, OBJECTIVE_NAMES      # noqa: E402
from src.segmentation import (                                     # noqa: E402
    decode_thresholds, display_image, MIN_THRESHOLD, MAX_THRESHOLD)
from src.metrics import evaluate                                   # noqa: E402
from src.algorithms.shade import SHADE                            # noqa: E402
from src.algorithms.lshade import LSHADE                           # noqa: E402
 
# String-keyed registry: names are stored in the results, classes are looked up here.
ALGORITHMS = {"SHADE": SHADE, "L-SHADE": LSHADE}                           # noqa: E402
 
# ═══════════════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_DATA_DIR = r"C:\Users\MolokoManthata\Documents\dev_2\Image-Processing-DE\data\BDS500"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
K_VALUES = [3, 5, 7, 9, 11, 12]
ALGO_NAMES = list(ALGORITHMS)                 # SHADE, L-SHADE
OBJECTIVES = list(OBJECTIVE_NAMES)            # otsu, kapur, tsallis
METRICS = ["psnr", "ssim", "uniformity"]
STAT_METRICS = ["fitness"] + METRICS
ALPHA = 0.05
BASE_SEED = 2024
CONV_GRID_N = 201
SEG_K_SHOWN = [3, 7, 12]
REFERENCE_ALGO = "L-SHADE"
 
METRIC_ALIASES = {
    "psnr": ["psnr"],
    "ssim": ["ssim"],
    "uniformity": ["uniformity", "unif", "uni"],
}
 
 
def get_metric(result, name):
    """Pull a metric out of whatever metrics.evaluate() returns (dict expected)."""
    if not isinstance(result, dict):
        raise TypeError(
            f"metrics.evaluate() returned {type(result)}; expected a dict with "
            "keys psnr / ssim / uniformity. Adapt get_metric() if it differs.")
    lowered = {str(k).lower(): v for k, v in result.items()}
    for alias in METRIC_ALIASES[name]:
        if alias in lowered:
            return float(lowered[alias])
    raise KeyError(f"'{name}' not found in evaluate() output keys {list(result)}")
 
 
def make_algo(name, objective, K, cfg, seed):
    common = dict(bounds=(MIN_THRESHOLD, MAX_THRESHOLD), max_fes=cfg.max_fes, seed=seed)
    if name == "L-SHADE":
        n0 = max(cfg.pop_size, 18 * K)
        return ALGORITHMS[name](objective, K, pop_size=n0, N_init=n0, N_min=4, **common)
    return ALGORITHMS[name](objective, K, pop_size=cfg.pop_size, **common)
 
 
def list_images(data_dir, n):
    paths = sorted(p for p in Path(data_dir).rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)[:n]
    if len(paths) < n:
        raise ValueError(f"Need {n} images in {data_dir}, found {len(paths)}.")
    return paths
 
 
# ═══════════════════════════════════════════════════════════════════════
#  STAGE 1 - EXPERIMENTS (one work unit = image x objective x K)
# ═══════════════════════════════════════════════════════════════════════
def run_unit(task):
    image_path, obj_name, K, cfg = task
    raw_dir = Path(cfg.out) / "raw"
    out_file = raw_dir / f"{Path(image_path).stem}__{obj_name}__K{K}.pkl"
    if out_file.exists():
        return str(out_file)
 
    img = np.array(Image.open(image_path).convert("L"))
    stats = ImageStats.from_image(img)
    objective = build(obj_name, stats)
    grid = np.linspace(0, cfg.max_fes, CONV_GRID_N)
 
    records = []
    for algo_name in ALGO_NAMES:
        for run in range(cfg.runs):
            seed = BASE_SEED + run          # same seed index across algorithms
            algo = make_algo(algo_name, objective, K, cfg, seed)
            t0 = time.perf_counter()
            res = algo.run()
            elapsed = time.perf_counter() - t0
 
            thr = decode_thresholds(res.best_x)
            m = evaluate(img, thr, stats)
            idx = np.clip(np.searchsorted(res.fes_hist, grid, side="right") - 1, 0, None)
 
            records.append(dict(
                image=Path(image_path).name, objective=obj_name, K=K,
                algorithm=algo_name, run=run, fitness=res.best_f,
                time=elapsed, fes=res.fes_used,
                psnr=get_metric(m, "psnr"), ssim=get_metric(m, "ssim"),
                uniformity=get_metric(m, "uniformity"),
                thresholds=[int(t) for t in thr],
                conv=res.best_hist[idx]))
 
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(records, f)
    return str(out_file)
 
 
def run_experiments(cfg):
    paths = list_images(cfg.data_dir, cfg.n_images)
    tasks = [(str(p), o, K, cfg) for p in paths for o in OBJECTIVES for K in K_VALUES]
    print(f"{len(tasks)} work units ({len(paths)} images x {len(OBJECTIVES)} objectives "
          f"x {len(K_VALUES)} K), {cfg.runs} runs x {len(ALGO_NAMES)} algorithms each, "
          f"max_fes={cfg.max_fes}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futs = [ex.submit(run_unit, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            print(f"  [{i}/{len(tasks)}] done  ({time.time() - t0:.0f}s elapsed)", flush=True)
 
 
# ═══════════════════════════════════════════════════════════════════════
#  STAGE 2 - ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def load_results(out):
    records = []
    for f in sorted((Path(out) / "raw").glob("*.pkl")):
        with open(f, "rb") as fh:
            records.extend(pickle.load(fh))
    if not records:
        raise RuntimeError("No raw results found - run the experiments first.")
    df = pd.DataFrame(records)
 
    def _algo_label(v):
        if isinstance(v, str):
            return v
        return getattr(v, "name", getattr(v, "__name__", str(v)))
 
    df["algorithm"] = df["algorithm"].map(_algo_label)
    # psnr() returns inf for an exact reconstruction; keep means/stds finite
    # (pandas skips NaN). Check df.psnr.isna().sum() if this ever triggers.
    num_cols = ["fitness", "time", "psnr", "ssim", "uniformity"]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df
 
 
def to_markdown(df):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)
 
 
def summary_tables(df, tdir):
    """Mean +- std over runs (per-run value = mean over images), as in Table 1."""
    decimals = {"psnr": 2, "ssim": 4, "uniformity": 4}
    for metric in METRICS:
        per_run = df.groupby(["objective", "K", "algorithm", "run"])[metric].mean().reset_index()
        agg = per_run.groupby(["K", "algorithm", "objective"])[metric].agg(["mean", "std"]).reset_index()
        agg.to_csv(tdir / f"summary_{metric}_long.csv", index=False)
        d = decimals[metric]
        md = []
        for K in K_VALUES:
            sub = agg[agg.K == K]
            tab = pd.DataFrame(index=pd.Index(ALGO_NAMES, name="Algorithm"), columns=OBJECTIVES)
            for _, r in sub.iterrows():
                tab.loc[r.algorithm, r.objective] = f"{r['mean']:.{d}f} ± {r['std']:.{d}f}"
            md.append(f"### {metric.upper()} (μ ± σ over {df.run.nunique()} runs), K = {K}\n\n"
                      + to_markdown(tab) + "\n")
        (tdir / f"table_{metric}.md").write_text("\n".join(md), encoding="utf-8")
 
    # per-image detail
    (df.groupby(["image", "objective", "K", "algorithm"])[STAT_METRICS + ["time"]]
       .agg(["mean", "std"]).to_csv(tdir / "summary_per_image.csv"))
 
    # execution time / scalability
    t = df.groupby(["K", "algorithm"])["time"].agg(["mean", "std"]).reset_index()
    t.to_csv(tdir / "time_long.csv", index=False)
    tab = pd.DataFrame(index=pd.Index(K_VALUES, name="K"), columns=ALGO_NAMES)
    for _, r in t.iterrows():
        tab.loc[r.K, r.algorithm] = f"{r['mean']:.2f} ± {r['std']:.2f}"
    (tdir / "table_time_seconds.md").write_text(to_markdown(tab), encoding="utf-8")
 
 
def paired_wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0
 
 
def statistical_tests(df, tdir):
    # ---- Wilcoxon signed-rank: REFERENCE_ALGO vs each other, paired by run
    rows = []
    for metric in STAT_METRICS:
        for (img, obj, K), g in df.groupby(["image", "objective", "K"]):
            wide = g.pivot(index="run", columns="algorithm", values=metric).sort_index()
            for other in ALGO_NAMES:
                if other == REFERENCE_ALGO:
                    continue
                p = paired_wilcoxon(wide[REFERENCE_ALGO], wide[other])
                diff = float(wide[REFERENCE_ALGO].mean() - wide[other].mean())
                outcome = "tie" if p >= ALPHA else ("win" if diff > 0 else "loss")
                rows.append(dict(metric=metric, image=img, objective=obj, K=K,
                                 reference=REFERENCE_ALGO, other=other,
                                 mean_diff=diff, p_value=p, outcome=outcome))
    w = pd.DataFrame(rows)
    w.to_csv(tdir / "wilcoxon_all.csv", index=False)
    summ = (w.groupby(["metric", "objective", "other", "outcome"]).size()
              .unstack("outcome", fill_value=0).reset_index())
    for c in ("win", "tie", "loss"):
        if c not in summ:
            summ[c] = 0
    summ.to_csv(tdir / "wilcoxon_summary_win_tie_loss.csv", index=False)
 
    # ---- Friedman: blocks = (image, objective, K); mean over runs
    rows = []
    if len(ALGO_NAMES) < 3:
        print("Skipping Friedman test (needs >= 3 algorithms)")
        return
    for metric in STAT_METRICS:
        blocks = (df.groupby(["image", "objective", "K", "algorithm"])[metric]
                    .mean().unstack("algorithm")[ALGO_NAMES])
        scopes = [("all", blocks)] + [(o, blocks.xs(o, level="objective")) for o in OBJECTIVES]
        for scope, B in scopes:
            if len(B) < 2:
                continue
            chi2, p = friedmanchisquare(*[B[a].values for a in ALGO_NAMES])
            ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, B.values).mean(axis=0)
            rows.append(dict(metric=metric, scope=scope, n_blocks=len(B),
                             chi2=float(chi2), p_value=float(p),
                             **{f"rank_{a}": r for a, r in zip(ALGO_NAMES, ranks)}))
    pd.DataFrame(rows).to_csv(tdir / "friedman.csv", index=False)
 
 
# ---------------------------------------------------------------------------
def plot_convergence(df, pdir, max_fes):
    grid = np.linspace(0, max_fes, CONV_GRID_N)
    colors = dict(zip(ALGO_NAMES, plt.cm.tab10.colors))
    cdir = pdir / "convergence"
    cdir.mkdir(exist_ok=True)
 
    mean_curves = (df.groupby(["image", "objective", "K", "algorithm"])["conv"]
                     .apply(lambda s: np.mean(np.stack(s.values), axis=0)))
    for (img, obj), _ in df.groupby(["image", "objective"]):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, K in zip(axes.ravel(), K_VALUES):
            for a in ALGO_NAMES:
                ax.plot(grid, mean_curves[(img, obj, K, a)], color=colors[a], lw=1.5, label=a)
            ax.set_title(f"K = {K}")
            ax.set_xlabel("Function evaluations")
            ax.set_ylabel("Mean best fitness")
            ax.grid(alpha=0.3)
        axes[0, 0].legend()
        fig.suptitle(f"{img} - convergence ({obj.upper()}), mean of {df.run.nunique()} runs")
        fig.tight_layout()
        fig.savefig(cdir / f"convergence_{Path(img).stem}_{obj}.png", dpi=120)
        plt.close(fig)
 
 
def plot_time(df, pdir):
    t = df.groupby(["K", "algorithm"])["time"].mean().unstack()[ALGO_NAMES]
    ax = t.plot(marker="o", figsize=(7, 4.5))
    ax.set_xlabel("K (number of thresholds)")
    ax.set_ylabel("Mean runtime per run (s)")
    ax.set_title("Execution time vs. K")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pdir / "time_vs_K.png", dpi=130)
    plt.close()
 
 
def plot_segmentations(df, pdir, path_by_name):
    sdir = pdir / "segmentation"
    sdir.mkdir(exist_ok=True)
    best = df.loc[df.groupby(["image", "objective", "K", "algorithm"])["fitness"].idxmax()]
    for (img, obj), g in best.groupby(["image", "objective"]):
        arr = np.array(Image.open(path_by_name[img]).convert("L"))
        nrow, ncol = len(SEG_K_SHOWN), 1 + len(ALGO_NAMES)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3 * nrow))
        for r, K in enumerate(SEG_K_SHOWN):
            axes[r, 0].imshow(arr, cmap="gray", vmin=0, vmax=255)
            axes[r, 0].set_title("Original" if r == 0 else "")
            axes[r, 0].set_ylabel(f"K = {K}")
            for c, a in enumerate(ALGO_NAMES, start=1):
                row = g[(g.K == K) & (g.algorithm == a)].iloc[0]
                seg = display_image(arr, np.array(row.thresholds))
                axes[r, c].imshow(seg, cmap="gray", vmin=0, vmax=255)
                axes[r, c].set_title(f"{a}\nPSNR {row.psnr:.2f}" if r == 0
                                     else f"PSNR {row.psnr:.2f}", fontsize=9)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{img} - {obj.upper()} (best run per algorithm)")
        fig.tight_layout()
        fig.savefig(sdir / f"segmentation_{Path(img).stem}_{obj}.png", dpi=120)
        plt.close(fig)
 
 
def analyze(cfg):
    out = Path(cfg.out)
    tdir, pdir = out / "tables", out / "plots"
    tdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)
 
    df = load_results(out)
    global ALGO_NAMES
    ALGO_NAMES = [a for a in ALGORITHMS if a in set(df.algorithm)]
    print("Algorithms analysed:", ALGO_NAMES)
    # Equal-budget sanity check
    print("FEs used (min/max per algorithm):")
    print(df.groupby("algorithm")["fes"].agg(["min", "max"]))
 
    df.drop(columns=["conv", "thresholds"]).to_csv(out / "all_runs.csv", index=False)
    summary_tables(df, tdir)
    statistical_tests(df, tdir)
    plot_time(df, pdir)
    plot_convergence(df, pdir, cfg.max_fes)
    path_by_name = {p.name: p for p in list_images(cfg.data_dir, df.image.nunique())}
    plot_segmentations(df, pdir, path_by_name)
    print(f"\nTables  -> {tdir}\nPlots   -> {pdir}\nRaw CSV -> {out / 'all_runs.csv'}")
 
 
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--max-fes", type=int, default=8000)
    ap.add_argument("--pop-size", type=int, default=30,
                    help="population for DE/JADE/SHADE (L-SHADE starts at max(this, 18K))")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true", help="2 images, 3 runs, 2000 FEs")
    ap.add_argument("--analyze-only", action="store_true")
    cfg = ap.parse_args()
    if cfg.quick:
        cfg.n_images, cfg.runs, cfg.max_fes = 2, 3, 2000
        cfg.out = cfg.out + "_quick"
 
    if not cfg.analyze_only:
        run_experiments(cfg)
    analyze(cfg)
 
 
if __name__ == "__main__":
    main()
 