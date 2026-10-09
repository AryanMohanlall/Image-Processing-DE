"""
Test SHADE & L-SHADE with Otsu, Kapur, and Tsallis objectives.

Usage:
    python test_objectives.py

Place objectives.py in the same folder as this script.
No src/ package needed — objectives.py is self-contained.
"""

# import sys, os
# # from tkinter import Image
# sys.path.insert(0, os.path.dirname(__file__))   # find objectives.py next to this file

import numpy as np
import matplotlib.pyplot as plt
# from objectives import ImageStats, build, OBJECTIVE_NAMES

import sys, os

# project root (parent of both src/ and tests/)
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)               # so "from src.segmentation import ..." works
sys.path.insert(0, os.path.join(ROOT, "src"))  # so "from objectives import ..." works

from objectives import ImageStats, build, OBJECTIVE_NAMES

class DEResult:
    def __init__(self, best_x, best_f, history, fes_used, algo_name="DE"):
        self.best_x    = best_x
        self.best_f    = best_f
        self.history   = history
        self.fes_used  = fes_used
        self.algo_name = algo_name

    @property
    def thresholds(self):
        return np.sort(np.clip(np.round(self.best_x), 1, 254)).astype(int)


class BaseDE:
    name = "BaseDE"
    def __init__(self, objective, K, bounds=(1, 254), pop_size=30, max_fes=10000, seed=None):
        self.objective = objective
        self.K = K;  self.lo, self.hi = bounds
        self.pop_size = pop_size;  self.max_fes = max_fes;  self.seed = seed
        self.pop = None;  self.fitness = None;  self.fes = 0;  self.history = []

    def _clip(self, x):  return np.clip(x, self.lo, self.hi)
    def _eval(self, x):
        self.fes += 1
        result = self.objective(np.atleast_2d(np.clip(np.round(x), 1, 254).astype(np.int64)))
        return float(result[0] if hasattr(result, '__len__') else result)

    def _init_population(self, rng):
        self.pop     = rng.uniform(self.lo, self.hi, size=(self.pop_size, self.K))
        self.fitness = np.array([self._eval(ind) for ind in self.pop])

    def run(self):
        rng = np.random.default_rng(self.seed)
        self._init_population(rng)
        self.history = [self.fitness.max()]
        while self.fes < self.max_fes:
            self._generation(rng)
            self.history.append(self.fitness.max())
        best_idx = int(np.argmax(self.fitness))
        return DEResult(self.pop[best_idx].copy(), float(self.fitness[best_idx]),
                        self.history, self.fes, self.name)

    def _generation(self, rng): raise NotImplementedError


class SHADE(BaseDE):
    name = "SHADE"
    def __init__(self, *args, H=10, use_archive=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.H = H;  self.use_archive = use_archive
        self.M_F = np.full(H, 0.5);  self.M_CR = np.full(H, 0.5)
        self.mem_idx = 0;  self.archive = []

    def _sample_F(self, rng, r):
        while True:
            f = rng.standard_cauchy() * 0.1 + self.M_F[r]
            if f > 0: return min(f, 1.0)

    def _sample_CR(self, rng, r):
        return float(np.clip(rng.normal(self.M_CR[r], 0.1), 0, 1))

    def _generation(self, rng):
        NP    = self.pop_size
        order = np.argsort(-self.fitness)
        S_F, S_CR, weights = [], [], []
        for i in range(NP):
            if self.fes >= self.max_fes: break
            r    = rng.integers(self.H)
            F_i  = self._sample_F(rng, r);  CR_i = self._sample_CR(rng, r)
            p    = rng.uniform(min(2/NP,0.2), max(2/NP,0.2))
            n_pb = max(1, int(p * NP))
            x_pb = self.pop[order[rng.integers(n_pb)]]
            idxs = [j for j in range(NP) if j != i]
            r1   = rng.choice(idxs)
            if self.use_archive and self.archive:
                combined = self.pop.tolist() + self.archive
                r2_vec   = combined[rng.integers(len(combined))]
            else:
                r2 = rng.choice([j for j in idxs if j != r1])
                r2_vec = self.pop[r2]
            mutant = self._clip(self.pop[i] + F_i*(x_pb - self.pop[i]) + F_i*(self.pop[r1] - r2_vec))
            mask   = rng.random(self.K) < CR_i
            if not mask.any(): mask[rng.integers(self.K)] = True
            trial  = np.where(mask, mutant, self.pop[i])
            f_t    = self._eval(trial)
            if f_t >= self.fitness[i]:
                if self.use_archive:
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > NP:
                        self.archive.pop(rng.integers(len(self.archive)))
                delta = f_t - self.fitness[i]
                self.pop[i] = trial;  self.fitness[i] = f_t
                S_F.append(F_i);  S_CR.append(CR_i);  weights.append(delta)
        if S_F:
            w   = np.array(weights); w = w/w.sum() if w.sum()>0 else np.ones_like(w)/len(w)
            sF  = np.array(S_F);  sCR = np.array(S_CR)
            mF  = np.sum(w*sF**2) / np.sum(w*sF)
            mCR = np.sum(w*sCR**2)/np.sum(w*sCR) if np.sum(w*sCR)>0 else np.mean(sCR)
            self.M_F[self.mem_idx] = mF;  self.M_CR[self.mem_idx] = mCR
            self.mem_idx = (self.mem_idx + 1) % self.H


class LSHADE(SHADE):
    name = "L-SHADE"
    def __init__(self, *args, N_init=None, N_min=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_init = N_init if N_init is not None else self.pop_size
        self.N_min  = N_min

    def _shrink_population(self):
        ratio = self.fes / self.max_fes
        N_new = max(self.N_min, int(round(self.N_init + (self.N_min - self.N_init) * ratio)))
        if N_new < len(self.pop):
            n_drop = len(self.pop) - N_new
            worst  = np.argsort(self.fitness)[:n_drop]
            keep   = np.ones(len(self.pop), dtype=bool);  keep[worst] = False
            self.pop = self.pop[keep];  self.fitness = self.fitness[keep]
        self.pop_size = len(self.pop)
        while len(self.archive) > self.pop_size: self.archive.pop(0)

    def _generation(self, rng):
        super()._generation(rng)
        self._shrink_population()


# ═══════════════════════════════════════════════════════════════
#  TEST DATASET (10 IMAGES)
# ═══════════════════════════════════════════════════════════════

from pathlib import Path
from PIL import Image

# Set this to the folder containing your 10 test images.
# Supported formats: PNG, JPG/JPEG, BMP, TIFF.
TEST_DATASET_DIR = Path(r"C:\Users\MolokoManthata\Documents\dev_2\Image-Processing-DE\data\BDS500")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NUM_TEST_IMAGES = 10

image_paths = sorted(
    p for p in TEST_DATASET_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)[:NUM_TEST_IMAGES] if TEST_DATASET_DIR.exists() else []

if len(image_paths) < NUM_TEST_IMAGES:
    raise ValueError(
        f"Expected at least {NUM_TEST_IMAGES} images in {TEST_DATASET_DIR}, "
        f"but found {len(image_paths)}. Update TEST_DATASET_DIR to your dataset folder."
    )

# Store each image's results separately so images do not share statistics.
all_results = {}

# ═══════════════════════════════════════════════════════════════
#  RUN SETTINGS
# ═══════════════════════════════════════════════════════════════

K_VALUES   = [3, 5, 7, 9, 11, 12]
ALGOS      = ["SHADE", "L-SHADE"]
OBJECTIVES = list(OBJECTIVE_NAMES)   # otsu, kapur, tsallis
MAX_FES    = 8000
SEED       = 42

for image_index, image_path in enumerate(image_paths, start=1):
    img_array = np.array(Image.open(image_path).convert("L"))
    stats = ImageStats.from_image(img_array)

    print("\\n" + "=" * 88)
    print(f"IMAGE {image_index}/{NUM_TEST_IMAGES}: {image_path.name}")
    print(f"  Shape: {img_array.shape} | range [{img_array.min()}, {img_array.max()}]")
    print("=" * 88)

    results = {obj: {algo: {} for algo in ALGOS} for obj in OBJECTIVES}
    print(f"  {'Obj':8} {'Algo':8} {'K':>4}  {'Fitness':>12}  Thresholds")
    print("-" * 72)

    for obj_name in OBJECTIVES:
        objective_fn = build(obj_name, stats)

        for K in K_VALUES:
            N_init = max(30, 18 * K)

            shade = SHADE(
                objective_fn, K=K, bounds=(1, 254), pop_size=30,
                max_fes=MAX_FES, seed=SEED + image_index
            )
            r_s = shade.run()
            results[obj_name]["SHADE"][K] = r_s

            lshade = LSHADE(
                objective_fn, K=K, bounds=(1, 254),
                pop_size=N_init, N_init=N_init, N_min=4,
                max_fes=MAX_FES, seed=SEED + image_index
            )
            r_l = lshade.run()
            results[obj_name]["L-SHADE"][K] = r_l

            for algo, result in [("SHADE", r_s), ("L-SHADE", r_l)]:
                thresholds = str(list(map(int, result.thresholds)))
                print(
                    f"  {obj_name:8} {algo:8} {K:>4}  "
                    f"{result.best_f:>12.6f}  {thresholds}"
                )

    all_results[image_path.name] = results

    # Save convergence plots for this image and objective.
    colors = plt.cm.tab10.colors
    safe_stem = image_path.stem.replace(" ", "_")

    for obj_name in OBJECTIVES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
        fig.suptitle(
            f"{image_path.name} — Convergence ({obj_name.upper()} objective)",
            fontsize=12
        )

        for ax, algo in zip(axes, ALGOS):
            for i, K in enumerate(K_VALUES):
                history = results[obj_name][algo][K].history
                ax.plot(history, color=colors[i], lw=1.5, label=f"K={K}")
            ax.set_title(algo)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Best fitness")
            ax.legend(ncol=3, fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"convergence_{safe_stem}_{obj_name}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  Saved -> {fname}")

    # Save a fitness comparison chart for this image.
    fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(6 * len(OBJECTIVES), 5))
    if len(OBJECTIVES) == 1:
        axes = [axes]

    for ax, obj_name in zip(axes, OBJECTIVES):
        x = np.arange(len(K_VALUES))
        width = 0.35
        for j, algo in enumerate(ALGOS):
            values = [results[obj_name][algo][K].best_f for K in K_VALUES]
            ax.bar(
                x + j * width, values, width, label=algo,
                color=["steelblue", "tomato"][j], alpha=0.85
            )
        ax.set_title(obj_name.upper())
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f"K={k}" for k in K_VALUES], rotation=30)
        ax.set_ylabel("Best fitness")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"{image_path.name} — SHADE vs L-SHADE Best Fitness", fontsize=12)
    plt.tight_layout()
    fitness_fname = f"fitness_comparison_{safe_stem}.png"
    plt.savefig(fitness_fname, dpi=120)
    plt.close()
    print(f"  Saved -> {fitness_fname}")

print("\\nCompleted test run for", len(all_results), "images.")
