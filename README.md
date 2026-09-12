# Multilevel Image Thresholding with Differential Evolution

COS791 Image Analysis and Understanding — implementation, analysis and empirical
evaluation of five Differential Evolution variants for multilevel image
thresholding, using Otsu's variance, Kapur's entropy and Tsallis entropy as
objective functions.

## Status

Scaffolded. The optimisation core is written; the experiment driver, statistics,
plotting and master script are stubs.

| Module | State |
|---|---|
| `src/objectives.py` | implemented — Otsu, Kapur, Tsallis |
| `src/algorithms/` | implemented — DE, JADE, SHADE, L-SHADE, LADE |
| `src/metrics.py` | implemented — PSNR, SSIM, Uniformity, Jaccard, Dice |
| `src/segmentation.py` | implemented — reconstruction, GT mask matching |
| `src/datasets.py` | implemented — BSD500 and CHAOS loaders |
| `src/experiment.py` | stub |
| `src/stats.py` | stub |
| `src/plots.py` | stub |
| `experiments/*.py` | stubs |
| `run_all.py` | stub |
| `tests/` | stubs |

## Layout

```
config.yaml            all experiment parameters; nothing is hard-coded
run_all.py             master script — reproduces every result
data/BDS500/           10 benchmark images (+ _gt companions)
data/CHAOS/            15 MRI slices
src/objectives.py      Otsu / Kapur / Tsallis, histogram-based and vectorised
src/algorithms/        the five DE variants over a shared base
src/metrics.py         PSNR, SSIM, Uniformity, Jaccard, Dice
src/segmentation.py    thresholds -> image; GT mask matching
src/experiment.py      the sweep driver
src/stats.py           Wilcoxon / Friedman, LaTeX tables
src/plots.py           convergence, scalability, boxplots, visual panels
experiments/           one module per experiment
results/raw/           per-run records (gitignored; regenerate with run_all.py)
results/tables/        LaTeX tables for the report (tracked)
results/figures/       figures for the report (tracked)
report/                ESWA LaTeX source
```

## Setup

```bash
python3 -m venv ~/.venvs/cos791-de
~/.venvs/cos791-de/bin/pip install -r requirements.txt
```

Run from the repo root so `src` is importable:

```bash
~/.venvs/cos791-de/bin/python run_all.py --smoke
```

## Design notes

**Objectives are histogram-based.** All three read only the 256-bin intensity
histogram via prefix sums, so one fitness evaluation is O(K) table lookups
rather than O(M×N). This is what makes the full sweep tractable.

**The FE budget lives in `Problem`, not in the solvers.** The assignment requires
equal function evaluations across algorithms; putting the counter in the shared
problem wrapper makes it structurally impossible for one variant to take more.

**Encoding.** Individuals are continuous vectors in [0, 254], rounded and sorted
at evaluation time. Ordering is enforced inside the objective rather than by
repairing candidates, so mutation is unconstrained.

## Scale of the sweep

25 images × 3 objectives × 6 K values × 5 algorithms × 30 runs = **67,500 runs**.
The driver is designed to checkpoint per configuration into `results/raw/` and
skip completed tasks, so the sweep is resumable.

## Open issues

1. **CHAOS has no ground-truth masks.** Experiment 2 (15 marks) requires a GT
   mask matching procedure and Jaccard/Dice, but `CHAOS_DATA.zip` contains only
   the 15 raw slices. The matching procedure is implemented and the loader looks
   for `data/CHAOS/masks/<same-filename>.png`; the masks themselves must be
   obtained from the CHAOS challenge dataset, or the requirement clarified with
   the lecturer.

2. **The brief contradicts itself on algorithm count.** §1.3 says "six DE
   variants" then lists five; the mark allocation also says five (DE, JADE,
   SHADE, L-SHADE, LADE) while the example table shows CoDE and SaDE. The five
   from the mark allocation are implemented. Worth confirming.

3. **Uniformity is biased upward in K.** More regions are necessarily more
   homogeneous, so U compares algorithms at a fixed K but does not justify
   choosing a larger K. Should be stated in the report.
