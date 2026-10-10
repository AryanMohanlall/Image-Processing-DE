# Multilevel Image Thresholding with Differential Evolution

COS791 group assignment. Five DE variants are compared on multilevel image
thresholding across three objective functions, six threshold levels and two
datasets, with 30 independent runs per combination.

## Setup

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

## Running

```sh
python run_all.py --smoke
python run_all.py
```

`run_all.py` is the master script: it runs both experiments, then the
statistics and figures, and writes a manifest recording the git commit and
environment that produced the results.

Work is sharded per cell and cached, so re-running skips anything already
computed and an interrupted sweep resumes where it stopped. Use `--overwrite` to
force recomputation and `--jobs N` to set the worker count.

Individual experiments:

```sh
python experiments/exp1_bsd500.py          # reconstruction metrics, BSD500
python experiments/exp2_chaos.py           # domain application, CHAOS MRI
python experiments/exp3_stats_scaling.py   # significance tests and figures
```

To regenerate SHADE and L-SHADE with the shared runner (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m experiments.exp1_bsd500 --algorithms shade lshade --runs 30 --max-evaluations 30000 --results results/exp1_shade_lshade_v2
.\.venv\Scripts\python.exe -m experiments.exp2_chaos --algorithms shade lshade --runs 30 --max-evaluations 30000 --results results/exp2_shade_lshade_v2
```

These fresh folders keep the earlier 8,000-evaluation results separate. The
runner includes all 10 BSD500 images and all 15 CHAOS slices, three objectives
(Tsallis q=0.8), and K=3,5,7,9,11,12. It saves per-run thresholds, fitness,
timing, PSNR, SSIM, uniformity, class separability, and convergence traces,
plus summary tables. Add `--smoke --jobs 1` for a short pipeline check.

The optimisation operators and adaptation rules are retained from the original
SHADE/L-SHADE implementations. Only the runner interface, evaluation accounting,
threshold decoding and saved output handling are adapted. These changes do not
validate or correct the original optimisation logic.
