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
