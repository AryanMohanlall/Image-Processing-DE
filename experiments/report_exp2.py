"""Update Experiment 2 in temp.tex from saved CHAOS results.

Run: python -m experiments.report_exp2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"jade": ROOT / "results/exp2_jade"}
LABELS = {"standard_de": "DE", "jade": "JADE", "shade": "SHADE",
          "lshade": "L-SHADE", "late_acceptance_de": "LADE"}
COLOURS = {"standard_de": "#4C72B0", "jade": "#55A868", "shade": "#C44E52",
           "lshade": "#DD8452", "late_acceptance_de": "#8172B3"}
OBJECTIVES = ("otsu", "kapur", "tsallis")
KS = (3, 5, 7, 9, 11, 12)
METRICS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "uniformity": "Uniformity"}
N_IMAGES, N_RUNS, MAX_EVALUATIONS = 15, 30, 30_000
START = r"\section{Experiment 2: CHAOS MRI Images}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_results(sources=None):
    sources = SOURCES if sources is None else sources
    frames = []
    keys = ["image_id", "objective", "k", "run"]
    for algorithm, root in sources.items():
        raw = root / "raw" / "chaos" / algorithm
        paths = sorted(raw.rglob("*.parquet"))
        require(bool(paths), f"{algorithm}: no parquet results under {raw}")
        frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        required = set(keys + list(METRICS) + ["algorithm", "dataset", "status",
                       "contract_ok", "max_evaluations", "used_evaluations", "q",
                       "seed", "best_fitness", "wall_time_s", "thresholds"])
        require(required <= set(frame.columns),
                f"{algorithm}: missing columns {sorted(required - set(frame.columns))}")
        require(len(frame) == N_IMAGES * len(OBJECTIVES) * len(KS) * N_RUNS,
                f"{algorithm}: expected 8,100 full-study runs, found {len(frame)}")
        for column, expected in [("algorithm", algorithm), ("dataset", "chaos"),
                                 ("status", "ok"), ("contract_ok", True),
                                 ("max_evaluations", MAX_EVALUATIONS),
                                 ("used_evaluations", MAX_EVALUATIONS)]:
            require(frame[column].eq(expected).all(),
                    f"{algorithm}: invalid {column}; expected {expected!r}")
        require(frame.image_id.nunique() == N_IMAGES and frame.image_id.notna().all(),
                f"{algorithm}: expected {N_IMAGES} distinct image IDs")
        require(set(frame.objective) == set(OBJECTIVES) and set(frame.k) == set(KS),
                f"{algorithm}: incomplete objectives or threshold counts")
        require(set(frame.run) == set(range(N_RUNS)) and not frame.duplicated(keys).any(),
                f"{algorithm}: missing or duplicate run indices")
        counts = frame.groupby(keys[:-1]).size()
        require(len(counts) == N_IMAGES * len(OBJECTIVES) * len(KS)
                and counts.eq(N_RUNS).all(), f"{algorithm}: incomplete cells")
        require(np.isclose(frame.loc[frame.objective.eq("tsallis"), "q"], 0.8).all(),
                f"{algorithm}: expected Tsallis q=0.8")
        numeric = list(METRICS) + ["best_fitness", "wall_time_s", "seed"]
        require(np.isfinite(frame[numeric].to_numpy(dtype=float)).all(),
                f"{algorithm}: non-finite metrics, fitness, timing or seeds")
        require(frame.wall_time_s.ge(0).all(), f"{algorithm}: negative timings")
        for row in frame.itertuples():
            thresholds = parse_thresholds(row.thresholds)
            require(len(thresholds) == row.k and np.all(np.diff(thresholds) > 0)
                    and np.all((thresholds >= 1) & (thresholds <= 254)),
                    f"{algorithm}: invalid thresholds for {row.image_id}, run {row.run}")
        frames.append(frame)
    require(bool(frames), "No algorithms selected")
    frame = pd.concat(frames, ignore_index=True)
    seeds = frame.pivot(index=keys, columns="algorithm", values="seed")
    require(seeds.notna().all().all(), "Algorithms do not cover identical paired cells")
    require(seeds.eq(seeds.iloc[:, 0], axis=0).all().all(), "Paired run seeds differ")
    return frame


def parse_thresholds(value):
    values = np.asarray([float(x) for x in str(value).split(",")])
    require(np.isfinite(values).all() and np.equal(values, np.round(values)).all(),
            "Thresholds must be comma-separated finite integers")
    return values.astype(int)


def summary(frame, keys, metrics):
    table = frame.groupby(keys)[metrics].agg(["mean", "std"])
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    return table.reset_index()


def escape_tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def table_tex(table, objective, metric, label):
    name = r"Tsallis ($q=0.8$)" if objective == "tsallis" else objective.title()
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{{label} on CHAOS MRI using {name}: pooled mean $\pm$ sample "
             f"standard deviation across {N_IMAGES * N_RUNS} observations per entry "
             f"({N_IMAGES} slices, {N_RUNS} runs each). Dashes indicate unavailable results.}}",
             rf"\label{{tab:chaos-{objective}-{metric}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}",
             r"\toprule", "Algorithm & " + " & ".join(f"$K={k}$" for k in KS) + r" \\",
             r"\midrule"]
    for algorithm, name in LABELS.items():
        subset = table[(table.algorithm == algorithm) &
                       (table.objective == objective)].set_index("k")
        values = [rf"${subset.loc[k, metric + '_mean']:.4f} \pm "
                  f"{subset.loc[k, metric + '_std']:.4f}$" if k in subset.index and metric + '_mean' in subset.columns else "--"
                  for k in KS]
        lines.append(name + " & " + " & ".join(values) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"])


def save_figure(fig, destination):
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figures(frame, sources, destination, image_id=None):
    from src.datasets import load_dataset
    from src.segmentation import display_image

    records = load_dataset("chaos")
    images = {}
    for record in records:
        for field in ("image_id", "id", "name"):
            value = getattr(record, field, None)
            if value is not None:
                images[str(value)] = record
        path = getattr(record, "path", None)
        if path is not None:
            images.setdefault(Path(path).stem, record)
    image_id = str(image_id) if image_id is not None else sorted(frame.image_id.unique())[0]
    require(image_id in images,
            f"Cannot match saved image ID {image_id!r} to ImageRecord; check its ID field")
    image = np.asarray(images[image_id].load()).squeeze()
    require(image.ndim == 2, "Expected a preprocessed grayscale MRI slice")
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5))
    for ax, objective in zip(axes, OBJECTIVES):
        for algorithm, root in sources.items():
            key = "tsallis_q0.8" if objective == "tsallis" else objective
            path = root / "curves/chaos" / algorithm / key / f"K12/{image_id}.npy"
            curves = np.load(path)
            require(curves.shape == (N_RUNS, 201), f"Unexpected curve shape: {path}")
            require(np.isfinite(curves[:, 1:]).all()
                    and np.all(np.diff(curves[:, 1:], axis=1) >= -1e-5),
                    f"Invalid best-so-far curves: {path}")
            expected = frame[(frame.algorithm == algorithm) & (frame.objective == objective)
                             & (frame.k == 12) & (frame.image_id == image_id)].sort_values("run")
            require(np.allclose(curves[:, -1], expected.best_fitness, rtol=1e-5),
                    f"Curve endpoints disagree with recorded fitness: {path}")
            grid = np.linspace(0, MAX_EVALUATIONS, curves.shape[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low, median, high = np.nanpercentile(curves, [25, 50, 75], axis=0)
            ax.plot(grid, median, label=LABELS[algorithm], color=COLOURS[algorithm])
            ax.fill_between(grid, low, high, color=COLOURS[algorithm], alpha=0.2)
        ax.set(title=objective.title(), xlabel="Function evaluations", ylabel="Best fitness")
        ax.grid(alpha=0.2)
        ax.legend()
    save_figure(fig, destination / "exp2_convergence.png")
    for objective in OBJECTIVES:
        fig, axes = plt.subplots(len(sources), 4, figsize=(9, 3 * len(sources)), squeeze=False)
        for row, algorithm in enumerate(sources):
            axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row, 0].set_title(f"{LABELS[algorithm]}: original")
            for col, k in enumerate((3, 7, 12), start=1):
                subset = frame[(frame.algorithm == algorithm) & (frame.objective == objective)
                               & (frame.image_id == image_id) & (frame.k == k)].copy()
                subset["distance"] = (subset.best_fitness - subset.best_fitness.median()).abs()
                selected = subset.sort_values(["distance", "run"]).iloc[0]
                axes[row, col].imshow(display_image(image, parse_thresholds(selected.thresholds)),
                                      cmap="gray", vmin=0, vmax=255)
                axes[row, col].set_title(f"K={k}, run={int(selected.run)}")
        for ax in axes.flat:
            ax.axis("off")
        save_figure(fig, destination / f"exp2_segmentation_{objective}.png")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for algorithm in sources:
        timing = frame[frame.algorithm == algorithm].groupby("k").wall_time_s.agg(["mean", "std"])
        ax.errorbar(timing.index, timing["mean"], yerr=timing["std"], marker="o", capsize=3,
                    label=LABELS[algorithm], color=COLOURS[algorithm])
    ax.set(xlabel="Number of thresholds K", ylabel="Optimisation time per run (s)")
    ax.set_xticks(KS)
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, destination / "exp2_runtime.png")
    return image_id


def figure_tex(filename, caption, label):
    return "\n".join([r"\begin{figure}[htbp]", r"\centering",
                       rf"\includegraphics[width=0.9\textwidth]{{{filename}}}",
                       rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{figure}"])


def section_tex(table, frame, sources, metrics, image_id, gt_description):
    names = ", ".join(LABELS[a] for a in sources)
    n_images = frame.image_id.nunique()
    n_runs = frame.groupby(["algorithm", "image_id", "objective", "k"]).size().iloc[0]
    budget = int(frame.max_evaluations.iloc[0])
    text = [START, r"\label{sec:experiment2}",
            r"\subsection{Experimental Objective}",
            f"Results for {names} on {n_images} CHAOS MRI images, using Otsu, Kapur "
            r"and Tsallis ($q=0.8$), with $K\in\{3,5,7,9,11,12\}$. "
            f"Each image--objective--threshold combination contains {n_runs} runs "
            f"with {budget:,} function evaluations per run.",
            r"\subsection{Ground-Truth Mask Matching}",
            escape_tex(gt_description) if gt_description else "--",
            r"\subsection{Reconstruction Metrics}",
            "Tables report pooled means and sample standard deviations across images "
            "and runs. Dashes indicate unavailable results."]
    for objective in OBJECTIVES:
        for metric, label in metrics.items():
            if metric not in ("dice", "jaccard"):
                text.append(table_tex(table, objective, metric, label))
    text.append(r"\subsection{Ground-Truth Segmentation Performance}")
    for objective in OBJECTIVES:
        for metric, label in [("dice", "Dice coefficient"), ("jaccard", "Jaccard index")]:
            text.append(table_tex(table, objective, metric, label))
    text.append(r"\subsection{Visual MRI Segmentation Results}")
    for objective in OBJECTIVES:
        text.append(figure_tex(f"exp2_segmentation_{objective}.png",
            f"Original and thresholded image {escape_tex(image_id)} using {objective.title()} "
            f"at $K=3,7,12$. Rows: {names}. Solutions are selected by final fitness "
            "closest to the median, with ties resolved by the lowest run index. "
            "The displayed run number identifies an independent repetition.",
            f"fig:chaos-{objective}"))
    text += [r"\subsection{Convergence Analysis}",
             figure_tex("exp2_convergence.png",
             f"Best-so-far fitness for image {escape_tex(image_id)} at $K=12$. "
             f"Lines show medians over {n_runs} runs; bands show interquartile ranges.",
             "fig:chaos-convergence"),
             r"\subsection{Execution Time}",
             figure_tex("exp2_runtime.png",
             "Mean recorded optimisation time versus $K$, with sample standard deviation "
             "as error bars, pooled across images, objectives and runs.",
             "fig:chaos-runtime"),
             r"\subsection{Discussion of Experiment 2}", "--"]
    return "\n\n".join(text) + "\n\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", action="append", metavar="ALGORITHM=PATH")
    parser.add_argument("--tex", type=Path, default=ROOT / "temp.tex")
    parser.add_argument("--image-id", help="saved CHAOS image ID for illustrative figures")
    parser.add_argument("--gt-description", help="plain text describing saved overlap metrics and mask matching")
    args = parser.parse_args(argv)
    sources = dict(SOURCES)
    if args.source:
        sources = {}
        for entry in args.source:
            algorithm, separator, path = entry.partition("=")
            require(separator and algorithm in LABELS and bool(path),
                    f"Invalid --source {entry!r}; expected a known ALGORITHM=PATH")
            require(algorithm not in sources, f"Duplicate source: {algorithm}")
            sources[algorithm] = Path(path).resolve()
    tex_path = args.tex.resolve()
    original = tex_path.read_bytes()
    tex = original.decode("utf-8")
    require(tex.count(START) == 1, "Expected exactly one Experiment 2 section in template")
    start = tex.index(START)
    following = re.search(r"(?m)^\\section\{", tex[start + len(START):])
    require(following is not None, "Cannot find the next section after Experiment 2")
    end = start + len(START) + following.start()
    frame = load_results(sources)
    metrics = dict(METRICS)
    for metric, label in [("class_separability", "Class separability"),
                          ("dice", "Dice coefficient"), ("jaccard", "Jaccard index")]:
        if metric in frame:
            require(np.isfinite(frame[metric].to_numpy(dtype=float)).all(),
                    f"Incomplete or invalid optional {metric} values")
            if metric in ("dice", "jaccard"):
                require(bool(args.gt_description), f"Saved {metric} requires --gt-description")
                require(frame[metric].between(0, 1).all(), f"{metric} must lie in [0,1]")
            metrics[metric] = label
    output_root = tex_path.parent
    report = output_root / "report"
    report.mkdir(parents=True, exist_ok=True)
    pooled = summary(frame, ["objective", "k", "algorithm"], list(metrics))
    per_image = summary(frame, ["image_id", "objective", "k", "algorithm"], list(metrics))
    image_id = figures(frame, sources, output_root / "figures", args.image_id)
    pooled.to_csv(report / "exp2_pooled_summary.csv", index=False)
    per_image.to_csv(report / "exp2_per_image_summary.csv", index=False)
    section = section_tex(pooled, frame, sources, metrics, image_id, args.gt_description)
    backup = tex_path.with_name(tex_path.name + ".exp2.bak")
    if not backup.exists():
        shutil.copy2(tex_path, backup)
    temporary = tex_path.with_name(tex_path.name + ".exp2.tmp")
    temporary.write_bytes((tex[:start] + section + tex[end:]).encode("utf-8"))
    temporary.replace(tex_path)
    print(f"Validated {len(frame):,} saved runs; updated Experiment 2 in {tex_path}.")
    print(f"Wrote five PNG figures and pooled/per-image CSVs. Original backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
