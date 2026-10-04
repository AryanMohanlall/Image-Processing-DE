"""Populate temp.tex with the completed DE/LADE BSD500 results.

Run from the repository root: python -m experiments.report_exp1
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets import load_dataset
from src.segmentation import display_image

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "standard_de": ROOT / "results/exp1_standardDE",
    "late_acceptance_de": ROOT / "results/exp1_late_acceptance_de",
}
LABELS = {"standard_de": "DE", "late_acceptance_de": "LADE"}
OBJECTIVES = ("otsu", "kapur", "tsallis")
KS = (3, 5, 7, 9, 11, 12)
METRICS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "uniformity": "Uniformity"}
COLOURS = {"standard_de": "#4C72B0", "late_acceptance_de": "#8172B3"}


def load_results():
    frames = []
    for algorithm, root in SOURCES.items():
        paths = sorted((root / "raw").rglob("*.parquet"))
        assert len(paths) == 180, (algorithm, "expected 180 result files")
        frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        assert len(frame) == 5400 and frame.algorithm.eq(algorithm).all()
        assert frame.dataset.eq("bsd500").all() and frame.status.eq("ok").all()
        assert frame.contract_ok.all()
        assert frame.max_evaluations.eq(30000).all()
        assert frame.used_evaluations.eq(30000).all()
        assert frame.loc[frame.objective.eq("tsallis"), "q"].eq(0.8).all()
        keys = ["image_id", "objective", "k"]
        assert frame.groupby(keys).size().eq(30).all()
        assert not frame.duplicated(keys + ["run"]).any()
        assert set(frame.image_id) == {f"img{i}" for i in range(1, 11)}
        assert set(frame.objective) == set(OBJECTIVES) and set(frame.k) == set(KS)
        assert set(frame.run) == set(range(30))
        assert np.isfinite(frame[list(METRICS) + ["best_fitness", "wall_time_s"]]).all().all()
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    seeds = frame.pivot(index=["image_id", "objective", "k", "run"],
                        columns="algorithm", values="seed")
    assert seeds.standard_de.eq(seeds.late_acceptance_de).all()
    return frame


def summary(frame, keys):
    table = frame.groupby(keys)[list(METRICS)].agg(["mean", "std"])
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    return table.reset_index()


def table_tex(table, objective, metric):
    name = objective.title() if objective != "tsallis" else r"Tsallis ($q=0.8$)"
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             f"\\caption{{{METRICS[metric]} on BSD500 using {name}: pooled mean "
             r"$\pm$ sample standard deviation across 300 observations per entry "
             "(10 images, 30 runs each).}",
             f"\\label{{tab:bsd-{objective}-{metric}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}",
             r"\toprule", "Algorithm & " + " & ".join(f"$K={k}$" for k in KS) + r" \\",
             r"\midrule"]
    for algorithm in SOURCES:
        subset = table[(table.algorithm == algorithm) & (table.objective == objective)].set_index("k")
        values = [f"${subset.loc[k, metric + '_mean']:.4f} \\pm "
                  f"{subset.loc[k, metric + '_std']:.4f}$" for k in KS]
        lines.append(LABELS[algorithm] + " & " + " & ".join(values) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"])


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figures(frame):
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5))
    for ax, objective in zip(axes, OBJECTIVES):
        for algorithm, root in SOURCES.items():
            key = "tsallis_q0.8" if objective == "tsallis" else objective
            curves = np.load(root / "curves/bsd500" / algorithm / key / "K12/img1.npy")
            assert curves.shape == (30, 201)
            assert np.all(np.diff(curves[:, 1:], axis=1) >= -1e-5)
            expected = frame[(frame.algorithm == algorithm) & (frame.objective == objective)
                             & (frame.k == 12) & (frame.image_id == "img1")].sort_values("run")
            assert np.allclose(curves[:, -1], expected.best_fitness, rtol=1e-5)
            grid = np.linspace(0, 30000, curves.shape[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low, median, high = np.nanpercentile(curves, [25, 50, 75], axis=0)
            ax.plot(grid, median, label=LABELS[algorithm], color=COLOURS[algorithm])
            ax.fill_between(grid, low, high, color=COLOURS[algorithm], alpha=0.2)
        ax.set(title=objective.title(), xlabel="Function evaluations", ylabel="Best fitness")
        ax.grid(alpha=0.2)
        ax.legend()
    save_figure(fig, "exp1_convergence.png")

    image = load_dataset("bsd500")[0].load()
    for objective in OBJECTIVES:
        fig, axes = plt.subplots(2, 4, figsize=(9, 6))
        for row, algorithm in enumerate(SOURCES):
            axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row, 0].set_title(f"{LABELS[algorithm]}: original")
            for col, k in enumerate((3, 7, 12), start=1):
                subset = frame[(frame.algorithm == algorithm) & (frame.objective == objective)
                               & (frame.image_id == "img1") & (frame.k == k)].copy()
                subset["distance"] = (subset.best_fitness - subset.best_fitness.median()).abs()
                selected = subset.sort_values(["distance", "run"]).iloc[0]
                thresholds = np.fromstring(selected.thresholds, sep=",", dtype=int)
                axes[row, col].imshow(display_image(image, thresholds), cmap="gray", vmin=0, vmax=255)
                axes[row, col].set_title(f"K={k}, run={int(selected.run)}")
        for ax in axes.flat:
            ax.axis("off")
        save_figure(fig, f"exp1_segmentation_{objective}.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for algorithm in SOURCES:
        timing = frame[frame.algorithm == algorithm].groupby("k").wall_time_s.agg(["mean", "std"])
        ax.errorbar(timing.index, timing["mean"], yerr=timing["std"], marker="o", capsize=3,
                    label=LABELS[algorithm], color=COLOURS[algorithm])
    ax.set(xlabel="Number of thresholds K", ylabel="Optimisation time per run (s)")
    ax.set_xticks(KS)
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, "exp1_runtime.png")


def section_tex(table, frame):
    lookup = table.set_index(["algorithm", "objective", "k"])
    def value(algorithm, objective, k, metric):
        return lookup.loc[(algorithm, objective, k), metric + "_mean"]
    text = [r"\section{Experiment 1: BSD500 Benchmark Images}", r"\label{sec:experiment1}",
            r"\subsection{Experimental Objective and Completed Runs}",
            "This section reports the completed Standard DE (DE) and Late Acceptance DE (LADE) "
            "experiments on the ten supplied BSD500 images. Results for JADE, SHADE and L-SHADE "
            "are not included in this comparison. Each algorithm was evaluated with Otsu, Kapur "
            "and Tsallis ($q=0.8$) at $K\\in\\{3,5,7,9,11,12\\}$, using 30 runs per combination "
            "and a budget of 30,000 function evaluations, including initialisation. The saved "
            "results contain 5,400 successful runs per algorithm (10,800 in total); every run "
            "used the full budget and passed the recorded fitness-contract check.",
            r"\subsection{Reconstruction Performance}",
            "Tables below report the mean and sample standard deviation of PSNR, SSIM and "
            "uniformity. Each entry pools 300 observations: 30 independent runs on each of ten "
            "images. Consequently, these standard deviations combine differences between images "
            "with stochastic variation within an image; they should not be interpreted solely "
            "as algorithm stability. Per-image means and standard deviations over the 30 runs "
            "are supplied in \\texttt{report/exp1\\_per\\_image\\_summary.csv}. "
            "Metrics use class-mean reconstruction, and higher values indicate better reconstruction "
            "quality. The displayed values are descriptive comparisons, without a claim of "
            "statistical significance."]
    for objective in OBJECTIVES:
        for metric in METRICS:
            text.append(table_tex(table, objective, metric))
    text += [r"\clearpage", r"\subsection{Effect of Threshold Level}",
             "Mean PSNR and SSIM increase from $K=3$ to $K=12$ for both algorithms under "
             "all three objectives. More thresholds permit finer intensity reconstruction, "
             "although gains diminish at higher $K$. For DE with Otsu, mean PSNR increases "
             f"from {value('standard_de','otsu',3,'psnr'):.4f} to "
             f"{value('standard_de','otsu',12,'psnr'):.4f} dB, while mean SSIM increases from "
             f"{value('standard_de','otsu',3,'ssim'):.4f} to "
             f"{value('standard_de','otsu',12,'ssim'):.4f}. "
             "Uniformity is not strictly monotonic: DE's Kapur uniformity decreases slightly "
             "between $K=11$ and $K=12$, and Tsallis uniformity falls between $K=3$ and $K=5$. "
             "This metric includes an explicit factor of $K$, so improved PSNR does not guarantee "
             "a larger uniformity score.",
             r"\subsection{Effect of Objective Function}",
             "Otsu gives the highest mean PSNR and uniformity for both algorithms at every "
             "tested threshold count. Otsu also gives the highest mean SSIM at every "
             "tested threshold count. For DE at $K=12$, the mean PSNR values are "
             f"{value('standard_de','otsu',12,'psnr'):.4f}, "
             f"{value('standard_de','kapur',12,'psnr'):.4f}, and "
             f"{value('standard_de','tsallis',12,'psnr'):.4f} dB for Otsu, Kapur and Tsallis, "
             "respectively. Entropy maximisation and reconstruction quality are different "
             "criteria: a higher entropy objective does not necessarily produce a lower "
             "reconstruction error. Only $q=0.8$ was evaluated, so these results cannot "
             "establish the effect of changing $q$.",
             r"\subsection{Convergence Analysis}",
             r"\begin{figure}[htbp]", r"\centering",
             r"\includegraphics[width=\textwidth]{exp1_convergence.png}",
             r"\caption{Best-so-far fitness for img1 at $K=12$. Lines show the median over "
             "30 runs and shaded bands show the interquartile range. Each panel uses its "
             "own objective scale; fitness values are not comparable across objectives.}",
             r"\label{fig:bsd_convergence}", r"\end{figure}",
             "Figure~\\ref{fig:bsd_convergence} compares convergence under an equal evaluation "
             "budget. Best-so-far fitness is retained independently of the current population "
             "in LADE, so its curves remain nondecreasing despite acceptance of some worse "
             "trials. These curves describe one illustrative image and do not alone establish "
             "a dataset-wide convergence advantage.",
             r"\subsection{Visual Segmentation Results}"]
    for objective in OBJECTIVES:
        text += [r"\begin{figure}[htbp]", r"\centering",
                 f"\\includegraphics[width=0.9\\textwidth]{{exp1_segmentation_{objective}.png}}",
                 f"\\caption{{Original and segmented img1 under {objective.title()} at "
                 "$K=3,7,12$. DE occupies the first row and LADE the second. Each solution "
                 "is selected by fitness closest to the median of its 30 runs (ties use "
                 "the lowest run index). Class intensities are spread evenly for display; "
                 "PSNR and SSIM are computed using class means instead.}",
                 f"\\label{{fig:bsd-segmentation-{objective}}}", r"\end{figure}"]
    text += [r"\clearpage", r"\subsection{Execution Time}",
             r"\begin{figure}[htbp]", r"\centering",
             r"\includegraphics[width=0.8\textwidth]{exp1_runtime.png}",
             r"\caption{Mean recorded optimisation time versus $K$. Error bars show sample "
             "standard deviation over 900 runs per algorithm and threshold count "
             "(10 images, three objectives, 30 runs). DE and LADE were run in separate sweeps, "
             "so machine load and worker settings can influence this comparison.}",
             r"\label{fig:bsd-runtime}", r"\end{figure}",
             "Timing covers optimiser execution and objective evaluations using "
             "\\texttt{time.perf\\_counter}; image loading, histogram preparation, exact "
             "reference computation, metric calculation and result writing are excluded. "
             "The saved results do not record hardware or worker counts, so runtime differences "
             "cannot be attributed exclusively to algorithm design.",
             r"\subsection{Discussion of Experiment 1}",
             "Neither algorithm consistently dominates the reconstruction metrics. DE has "
             "slightly higher mean Otsu PSNR from $K=7$ through $K=12$, whereas LADE has "
             "higher mean Kapur PSNR at $K=5,9,11,12$. At $K=12$, LADE exceeds DE's "
             "Kapur PSNR by "
             f"{value('late_acceptance_de','kapur',12,'psnr')-value('standard_de','kapur',12,'psnr'):.4f} "
             "dB, while DE exceeds LADE's Otsu PSNR by "
             f"{value('standard_de','otsu',12,'psnr')-value('late_acceptance_de','otsu',12,'psnr'):.4f} "
             "dB. The Otsu results are identical at $K=3$. These small descriptive differences "
             "do not demonstrate significant superiority. LADE retains fixed DE control "
             "parameters and changes selection through a best-per-history-slot policy; "
             "this policy should be considered when interpreting the comparison. "
             "The experiment supports the expected improvement in reconstruction with more "
             "thresholds and shows a stronger descriptive effect of objective choice than "
             "of switching between these two optimisers. Full group conclusions require "
             "the remaining algorithms, paired statistical analysis and the MRI experiment."]
    return "\n\n".join(text) + "\n\n"


def main():
    frame = load_results()
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "report").mkdir(exist_ok=True)
    pooled = summary(frame, ["objective", "k", "algorithm"])
    per_image = summary(frame, ["image_id", "objective", "k", "algorithm"])
    pooled.to_csv(ROOT / "report/exp1_pooled_summary.csv", index=False)
    per_image.to_csv(ROOT / "report/exp1_per_image_summary.csv", index=False)
    figures(frame)
    path = ROOT / "temp.tex"
    tex = path.read_text(encoding="utf-8")
    tex = tex.removeprefix("```latex\n").rstrip().removesuffix("```").rstrip() + "\n"
    if r"\usepackage{graphicx}" not in tex:
        tex = tex.replace(r"\usepackage{hyperref}", "\\usepackage{graphicx}\n\\usepackage{hyperref}")
    start = tex.index(r"\section{Experiment 1: BSD500 Benchmark Images}")
    end = tex.index(r"\section{Experiment 2: CHAOS MRI Images}")
    tex = tex[:start] + section_tex(pooled, frame) + tex[end:]
    path.write_text(tex, encoding="utf-8")
    print("Validated 10,800 runs; populated Experiment 1 in temp.tex.")
    print("Wrote five PNG figures at 300 DPI and pooled/per-image summary CSVs.")


if __name__ == "__main__":
    main()
