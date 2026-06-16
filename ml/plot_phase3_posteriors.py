import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml.phase3_common import PARAM_NAMES, ensure_dir, load_npz_with_metadata, normalize_log_grid
from ml.posterior_metrics import compare_posteriors, marginalize_to_axes


LABELS = {"h": r"$h$", "OmegaM": r"$\Omega_M$", "sigma8": r"$\sigma_8$"}


def contour_levels(post: np.ndarray) -> list[float]:
    flat = np.sort(post.ravel())[::-1]
    csum = np.cumsum(flat)
    levels = []
    for mass in [0.95, 0.68]:
        idx = min(int(np.searchsorted(csum, mass)), flat.size - 1)
        levels.append(float(flat[idx]))
    return sorted(set(levels))


def plot_pair(ax, axes, sim_post, nsf_post, pair, truth):
    kept_axes, sim_m = marginalize_to_axes(axes, sim_post, pair)
    _, nsf_m = marginalize_to_axes(axes, nsf_post, pair)
    x = kept_axes[pair[0]]
    y = kept_axes[pair[1]]
    ax.contour(x, y, sim_m.T, levels=contour_levels(sim_m), colors="#d95f02", linestyles="--", linewidths=1.8)
    ax.contour(x, y, nsf_m.T, levels=contour_levels(nsf_m), colors="#1b9e77", linewidths=1.8)
    ax.plot(truth[pair[0]], truth[pair[1]], "ks", ms=4)
    ax.set_xlabel(LABELS[pair[0]])
    ax.set_ylabel(LABELS[pair[1]])


def make_figures(grid_file: str | Path, output_dir: str | Path) -> dict:
    arrays, md = load_npz_with_metadata(grid_file)
    axes = {k.removeprefix("axis_"): arrays[k] for k in arrays if k.startswith("axis_")}
    truth = md["truth"]
    sim_post = normalize_log_grid(arrays["sim_log_likelihood"])
    nsf_post = normalize_log_grid(arrays["nsf_log_likelihood"])
    metrics = compare_posteriors(axes, arrays["sim_log_likelihood"], arrays["nsf_log_likelihood"], truth)
    out = ensure_dir(output_dir)
    paths = {}

    params = [p for p in PARAM_NAMES if p in axes]
    pairs = [(params[i], params[j]) for i in range(len(params)) for j in range(i + 1, len(params))]
    if pairs:
        fig, axs = plt.subplots(1, len(pairs), figsize=(5.0 * len(pairs), 4.2), squeeze=False)
        for ax, pair in zip(axs[0], pairs):
            plot_pair(ax, axes, sim_post, nsf_post, pair, truth)
        axs[0, 0].plot([], [], color="#d95f02", ls="--", label="Simulator")
        axs[0, 0].plot([], [], color="#1b9e77", label="NSF")
        axs[0, 0].legend(frameon=False)
        fig.tight_layout()
        paths["contour_overlays"] = out / "contour_overlays.png"
        fig.savefig(paths["contour_overlays"], dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    names = list(metrics["normalized_posterior_shift"])
    vals = [metrics["normalized_posterior_shift"][n] for n in names]
    ax.bar([LABELS[n] for n in names], vals, color="#7570b3")
    ax.axhline(0.25, color="black", lw=1, ls=":", label="Ideal")
    ax.axhline(0.50, color="#d95f02", lw=1, ls="--", label="Acceptable")
    ax.set_ylabel(r"$|\mu_{\rm NSF}-\mu_{\rm sim}|/\sigma_{\rm sim}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths["posterior_mean_shift"] = out / "posterior_mean_shift.png"
    fig.savefig(paths["posterior_mean_shift"], dpi=200)
    plt.close(fig)

    return {k: str(v) for k, v in paths.items()}


def plot_coverage(coverage_json: str | Path, output_dir: str | Path) -> str | None:
    path = Path(coverage_json)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    stats = data.get("coverage_summary", {}).get("parameters", {})
    if not stats:
        return None
    names = list(stats)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x - 0.08, [stats[n]["coverage_68"] for n in names], yerr=[stats[n]["binomial_sigma_68"] for n in names], fmt="o", label="68%")
    ax.errorbar(x + 0.08, [stats[n]["coverage_95"] for n in names], yerr=[stats[n]["binomial_sigma_95"] for n in names], fmt="s", label="95%")
    ax.axhline(0.68, color="black", ls=":", lw=1)
    ax.axhline(0.95, color="black", ls="--", lw=1)
    ax.set_xticks(x, [LABELS.get(n, n) for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Observed coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = ensure_dir(output_dir) / "coverage.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return str(out)


def plot_runtime(runtime_json: str | Path, output_dir: str | Path) -> str | None:
    path = Path(runtime_json)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    timings = data.get("timings", data)
    labels = [k for k in timings if k.endswith("_seconds")]
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([x.replace("_seconds", "").replace("_", "\n") for x in labels], [timings[k] for k in labels], color="#66a61e")
    ax.set_yscale("log")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    out = ensure_dir(output_dir) / "runtime_speedup.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 3 posterior benchmark figures.")
    parser.add_argument("--grid_file", default=None)
    parser.add_argument("--grid_dir", default="data/results/phase3_posterior_grids")
    parser.add_argument("--coverage_json", default="data/results/phase3_coverage_results.json")
    parser.add_argument("--runtime_json", default="data/results/phase3_runtime_results.json")
    parser.add_argument("--output_dir", default="plots/figures/phase3_posterior_benchmark")
    parser.add_argument("--output_report", default="docs/phase3_figures_summary.md")
    args = parser.parse_args()

    grid_file = Path(args.grid_file) if args.grid_file else None
    if grid_file is None:
        files = sorted(Path(args.grid_dir).glob("*combined*.npz"))
        grid_file = files[0] if files else None
    paths = {}
    if grid_file:
        paths.update(make_figures(grid_file, args.output_dir))
    cov = plot_coverage(args.coverage_json, args.output_dir)
    if cov:
        paths["coverage"] = cov
    rt = plot_runtime(args.runtime_json, args.output_dir)
    if rt:
        paths["runtime"] = rt

    lines = ["# Phase 3 Figures Summary", "", f"Generated {len(paths)} figure(s).", ""]
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`")
    ensure_dir(Path(args.output_report).parent)
    Path(args.output_report).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
