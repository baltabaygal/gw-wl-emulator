import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from ml.data import load_dataset, get_recommended_bin_edges
from ml.nsf_model import ConditionalNSF
from ml.baselines import (
    compute_binned_moments,
    compute_kl_divergence,
    compute_js_divergence,
    compute_wasserstein_distance
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate NSF continuous/binned densities against simulator.")
    parser.add_argument("--model_path", type=str, default="data/models/conditional_nsf_backend_current.pt")
    parser.add_argument("--dataset_dir", type=str, default="datasets/backend_current_1k")
    parser.add_argument("--plots_dir", type=str, default="plots/figures/phase2b_nsf_pdf_validation")
    parser.add_argument("--output_report", type=str, default="docs/phase2/phase2b_nsf_pdf_validation.md")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)

    # 1. Load model
    print(f"Loading NSF checkpoint from {args.model_path}...")
    model = ConditionalNSF(input_dim=1, context_dim=4)
    model.load_checkpoint(args.model_path)
    model.eval()

    # 2. Setup grids and bins
    bin_edges = get_recommended_bin_edges()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    # 3. Load dataset
    print(f"Loading dataset from {args.dataset_dir}...")
    raw_dataset = load_dataset(args.dataset_dir)

    metrics_by_split = {}
    representative_configs = {"interpolation": None, "ood": None}

    for split_name in ["validation", "test"]:
        if split_name not in raw_dataset:
            continue
            
        print(f"Evaluating {split_name} split...")
        split_data = raw_dataset[split_name]
        lnmu = split_data["lnmu"]
        counts = split_data["valid_counts"]
        z = split_data["z"]
        h = split_data["h"]
        om = split_data["OmegaM"]
        s8 = split_data["sigma8"]
        
        num_configs = lnmu.shape[0]
        
        split_types = []
        if "split_type" in split_data:
            split_types = [
                s.decode("utf-8") if isinstance(s, bytes) else s
                for s in split_data["split_type"]
            ]
        else:
            for i in range(num_configs):
                is_id = (0.20 <= om[i] <= 0.40) and (0.65 <= s8[i] <= 1.05)
                split_types.append("interpolation" if is_id else "ood")

        for i in range(num_configs):
            st = split_types[i]
            if st not in metrics_by_split:
                metrics_by_split[st] = []
                
            z_val, h_val, om_val, s8_val = z[i], h[i], om[i], s8[i]
            context = np.array([z_val, h_val, om_val, s8_val], dtype=np.float32)
            
            # Simulator samples
            n = int(counts[i])
            if n <= 0:
                continue
            lnmu_sim = lnmu[i, :n]
            clamped_sim = np.clip(lnmu_sim, bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
            
            # Simulator binned PDF (sums to 1)
            counts_sim, _ = np.histogram(clamped_sim, bins=bin_edges)
            probs_sim = counts_sim / np.sum(counts_sim)
            
            # NSF PDF evaluated at bin centers (midpoint approximation)
            density_nsf = model.density_grid(context, bin_centers)
            probs_nsf = density_nsf * bin_widths
            probs_nsf = probs_nsf / np.sum(probs_nsf)  # normalize to sum to 1.0
            
            # Compute grid metrics
            kl = compute_kl_divergence(probs_sim, probs_nsf)
            jsd = compute_js_divergence(probs_sim, probs_nsf)
            was = compute_wasserstein_distance(probs_sim, probs_nsf, bin_edges)
            
            # Moments from binned probabilities
            sim_mean, sim_var, sim_skew, sim_kurt = compute_binned_moments(probs_sim, bin_centers)
            nsf_mean, nsf_var, nsf_skew, nsf_kurt = compute_binned_moments(probs_nsf, bin_centers)
            
            # Tail metrics from samples
            nsf_samples = model.sample(context, 10000).flatten()
            
            # Tail masses
            p_gt_02_sim = np.mean(clamped_sim > 0.2)
            p_gt_05_sim = np.mean(clamped_sim > 0.5)
            p_gt_10_sim = np.mean(clamped_sim > 1.0)
            
            p_gt_02_nsf = np.mean(nsf_samples > 0.2)
            p_gt_05_nsf = np.mean(nsf_samples > 0.5)
            p_gt_10_nsf = np.mean(nsf_samples > 1.0)
            
            # Tail quantiles
            q99_sim = np.quantile(clamped_sim, 0.99)
            q995_sim = np.quantile(clamped_sim, 0.995)
            q999_sim = np.quantile(clamped_sim, 0.999)
            
            q99_nsf = np.quantile(nsf_samples, 0.99)
            q995_nsf = np.quantile(nsf_samples, 0.995)
            q999_nsf = np.quantile(nsf_samples, 0.999)

            metrics = {
                "kl": float(kl),
                "jsd": float(jsd),
                "was": float(was),
                "sim_moments": [float(sim_mean), float(sim_var), float(sim_skew), float(sim_kurt)],
                "nsf_moments": [float(nsf_mean), float(nsf_var), float(nsf_skew), float(nsf_kurt)],
                "sim_tail_mass": [float(p_gt_02_sim), float(p_gt_05_sim), float(p_gt_10_sim)],
                "nsf_tail_mass": [float(p_gt_02_nsf), float(p_gt_05_nsf), float(p_gt_10_nsf)],
                "sim_quantiles": [float(q99_sim), float(q995_sim), float(q999_sim)],
                "nsf_quantiles": [float(q99_nsf), float(q995_nsf), float(q999_nsf)]
            }
            metrics_by_split[st].append(metrics)

            # Store first representative config for plotting
            if representative_configs[st] is None:
                representative_configs[st] = {
                    "z": z_val, "h": h_val, "om": om_val, "s8": s8_val,
                    "clamped_sim": clamped_sim, "probs_sim": probs_sim,
                    "probs_nsf": probs_nsf, "nsf_samples": nsf_samples,
                    "context": context
                }

    # 4. Generate plots and write report
    report_lines = [
        "# Phase 2B — NSF PDF and Tail Validation Report",
        "",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Model: `{args.model_path}`",
        "",
        "## 1. Summary Metrics",
        ""
    ]
    
    report_lines.append("| Split Type | Count | Mean KL | Mean JSD | Mean Wasserstein |")
    report_lines.append("|---|---|---|---|---|")
    
    summary_data = {}
    for st, metric_list in metrics_by_split.items():
        if len(metric_list) == 0:
            continue
        kls = [m["kl"] for m in metric_list]
        jsds = [m["jsd"] for m in metric_list]
        wass = [m["was"] for m in metric_list]
        
        report_lines.append(f"| {st} | {len(metric_list)} | {np.mean(kls):.6f} | {np.mean(jsds):.6f} | {np.mean(wass):.6f} |")
        summary_data[st] = {
            "kl": np.mean(kls),
            "jsd": np.mean(jsds),
            "was": np.mean(wass)
        }
        
    # Generate Plots and add to report
    report_lines.append("")
    report_lines.append("## 2. Representative PDF Overlays")
    report_lines.append("")

    for st in ["interpolation", "ood"]:
        config = representative_configs[st]
        if config is None:
            continue
            
        z_v, h_v, om_v, s8_v = config["z"], config["h"], config["om"], config["s8"]
        print(f"Plotting representative config for {st}: z={z_v:.2f}, h={h_v:.3f}, om={om_v:.3f}, s8={s8_v:.3f}")
        
        # Grid density evaluate
        x_dense = np.linspace(-0.5, 2.5, 400)
        density_dense = model.density_grid(config["context"], x_dense)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: PDF comparison
        ax = axes[0]
        # Simulator binned PDF
        ax.bar(bin_centers, config["probs_sim"] / bin_widths, width=bin_widths, alpha=0.4, 
               color='#ff7f0e', edgecolor='#d62728', label='Sim Histogram PDF')
        # NSF continuous density
        ax.plot(x_dense, density_dense, color='#1f77b4', lw=2.5, label='NSF Continuous PDF')
        ax.set_xlabel(r"$\ln \mu$")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"PDF Comparison ({st} config: z={z_v:.2f}, om={om_v:.3f}, s8={s8_v:.3f})")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Right plot: Zoomed log tail comparison
        ax = axes[1]
        # Simulator PDF
        ax.step(bin_centers, config["probs_sim"] / bin_widths, where='mid', color='#ff7f0e', lw=2, label='Sim PDF')
        # NSF Continuous PDF
        ax.plot(x_dense, density_dense, color='#1f77b4', lw=2, label='NSF Continuous PDF')
        
        ax.set_yscale("log")
        ax.set_xlim(0.2, 1.8)
        ax.set_ylim(1e-4, 5e0)
        ax.set_xlabel(r"$\ln \mu$")
        ax.set_ylabel("Log Probability Density")
        ax.set_title("High-Magnification Tail Zoom")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
        plt.tight_layout()
        fig_name = f"pdf_overlay_{st}.png"
        fig_path = os.path.join(args.plots_dir, fig_name)
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        
        report_lines.append(f"### {st.capitalize()} Cosmology Configuration")
        report_lines.append(f"Parameters: $z = {z_v:.2f}, h = {h_v:.3f}, \\Omega_M = {om_v:.3f}, \\sigma_8 = {s8_v:.3f}$")
        report_lines.append(f"![PDF Overlay {st}](figures/phase2b_nsf_pdf_validation/{fig_name})")
        report_lines.append("")

    # Moments table
    report_lines.append("## 3. Moment and Tail Quantiles Analysis")
    report_lines.append("")
    report_lines.append("We compare the recovered moments and high-magnification tails for the representative configurations.")
    report_lines.append("")
    
    for st in ["interpolation", "ood"]:
        config = representative_configs[st]
        if config is None:
            continue
        # Find metrics in list
        m_list = metrics_by_split[st]
        # just use the first element (which corresponds to config)
        m = m_list[0]
        
        report_lines.append(f"### {st.capitalize()} Moments Table")
        report_lines.append("| Metric | Simulator | NSF | Difference |")
        report_lines.append("|---|---|---|---|")
        report_lines.append(f"| Mean | {m['sim_moments'][0]:.6f} | {m['nsf_moments'][0]:.6f} | {m['nsf_moments'][0] - m['sim_moments'][0]:+.6f} |")
        report_lines.append(f"| Variance | {m['sim_moments'][1]:.6f} | {m['nsf_moments'][1]:.6f} | {m['nsf_moments'][1] - m['sim_moments'][1]:+.6f} |")
        report_lines.append(f"| Skewness | {m['sim_moments'][2]:.6f} | {m['nsf_moments'][2]:.6f} | {m['nsf_moments'][2] - m['sim_moments'][2]:+.6f} |")
        report_lines.append(f"| Kurtosis | {m['sim_moments'][3]:.6f} | {m['nsf_moments'][3]:.6f} | {m['nsf_moments'][3] - m['sim_moments'][3]:+.6f} |")
        report_lines.append("")
        
        report_lines.append(f"### {st.capitalize()} High-Magnification Tail Quantiles")
        report_lines.append("| Quantile | Simulator | NSF | Difference |")
        report_lines.append("|---|---|---|---|")
        report_lines.append(f"| 99% | {m['sim_quantiles'][0]:.6f} | {m['nsf_quantiles'][0]:.6f} | {m['nsf_quantiles'][0] - m['sim_quantiles'][0]:+.6f} |")
        report_lines.append(f"| 99.5% | {m['sim_quantiles'][1]:.6f} | {m['nsf_quantiles'][1]:.6f} | {m['nsf_quantiles'][1] - m['sim_quantiles'][1]:+.6f} |")
        report_lines.append(f"| 99.9% | {m['sim_quantiles'][2]:.6f} | {m['nsf_quantiles'][2]:.6f} | {m['nsf_quantiles'][2] - m['sim_quantiles'][2]:+.6f} |")
        report_lines.append("")
        
        report_lines.append(f"### {st.capitalize()} Tail Probability Masses")
        report_lines.append("| Tail Bound | Simulator | NSF | Difference |")
        report_lines.append("|---|---|---|---|")
        report_lines.append(f"| P(lnmu > 0.2) | {m['sim_tail_mass'][0]:.6f} | {m['nsf_tail_mass'][0]:.6f} | {m['nsf_tail_mass'][0] - m['sim_tail_mass'][0]:+.6f} |")
        report_lines.append(f"| P(lnmu > 0.5) | {m['sim_tail_mass'][1]:.6f} | {m['nsf_tail_mass'][1]:.6f} | {m['nsf_tail_mass'][1] - m['sim_tail_mass'][1]:+.6f} |")
        report_lines.append(f"| P(lnmu > 1.0) | {m['sim_tail_mass'][2]:.6f} | {m['nsf_tail_mass'][2]:.6f} | {m['nsf_tail_mass'][2] - m['sim_tail_mass'][2]:+.6f} |")
        report_lines.append("")

    report_str = "\n".join(report_md) if 'report_md' in locals() else "\n".join(report_lines)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"Validation report written to {args.output_report}")

if __name__ == "__main__":
    main()
