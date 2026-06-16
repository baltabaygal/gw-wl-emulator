from pathlib import Path

def test_phase3d_output_reports_exist():
    reports = [
        "docs/phase3d_scaling_report.md",
        "docs/phase3d_scaling_metrics.md",
        "docs/phase3d_redshift_scaling_decomposition.md",
        "docs/phase3d_cache_reproducibility.md",
    ]
    for r in reports:
        assert Path(r).exists(), f"Phase 3D report {r} is missing"

def test_phase3d_output_plots_exist():
    plots = [
        "plots/figures/phase3d_scaling/jsd_scaling.png",
        "plots/figures/phase3d_scaling/tv_scaling.png",
        "plots/figures/phase3d_redshift_scaling/jsd_scaling.png",
        "plots/figures/phase3d_redshift_scaling/tv_scaling.png",
    ]
    for p in plots:
        assert Path(p).exists(), f"Phase 3D plot {p} is missing"
