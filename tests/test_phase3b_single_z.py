from pathlib import Path

from ml import phase3b_diagnostics as d


def _fake_metrics():
    return {
        "posterior_jsd": 0.1,
        "total_variation": 0.2,
        "credible_region_overlap": {"credible_region_68": 0.5, "credible_region_95": 0.8},
        "normalized_posterior_shift": {"OmegaM": 0.3, "sigma8": 0.4},
        "mle_offset": {"OmegaM": 0.01, "sigma8": -0.02},
        "sim_summary": {},
        "nsf_summary": {},
    }


def test_phase3b_single_z_pipeline_returns_expected_structure(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "single_z_catalog_path", lambda z, n, seed, output_dir: tmp_path / f"cat_{z}.npz")
    monkeypatch.setattr(d, "run_grid", lambda *args, **kwargs: tmp_path / "grid.npz")
    monkeypatch.setattr(d, "summarize_grid_metrics", lambda grid: _fake_metrics())

    def fake_plot(grid, output_path, title):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("plot")

    monkeypatch.setattr(d, "plot_overlay", fake_plot)
    payload = d.run_single_z_isolation(z_values=(0.5,), n=10, resolution=3, nsim_per_z=4)
    rec = payload["results"]["z=0.5"]
    assert rec["posterior_jsd"] == 0.1
    assert rec["credible_region_overlap"]["credible_region_95"] == 0.8
    assert rec["mle_offset"]["OmegaM"] == 0.01
