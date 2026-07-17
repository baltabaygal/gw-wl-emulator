from pathlib import Path

from ml import phase3b_diagnostics as d


def test_phase3b_redshift_decomposition_returns_expected_structure(monkeypatch, tmp_path):
    metrics = {
        "posterior_jsd": 0.3,
        "total_variation": 0.5,
        "credible_region_overlap": {"credible_region_68": 0.1, "credible_region_95": 0.4},
        "normalized_posterior_shift": {"OmegaM": 2.0, "sigma8": 2.5},
        "mle_offset": {"OmegaM": -0.01, "sigma8": 0.04},
    }
    monkeypatch.setattr(d, "summarize_grid_metrics", lambda grid: metrics)
    monkeypatch.setattr(d, "write_slice_catalog", lambda source, z_value, output_dir: tmp_path / f"slice_{z_value}.npz")
    monkeypatch.setattr(d, "run_grid", lambda *args, **kwargs: tmp_path / "grid.npz")
    monkeypatch.setattr(d, "plot_overlay", lambda grid, output_path, title: Path(output_path).parent.mkdir(parents=True, exist_ok=True))

    payload = d.run_redshift_slice_decomposition(nsim_per_z=5, resolution=3)
    assert set(payload["slices"]) == {"low_z", "mid_z", "high_z"}
    assert payload["slices"]["low_z"]["z"] == 0.5
    assert payload["slices"]["high_z"]["posterior_jsd"] == 0.3
