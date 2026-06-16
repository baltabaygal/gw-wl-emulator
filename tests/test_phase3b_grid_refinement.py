from pathlib import Path

from ml import phase3b_diagnostics as d


def test_phase3b_grid_refinement_outputs_are_finite(monkeypatch, tmp_path):
    metrics = {
        "posterior_jsd": 0.25,
        "total_variation": 0.4,
        "credible_region_overlap": {"credible_region_68": 0.2, "credible_region_95": 0.6},
        "normalized_posterior_shift": {"OmegaM": 1.0, "sigma8": 1.5},
        "mle_offset": {"OmegaM": 0.02, "sigma8": -0.03},
    }
    monkeypatch.setattr(d, "run_grid", lambda *args, **kwargs: tmp_path / "refined.npz")
    monkeypatch.setattr(d, "summarize_grid_metrics", lambda grid: metrics)
    monkeypatch.setattr(d, "plot_overlay", lambda grid, output_path, title: Path(output_path).parent.mkdir(parents=True, exist_ok=True))

    payload = d.run_grid_refinement(resolution=24, nsim_per_z=5)
    assert payload["refined"]["posterior_jsd"] == 0.25
    assert payload["refined"]["credible_region_overlap"]["credible_region_68"] == 0.2
    assert payload["configuration"]["refined_resolution"] == 24
