import numpy as np

from ml.posterior_metrics import compare_posteriors, posterior_summary


def test_phase3_posterior_summaries_are_finite_and_ordered():
    axes = {"h": np.linspace(0.59, 0.76, 5), "OmegaM": np.linspace(0.20, 0.40, 5)}
    post = np.ones((5, 5)) / 25
    summary = posterior_summary(axes, post)
    for param_summary in summary.values():
        assert np.isfinite(param_summary["mean"])
        assert np.isfinite(param_summary["std"])
        assert param_summary["ci68"][0] <= param_summary["ci68"][1]
        assert param_summary["ci95"][0] <= param_summary["ci95"][1]


def test_phase3_posterior_comparison_metrics_are_valid():
    axes = {"h": np.linspace(0.59, 0.76, 5), "OmegaM": np.linspace(0.20, 0.40, 5)}
    sim = -np.square(np.arange(25).reshape(5, 5) - 12)
    nsf = sim + 0.01
    metrics = compare_posteriors(axes, sim, nsf, {"h": 0.67, "OmegaM": 0.30, "sigma8": 0.85})
    assert metrics["posterior_jsd"] >= 0.0
    assert 0.0 <= metrics["total_variation"] <= 1.0
    assert all(np.isfinite(v) for v in metrics["normalized_posterior_shift"].values())
