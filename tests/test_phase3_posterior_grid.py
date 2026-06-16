import numpy as np

from ml.phase3_common import PRIOR_BOUNDS, build_grid, in_prior, normalize_log_grid


def test_phase3_grid_uses_required_prior_bounds():
    grid = build_grid("3d", 4)
    assert grid["prior_bounds"]["h"] == list(PRIOR_BOUNDS["h"])
    assert grid["prior_bounds"]["OmegaM"] == list(PRIOR_BOUNDS["OmegaM"])
    assert grid["prior_bounds"]["sigma8"] == list(PRIOR_BOUNDS["sigma8"])
    assert in_prior(np.array([0.67, 0.30, 0.85]))
    assert not in_prior(np.array([0.80, 0.30, 0.85]))


def test_phase3_posterior_grid_normalizes_and_rejects_bad_values():
    post = normalize_log_grid(np.array([[0.0, -1.0], [-2.0, -3.0]]))
    assert np.isclose(post.sum(), 1.0)
    assert np.all(post > 0)
    try:
        normalize_log_grid(np.array([0.0, np.nan]))
    except ValueError as exc:
        assert "NaN or Inf" in str(exc)
    else:
        raise AssertionError("NaN likelihood grid was accepted")
