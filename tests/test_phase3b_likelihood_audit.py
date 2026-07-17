import numpy as np

from ml.phase3b_diagnostics import likelihood_audit_payload, validate_grid_arrays


def test_phase3b_likelihood_audit_detects_tail_handling_mismatches():
    payload = likelihood_audit_payload()
    failed = {item["topic"] for item in payload["blocking_mismatches"]}
    assert payload["verdict"] == "fail"
    assert "zero_probability_handling" in failed
    assert "clamping" in failed
    assert payload["checks"]["catalog_hash_matches_grid"]


def test_phase3b_grid_validation_rejects_nonfinite_values():
    checks = validate_grid_arrays(
        {
            "axis_OmegaM": np.array([0.2, 0.3, 0.4]),
            "sim_log_likelihood": np.array([[0.0, -1.0], [-2.0, -3.0]]),
        }
    )
    assert checks["axis_OmegaM"]["strictly_increasing"]
    assert checks["sim_log_likelihood"]["finite"]
    assert np.isclose(checks["sim_log_likelihood"]["posterior_sum"], 1.0)

    try:
        validate_grid_arrays({"nsf_log_likelihood": np.array([0.0, np.inf])})
    except ValueError as exc:
        assert "NaN or Inf" in str(exc)
    else:
        raise AssertionError("Non-finite grid was accepted")
