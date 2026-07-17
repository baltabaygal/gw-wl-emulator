from ml.posterior_metrics import coverage_summary


def test_phase3_coverage_returns_valid_fractions():
    records = [
        {"truth_coverage": {"nsf_68": {"h": True}, "nsf_95": {"h": True}}},
        {"truth_coverage": {"nsf_68": {"h": False}, "nsf_95": {"h": True}}},
    ]
    summary = coverage_summary(records)
    assert summary["n_catalogs"] == 2
    assert 0.0 <= summary["parameters"]["h"]["coverage_68"] <= 1.0
    assert 0.0 <= summary["parameters"]["h"]["coverage_95"] <= 1.0
