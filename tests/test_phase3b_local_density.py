import json

from ml.phase3b_diagnostics import write_local_density_outputs


def test_phase3b_local_density_residual_outputs_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = {
        "theta_points": {
            "truth": {
                "sum_residual_nsf_minus_sim": 1.0,
                "mean_residual_nsf_minus_sim": 0.1,
                "positive_residual_fraction": 0.6,
                "by_z": {
                    "0.50": {"sum_residual_nsf_minus_sim": 0.4},
                    "1.50": {"sum_residual_nsf_minus_sim": 0.3},
                },
            }
        }
    }
    write_local_density_outputs(payload)
    result_path = tmp_path / "data/results/phase3b_local_density_check.json"
    # ml.phase3b_diagnostics.DOCS_DIR is Path("docs"), not "docs/phase3" -- this
    # expectation had been stale for a long time (the "pre-existing unrelated
    # failure" noted in CLAUDE.md). Fixed 2026-07-29; nothing to do with physics.
    report_path = tmp_path / "docs/phase3b_local_density_check.md"
    plot_path = tmp_path / "plots/figures/phase3b_local_density_check/residual_sum_by_z.png"
    assert result_path.exists()
    assert report_path.exists()
    assert plot_path.exists()
    assert json.loads(result_path.read_text())["theta_points"]["truth"]["positive_residual_fraction"] == 0.6
