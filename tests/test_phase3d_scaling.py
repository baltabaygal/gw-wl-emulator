import json
from pathlib import Path

def test_phase3d_scaling_results_exist_and_valid():
    results_path = Path("data/results/phase3d_scaling_results.json")
    assert results_path.exists(), "Phase 3D scaling results JSON is missing"
    
    with open(results_path, "r") as f:
        data = json.load(f)
        
    assert isinstance(data, list)
    assert len(data) > 0
    for run in data:
        assert "catalog_type" in run
        assert "N" in run
        assert "seed" in run
        assert "jsd" in run
        assert "tv" in run
        assert "overlap_68" in run
        assert "overlap_95" in run
        assert "nsf_width" in run
        assert "sim_width" in run
        
        assert 0.0 <= run["jsd"] <= 1.0
        assert 0.0 <= run["tv"] <= 1.0
