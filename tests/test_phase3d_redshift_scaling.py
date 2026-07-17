import json
from pathlib import Path

def test_phase3d_redshift_scaling_results_exist_and_valid():
    results_path = Path("data/results/phase3d_redshift_scaling_results.json")
    assert results_path.exists(), "Phase 3D redshift scaling results JSON is missing"
    
    with open(results_path, "r") as f:
        data = json.load(f)
        
    for cat_type in ["mixed_uniform", "mixed_low_z_dominated", "mixed_high_z_dominated"]:
        assert cat_type in data
        # Check that sizes are represented as keys
        sizes = list(data[cat_type].keys())
        assert len(sizes) > 0
        for size_str in sizes:
            runs = data[cat_type][size_str]
            assert isinstance(runs, list)
            assert len(runs) > 0
            for run in runs:
                assert run["catalog_type"] == cat_type
