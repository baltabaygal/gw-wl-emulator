import json
from pathlib import Path
from ml.phase3_common import load_npz_with_metadata

def test_phase3c_smoke_retest_outputs_exist():
    result_path = Path("data/results/phase3c_smoke_retest.json")
    report_path = Path("docs/phase3/phase3c_smoke_retest_report.md")
    
    assert result_path.exists(), "Smoke retest JSON is missing"
    assert report_path.exists(), "Smoke retest markdown report is missing"
    
    with open(result_path, "r") as f:
        data = json.load(f)
        
    assert "continuous" in data
    assert "compatible" in data
    assert 0.0 <= data["compatible"]["posterior_jsd"] <= 1.0
    assert 0.0 <= data["compatible"]["total_variation"] <= 1.0

def test_phase3c_smoke_grid_cache_metadata():
    grid_dir = Path("data/results/phase3c_smoke_grids")
    assert grid_dir.exists(), "Smoke grids output directory is missing"
    
    combined_grids = list(grid_dir.glob("combined_2d_mixed_uniform_central_*.npz"))
    assert len(combined_grids) > 0, "No combined grid NPZ found in smoke retest output"
    
    # Check that at least one of them contains the correct metadata
    has_likelihood_mode = False
    for grid_file in combined_grids:
        _, md = load_npz_with_metadata(grid_file)
        if "likelihood_mode" in md:
            has_likelihood_mode = True
            assert md["likelihood_mode"] in ["continuous", "simulator_compatible"]
            
    assert has_likelihood_mode, "Grid metadata is missing 'likelihood_mode'"
