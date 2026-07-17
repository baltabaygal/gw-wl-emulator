import json
from pathlib import Path

def test_phase3c_single_z_retest_outputs_exist():
    result_path = Path("data/results/phase3c_single_z_retest.json")
    report_path = Path("docs/phase3/phase3c_single_z_retest.md")
    
    assert result_path.exists(), "Single-z retest JSON is missing"
    assert report_path.exists(), "Single-z retest markdown report is missing"
    
    with open(result_path, "r") as f:
        data = json.load(f)
        
    for z_key in ["z=0.5", "z=1.5", "z=2.5"]:
        assert z_key in data
        assert "continuous" in data[z_key]
        assert "compatible" in data[z_key]
        assert 0.0 <= data[z_key]["compatible"]["posterior_jsd"] <= 1.0
