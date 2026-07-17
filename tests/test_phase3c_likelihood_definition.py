import json
from pathlib import Path

def test_phase3c_likelihood_audit_exists_and_valid():
    audit_path = Path("data/results/phase3c_likelihood_definition_audit.json")
    assert audit_path.exists(), "Phase 3C Likelihood Definition Audit JSON is missing"
    
    with open(audit_path, "r") as f:
        payload = json.load(f)
        
    assert payload["phase"] == "3C"
    assert "verdict" in payload
    assert "matrix" in payload
    assert "event_level_formula" in payload["matrix"]
    assert "support_truncation" in payload["matrix"]
    assert "support_floor" in payload["matrix"]
