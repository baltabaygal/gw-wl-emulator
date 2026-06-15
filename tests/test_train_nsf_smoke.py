import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import subprocess
import pytest

def test_train_nsf_smoke():
    output_model = "data/models/test_nsf_checkpoint_smoke.pt"
    plots_dir = "plots/figures/phase2b_nsf_training_smoke"
    output_report = "docs/phase2b_nsf_training_report_smoke.md"
    
    # Remove files if they exist
    for f in [output_model, output_report]:
        if os.path.exists(f):
            os.remove(f)
            
    try:
        # Run train_nsf.py using the active conda environment's python
        cmd = [
            sys.executable, "ml/train_nsf.py",
            "--dataset_dir", "datasets_tiny",
            "--epochs", "2",
            "--batch_size", "256",
            "--patience", "2",
            "--output_model", output_model,
            "--plots_dir", plots_dir,
            "--output_report", output_report
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        print(proc.stdout)
        
        # Verify outputs
        assert os.path.exists(output_model)
        assert os.path.exists(output_report)
        assert os.path.exists(os.path.join(plots_dir, "training_loss.png"))
        
    finally:
        # Clean up
        for f in [output_model, output_report]:
            if os.path.exists(f):
                os.remove(f)
        import shutil
        if os.path.exists(plots_dir):
            shutil.rmtree(plots_dir)
