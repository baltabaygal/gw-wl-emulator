import numpy as np
import scipy.stats as stats

def test_derivative_audit_metrics():
    # Define test mock difference vectors
    dp_sim = np.array([0.1, -0.2, 0.3, 0.0, -0.5])
    dp_emu = np.array([0.09, -0.18, 0.31, 0.01, -0.48]) # close to sim
    
    # Check Cosine Similarity
    dot_prod = np.dot(dp_emu, dp_sim)
    norm_emu = np.linalg.norm(dp_emu)
    norm_sim = np.linalg.norm(dp_sim)
    cosine_sim = dot_prod / (norm_emu * norm_sim)
    
    assert cosine_sim > 0.95, f"Cosine similarity should be high, got {cosine_sim}"
    
    # Check Pearson Correlation
    pearson_corr, _ = stats.pearsonr(dp_emu, dp_sim)
    assert pearson_corr > 0.95, f"Pearson correlation should be high, got {pearson_corr}"
    
    # Check Sign Agreement
    sign_sim = np.sign(dp_sim)
    sign_emu = np.sign(dp_emu)
    
    # Exclude values close to zero in simulator
    val_mask = np.abs(dp_sim) > 1e-5
    sign_agree = np.mean(sign_sim[val_mask] == sign_emu[val_mask])
    
    assert np.isclose(sign_agree, 1.0), f"Sign agreement should be 1.0, got {sign_agree}"
    
    # Test opposite direction
    dp_emu_opp = -dp_emu
    dot_prod_opp = np.dot(dp_emu_opp, dp_sim)
    cosine_sim_opp = dot_prod_opp / (norm_emu * norm_sim)
    assert cosine_sim_opp < -0.95, f"Opposite vectors should give negative similarity, got {cosine_sim_opp}"
    
    pearson_corr_opp, _ = stats.pearsonr(dp_emu_opp, dp_sim)
    assert pearson_corr_opp < -0.95, f"Opposite vectors should give negative correlation, got {pearson_corr_opp}"
    
    sign_agree_opp = np.mean(np.sign(dp_sim)[val_mask] == np.sign(dp_emu_opp)[val_mask])
    assert np.isclose(sign_agree_opp, 0.0), f"Opposite vectors should give 0.0 sign agreement, got {sign_agree_opp}"
