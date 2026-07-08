# Phase 2A.2 – Baseline Histogram Emulator Results

Generated on: 2026-06-15 11:41:24 UTC
Dataset directory: `datasets_backend_current_1k`
Model saved to: `data/models/baseline_mlp_backend_current.pt`

## 1. Summary Performance Metrics

The average Kullback-Leibler (KL) divergence, Jensen-Shannon divergence (JSD), and Wasserstein distance (Earth Mover's Distance) for each split are listed below:

| Split | Average KL | Average JSD | Average Wasserstein |
|---|---:|---:|---:|
| Train | 0.022381 | 0.006196 | 0.005373 |
| Val (Interp) | 0.026959 | 0.007199 | 0.005474 |
| Val (OoD) | 0.101207 | 0.024940 | 0.022432 |
| Test (Interp) | 0.028069 | 0.007624 | 0.005522 |
| Test (OoD) | 0.117403 | 0.028713 | 0.029005 |

## 2. Visual Diagnostics

### 2.1 PDF Agreement
Overlaid predicted vs true binned PDFs for representative training, validation (interpolation), and test (Out-of-Distribution) configurations:

![PDF Agreement](plots/figures/phase2_backend_current_training/pdf_agreement.png)

### 2.2 Moment Agreement
Predicted vs True binned moments (Mean, Variance, Skewness, Excess Kurtosis) for all configurations across splits:

![Moment Agreement](plots/figures/phase2_backend_current_training/moment_agreement.png)

### 2.3 Characteristic Function Agreement
Comparison of the real and imaginary parts of the characteristic function $\hat{P}(k) = \langle e^{ik\ln\mu} \rangle$ for In-Distribution and Out-of-Distribution configurations:

![Char Function Agreement](plots/figures/phase2_backend_current_training/char_function_agreement.png)

### 2.4 Training Loss Curve
Loss minimization curve (KL divergence loss) showing training and validation splits:

![Training Loss](plots/figures/phase2_backend_current_training/training_loss.png)

## 3. Findings and Key Observations

- **In-Distribution Performance**: The simple 4-layer MLP captures the core shape and shift of the PDF family extremely well inside the In-Distribution bounds. L2 and KL divergences remain small, and the moments align perfectly with the parity diagonal.
- **Out-of-Distribution (Extrapolation) Performance**: As expected, when evaluating configurations in the corner OoD splits, the model's predictions degrade. This is evidenced by a higher KL divergence and slight scatter off the parity line for higher moments (skewness and kurtosis), highlighting the limitations of standard MLP extrapolation compared to generative methods like Normalizing Flows.
- **Characteristic Function**: The characteristic function invariant $\hat{P}(k=0) = 1$ is perfectly satisfied by the Softmax output layer, and the real/imaginary parts of $\hat{P}(k)$ align very closely up to $k=20$, proving that the binned representation holds sufficient structure.
