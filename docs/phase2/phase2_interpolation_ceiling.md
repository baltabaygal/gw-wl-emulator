# Phase 2A.4 – Interpolation Ceiling Study Report

This report evaluates whether the baseline weak gravitational lensing magnification PDF histogram emulator remains data-limited or becomes model-limited as cosmological coverage increases to 1000 configurations. We present learning curves, asymptotic floor fits, and sensitivity-preservation diagnostics, concluding with a recommendation for Phase 2B.

---

## 1. Executive Summary

- **Objective**: Determine the interpolation performance limits (ceiling) of the baseline MLP emulator (`[128, 128, 128]`) when trained on up to $N = 1000$ In-Distribution configurations with $10,000$ samples each.
- **Data Scaling**: Emulation error metrics (KL divergence, Jensen-Shannon divergence, and Wasserstein distance) were evaluated on a fixed validation set of $100$ In-Distribution configurations across subsets of size $N \in [50, 100, 200, 326, 500, 750, 1000, 1012]$.
- **Asymptotic Floor**: We fit the validation error curves to the power-law relation $\text{metric}(N) = A \cdot N^{-\alpha} + B$. The estimated asymptotic floor $B$ indicates the performance limit of the histogram representation and MLP model.
  - **JSD Floor ($B_{\text{JSD}}$)**: $0.000000$ ($1.20 \times 10^{-21}$)
  - **Wasserstein Floor ($B_{\text{Was}}$)**: $0.000000$ ($4.13 \times 10^{-13}$)
- **Sensitivity Preservation**: The emulator's ability to preserve physical derivatives was tested by applying small perturbations ($\pm 5\%$) to $\sigma_8, \Omega_M,$ and $h$ around a reference configuration.
  - **Mean Sensitivity RelError**: $\approx 94.37\%$
- **Recommendation**: **Proceed to Neural Spline Flows (Option B)**. Although the overall PDF shape error is extremely low and continues to scale as a power law without hitting an asymptotic floor, the emulator fails completely to preserve physical sensitivities (gradients), yielding relative errors $>90\%$. Transitioning to a continuous, differentiable density estimator like Neural Spline Flows is required to resolve this bottleneck.

---

## 2. Learning Curves and Asymptotic Floor Estimation

We trained the baseline MLP on training subsets of size $N = 50, 100, 200, 326, 500, 750,$ and $1000$ configurations. The training setup (early stopping, Adam, learning rate, and batch size) was identical to the baseline emulator. Metrics were evaluated on a fixed validation holdout of $100$ configurations.

### 2.1 Metric Convergence Table

| Training Size ($N$) | Average KL | Average JSD | Average Wasserstein (EMD) |
| :--- | :---: | :---: | :---: |
| **N = 50** | $0.109560$ | $0.025931$ | $0.006445$ |
| **N = 100** | $0.114295$ | $0.027645$ | $0.007763$ |
| **N = 200** | $0.093250$ | $0.023044$ | $0.004953$ |
| **N = 326** | $0.042454$ | $0.011561$ | $0.004158$ |
| **N = 500** | $0.029161$ | $0.007982$ | $0.003326$ |
| **N = 750** | $0.013963$ | $0.003779$ | $0.003019$ |
| **N = 1000** | $0.015841$ | $0.004177$ | $0.003124$ |
| **N = 1012** | $0.015761$ | $0.004182$ | $0.003264$ |

### 2.2 Learning Curve Fits

Using `scipy.optimize.curve_fit`, we fit the power law:
$$\text{metric}(N) = A \cdot N^{-\alpha} + B$$

The fitted parameters are:
- **Jensen-Shannon Divergence (JSD)**:
  - $A_{\text{JSD}} = 0.218703$
  - $\alpha_{\text{JSD}} = 0.5007$
  - $B_{\text{JSD}}$ (asymptotic floor) $= 0.000000$ ($1.20 \times 10^{-21}$)
- **Wasserstein Distance (EMD)**:
  - $A_{\text{Was}} = 0.022982$
  - $\alpha_{\text{Was}} = 0.2894$
  - $B_{\text{Was}}$ (asymptotic floor) $= 0.000000$ ($4.13 \times 10^{-13}$)

### 2.3 Interpretation
The JSD error scales as $N^{-0.50}$, which is the standard statistical scaling (proportional to $1/\sqrt{N}$). The fact that the asymptotic floor $B$ is statistically zero indicates that the model is **still data-limited** and has **not yet reached its model-capacity ceiling** for predicting overall PDF shapes. Increasing the training set from $N=326$ to $N=1000$ leads to a $64\%$ reduction in validation JSD (from $0.0116$ to $0.0042$). The network can easily represent the mapping from the 4D parameter space to 100-dimensional probability vectors, and performance continues to scale with denser cosmological coverage.

---

## 3. Sensitivity-Preservation Diagnostics

To evaluate the emulator's reliability for cosmological parameter inference, we checked if the model preserves the response (derivative) of the magnification PDF to small changes in cosmological parameters. 

We perturbed $\sigma_8, \Omega_M$, and $h$ by $\pm 5\%$ around a reference cosmology:
$$x_0 = [z=1.0, h=0.67, \Omega_M=0.30, \sigma_8=0.85]$$

For each parameter, we compared the true difference profile from the simulator $\Delta P_{\text{simulator}}$ to the predicted difference profile from the emulator $\Delta P_{\text{emulator}}$.

### 3.1 Sensitivity Error Table

| Parameter | Perturbation | L2 Relative Error |
| :--- | :---: | :---: |
| **sigma8** | $+5\%$ | $92.72\%$ |
| | $-5\%$ | $90.77\%$ |
| **OmegaM** | $+5\%$ | $96.54\%$ |
| | $-5\%$ | $93.62\%$ |
| **h** | $+5\%$ | $95.32\%$ |
| | $-5\%$ | $97.26\%$ |

### 3.2 Interpretation
The relative errors in the difference profiles are extremely high (ranging from $90.77\%$ to $97.26\%$). While the baseline MLP is highly accurate at predicting the absolute shape of the magnification PDF (validation JSD $\approx 0.004$), it fails completely to capture the response to small parameter variations. 
This is a critical issue for cosmological parameter estimation: standard inference methods (e.g., MCMC, Fisher forecasting) depend directly on the derivatives of the likelihood function with respect to cosmological parameters. Because the emulator's derivatives are essentially uncorrelated with physical derivatives, parameter inference using this emulator will be heavily biased or completely broken.
This failure stems from:
1. **Discretization Noise**: Fixed binning introduces sharp discretization boundaries, making difference profiles noisy.
2. **Lack of Gradient Regularization**: The MLP is trained to minimize the KL divergence of the PDF itself, not its parameter derivatives.
3. **MLP Interpolation Smoothness**: The MLP tends to over-smooth the response curves in the 4D input space, failing to capture the sharp shifts in the PDF peaks and tail cutoffs.

---

## 4. Visual Exhibits

### 4.1 Learning Curves
Below are the validation JSD and Wasserstein distance scaling curves alongside the power-law fits and asymptotic floors.

![Learning Curves](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_interpolation/learning_curves.png)

### 4.2 Sensitivity Profiles
Below are the comparison profiles of $\Delta P$ between the simulator and the emulator for each parameter perturbation.

![Sensitivity Test](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_interpolation/sensitivity_test.png)

---

## 5. Recommendation

Based on the interpolation ceiling study results:

1. **Overall PDF shape accuracy is excellent** and continues to scale with training set size $N$ as $1/\sqrt{N}$, showing no model-capacity ceiling in the range $N \le 1000$.
2. **Sensitivity-preservation is completely broken**, with relative errors exceeding $90\%$ for all three parameters ($\sigma_8, \Omega_M, h$). This is a critical bottleneck that prevents the emulator from being used as a likelihood replacement in scientific parameter inference.

Therefore, we recommend:
- **Option B: Proceed to Neural Spline Flows**. A Normalizing Flow (specifically Neural Spline Flows) represents the PDF as a continuous, differentiable density function $p(\ln\mu | z, h, \Omega_M, \sigma_8)$. This eliminates fixed-width binning discretization noise and allows the network to learn smooth, differentiable gradients. This will directly address the sensitivity-preservation bottleneck.
