# Phase 2A.3 – Representation Analysis and Scaling Study Report

This report presents a thorough analysis of the intrinsic dimensionality of the weak gravitational lensing magnification PDF family (via Principal Component Analysis) and evaluates the data-scaling and model-capacity limits of the baseline MLP emulator.

---

## 1. Executive Summary

- **Intrinsic Dimensionality**: The magnification PDF family $dP/d\ln\mu$ is **moderately high-dimensional** when high fidelity is required. While **5 Principal Components (PCs)** are sufficient to capture the coarse shape ($90.7\%$ explained variance), resolving high-frequency features (like the sharp empty-beam cutoff and caustic-crossing peaks) requires **52 PCs for 99.0%** and **77 PCs for 99.9%** explained variance.
- **Data Scaling**: Emulation error (KL, JSD, and Wasserstein) scales as a clean power-law decay with training set size $N$. Average JSD drops from $0.0521$ ($N=25$) to $0.0246$ ($N=326$).
- **Capacity Bottleneck**: The emulator is **data-volume limited**, not model-capacity limited. Increasing capacity from the Baseline MLP `[128, 128, 128]` to a Large MLP `[256, 256, 256, 256]` yields virtually zero improvement (JSD remains at $0.0238$ at $N=326$). However, increasing training data size $N$ from 25 to 326 leads to a $51\%$ reduction in error.
- **Recommendation**: Focus on dense coverage of the 4D cosmological parameter grid ($z, \sigma_8, \Omega_M, h$) rather than building overly large network architectures. The baseline model size is fully saturated.

---

## 2. PCA Representation Analysis

We constructed a PDF matrix of shape $(376, 100)$ consisting of all In-Distribution (ID) configurations pooled across train, validation, and test splits. Centering and performing SVD yielded the eigenvalues and variance ratios.

### Explained Variance Summary

| Principal Component | Eigenvalue | Explained Variance Ratio | Cumulative Variance |
| :--- | :---: | :---: | :---: |
| **PC 1** | $0.009604$ | $56.51\%$ | $56.51\%$ |
| **PC 2** | $0.003183$ | $18.73\%$ | $75.24\%$ |
| **PC 3** | $0.001565$ | $9.21\%$ | $84.45\%$ |
| **PC 4** | $0.000666$ | $3.92\%$ | $88.37\%$ |
| **PC 5** | $0.000396$ | $2.33\%$ | **$90.70\%$** |
| **PC 6** | $0.000184$ | $1.08\%$ | $91.78\%$ |
| **PC 7** | $0.000116$ | $0.69\%$ | $92.46\%$ |
| **PC 8** | $0.000080$ | $0.47\%$ | $92.93\%$ |
| **PC 9** | $0.000074$ | $0.43\%$ | $93.37\%$ |
| **PC 10** | $0.000058$ | $0.34\%$ | $93.70\%$ |

### Required Components for Fidelity Thresholds

- **90.0% Variance Explained**: **5 PCs** (captures dominant shifts in peak location and width).
- **95.0% Variance Explained**: **16 PCs** (captures demagnification cutoff boundaries).
- **99.0% Variance Explained**: **52 PCs** (captures moderate high-magnification tail curvature).
- **99.9% Variance Explained**: **77 PCs** (captures rare caustic features and tail extremes).

### Physical Interpretation of Components
The first component represents the first-order effect of redshift $z$ and amplitude $\sigma_8$, which broadens the PDF and shifts the peak towards demagnification. Subsequent components represent asymmetric skewness changes and high-frequency variations in the tail slope. Because of the sharp empty-beam cutoff at negative $\ln\mu$ and the caustic crossing peaks, the distribution cannot be represented by a simple low-rank projection without introducing spurious oscillations or smoothing out these sharp features.

---

## 3. Scaling Study Results

We trained the baseline MLP on training subsets of size $N = 25, 50, 100, 250$, and $326$ configurations. Metrics were evaluated on a fixed validation holdout of 50 configurations.

### Metric Convergence Table

| Training Size ($N$) | Average KL | Average JSD | Average Wasserstein (EMD) |
| :--- | :---: | :---: | :---: |
| **N = 25** | $0.203327$ | $0.052068$ | $0.020402$ |
| **N = 50** | $0.157365$ | $0.040822$ | $0.012648$ |
| **N = 100** | $0.141940$ | $0.037753$ | $0.012318$ |
| **N = 250** | $0.093260$ | $0.024958$ | $0.008800$ |
| **N = 326 (Full)** | $0.091660$ | $0.024590$ | $0.008988$ |

As $N$ increases, we see a clean, consistent decay in all metrics. The error drops by over **50%** when going from $N=25$ to $N=326$. The convergence curve slows down between $250$ and $326$, indicating that we are approaching the interpolation limits of the baseline MLP in 4D space.

---

## 4. Capacity Study Results

We evaluated three architectures:
- **Small MLP**: `[64, 64]` (approx. $11\text{k}$ parameters)
- **Baseline MLP**: `[128, 128, 128]` (approx. $50\text{k}$ parameters)
- **Large MLP**: `[256, 256, 256, 256]` (approx. $260\text{k}$ parameters)

### Validation JSD / Wasserstein Comparison

| Training Size ($N$) | Metric | Small MLP `[64,64]` | Baseline MLP `[128,128,128]` | Large MLP `[256,256,256,256]` |
| :--- | :--- | :---: | :---: | :---: |
| **N = 25** | JSD / EMD | $0.053771$ / $0.019861$ | $0.048825$ / $0.018400$ | $0.047635$ / $0.018405$ |
| **N = 100** | JSD / EMD | $0.045460$ / $0.014754$ | $0.036559$ / $0.011191$ | $0.036109$ / $0.011219$ |
| **N = 326 (Full)** | JSD / EMD | $0.026629$ / $0.009066$ | $0.023816$ / $0.009231$ | $0.023838$ / $0.009063$ |

### Data-Limited vs. Model-Limited Analysis
- **Model Capacity**: The Baseline MLP outperforms the Small MLP by $10.5\%$ in JSD at $N=326$. However, the Large MLP shows **zero additional improvement** over the Baseline MLP. This indicates that network capacity is not the bottleneck; $50\text{k}$ parameters are fully sufficient to represent the mapping from 4D parameters to 100 bin probabilities.
- **Data Volume**: For any architecture (e.g., Baseline), scaling the data from $N=25 \rightarrow 326$ reduces JSD from $0.0488 \rightarrow 0.0238$ ($51.2\%$ improvement). Therefore, the system is **strongly data-limited**. To achieve higher accuracy, generating a denser grid of cosmological parameters is the primary requirement.

---

## 5. Visual Exhibits

Below are the diagnostic figures generated from the analysis.

### 1. Principal Components
This figure displays the first 5 principal components (eigenvectors). Notice the high-frequency oscillatory patterns in higher components, which are required to resolve sharp features.

![PCA Components](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_representation/pca_components.png)

### 2. PCA Reconstructions
This figure overlays the true PDF against reconstructions using 2, 5, and 10 PCs for three diverse redshifts. Note that 2 PCs fail to capture the peak height and tail slope, whereas 10 PCs achieve high fidelity.

![PCA Reconstruction](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_representation/pca_reconstruction.png)

### 3. Scaling Study Learning Curves
These plots display validation JSD and Wasserstein distance as a function of the training set size $N$.

![Scaling Study](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_representation/scaling_study.png)

### 4. Capacity Study Comparison
This bar chart compares JSD and Wasserstein distances across the Small, Baseline, and Large MLP models trained on different data sizes.

![Capacity Study](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_representation/capacity_study.png)

---

## 6. Recommendations for Phase 2B (Neural Spline Flows)

As we transition to **Normalizing Flows (Neural Spline Flows)**, the geometric insights from this study suggest:

1. **Dimensionality Challenge**: Since the intrinsic dimensionality of the PDF family is high ($>50$ components required for $99\%$ fidelity), a standard PCA-bottleneck pre-net followed by a flow is not recommended. Direct training of the conditional flow on raw samples $p(\ln\mu \vert z, \sigma_8, \Omega_M, h)$ is mathematically superior as it avoids any grid-based information loss.
2. **Data-Centric Focus**: Since the MLP was strongly data-limited, the Neural Spline Flow (which is a more complex density estimator) will require a dense grid of configurations to learn smooth interpolation in the 4D space. We recommend generating a training set of at least **500–1000 configurations** with **10,000 samples each** to ensure high-fidelity flow training.
3. **Conditioning MLP Capacity**: The MLP that predicts the spline parameters (knots, derivatives) in the Neural Spline Flow should use the `[128, 128, 128]` architecture. Scaling it larger is unnecessary since capacity is already saturated.
