# Phase 2A.1 – Histogram Target Design: Lensing Magnification PDF Emulator

This document outlines the target representation design for the Phase 2A baseline Multi-Layer Perceptron (MLP) emulator. We present a statistical analysis of the $\ln\mu$ distribution, evaluate candidate binning strategies, and justify our recommended hybrid target representation.

---

## 1. Executive Summary

- **Objective**: Represent the weak gravitational lensing magnification probability density function (PDF), $dP/d\ln\mu$, as a compact target vector for MLP training.
- **Recommended Scheme**: **Three-Zone Hybrid (Linear-Log) Binning** over the range $[-0.5, 2.5]$ with **100 bins**.
  - **Zone 1 (Negative Tail)**: $[-0.5, -0.1]$ with **15 bins** (linear spacing; width $\Delta \approx 0.0267$).
  - **Zone 2 (Peak)**: $[-0.1, 0.2]$ with **65 bins** (linear spacing; width $\Delta \approx 0.0046$).
  - **Zone 3 (Positive Tail)**: $[0.2, 2.5]$ with **20 bins** (logarithmic spacing; width $\Delta \in [0.0660, 0.1824]$).
- **Target Representation**: A 100-dimensional vector representing the normalized probability densities ($P_i = \text{density}_i$) or log-densities ($\ln P_i$).
- **Justification**: The hybrid scheme achieves a mean L2 reconstruction error ($0.5358$) nearly identical to the optimal quantile-based scheme ($0.5337$), but resolves the high-magnification power-law tail. It yields a bin width ratio of only $39.52$, compared to the extreme ratio of $3936.73$ for the quantile scheme.

---

## 2. Statistical Analysis of the $\ln\mu$ Distribution

We analyzed a total of **14,326 valid samples** generated across 20 distinct cosmological configurations ($z \in [0.5, 9.6]$, $\sigma_8 \in [0.5, 1.4]$, $\Omega_M \in [0.15, 0.47]$, $h \in [0.6, 0.73]$).

### Tabulated Statistics

| Metric | Value | Physical Interpretation |
| :--- | :--- | :--- |
| **Minimum** | $-3.935539$ | Deep demagnification (rare empty-beam configurations) |
| **Maximum** | $6.411273$ | Strong magnification (rare caustic crossings) |
| **Mean** | $0.006153$ | Close to 0 (conservation of flux constraint $\langle \mu \rangle \approx 1$) |
| **Std Dev** | $0.198042$ | Typical width of magnification fluctuation |
| **Skewness** | $5.694618$ | Extremely right-skewed (asymmetric magnification tail) |
| **Excess Kurtosis** | $146.677139$ | Heavily leptokurtic (power-law tail rather than Gaussian) |

### Percentiles

| Percentile | $\ln\mu$ Value | Description |
| :--- | :--- | :--- |
| **0.1%** | $-0.488294$ | Bottom 1/1000 threshold (extreme demagnification) |
| **1.0%** | $-0.345403$ | Typical empty beam cutoff range |
| **10.0%** | $-0.160482$ | Median-left bulk |
| **50.0%** | $-0.011903$ | Median (peak is slightly negative due to skewness) |
| **90.0%** | $0.177175$ | Right bulk transition |
| **99.0%** | $0.629162$ | Top 1% threshold (magnification tail) |
| **99.9%** | $1.704426$ | Top 1/1000 threshold (strong lensing boundary) |

### Tail Behavior

1. **Demagnification Cutoff (Negative Tail)**:
   - There is a strict physical lower bound on magnification $\mu_{\text{min}}$ (representing a completely empty cylinder along the line of sight).
   - For almost all cosmologies, $\mu_{\text{min}} \ge 0.6 \implies \ln\mu_{\text{min}} \ge -0.51$.
   - **99.85%** of all generated samples fall in the range $[-0.5, 2.0]$. Only **0.07%** of samples are below $-0.5$.
2. **Magnification Caustics (Positive Tail)**:
   - Light rays passing close to halos experience strong lensing, producing a heavy power-law tail.
   - Only **0.08%** of samples exceed $2.0$. However, resolving this tail is crucial for standard siren parameter estimation.

---

## 3. Binning Parameterizations Evaluated

We evaluated 6 binning configurations with $N=100$ bins to choose the optimal target representation:

| Scheme | Bin Range | Bin Width Range ($\Delta$) | Width Ratio ($W_{\text{max}}/W_{\text{min}}$) | Mean Empty Bins | L2 Reconstruction Error | Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Fixed-Width (Full)** | $[-3.94, 6.41]$ | $[0.1035, 0.1035]$ | $1.00$ | $87.8\%$ | $0.974874$ | **Rejected** (extremely sparse, loses peak) |
| **Fixed-Width (Trunc)** | $[-0.50, 2.50]$ | $[0.0300, 0.0300]$ | $1.00$ | $68.8\%$ | $0.684243$ | **Rejected** (poor peak resolution for low-$z$) |
| **Quantile (Full)** | $[-3.94, 6.41]$ | $[0.0015, 5.7821]$ | $3936.73$ | $10.2\%$ | **$0.533717$** | **Rejected** (tail is lumped into one giant bin) |
| **Hybrid (Linear-Log)** | $[-0.50, 2.50]$ | $[0.0046, 0.1824]$ | **$39.52$** | $24.1\%$ | **$0.535832$** | **Recommended** (excellent peak & tail resolution) |
| **Hybrid Variant 2** | $[-0.40, 2.20]$ | $[0.0031, 0.2110]$ | $68.82$ | $16.6\%$ | $0.543902$ | **Rejected** (higher error, misses tail bounds) |
| **Hybrid Variant 3** | $[-0.50, 3.00]$ | $[0.0050, 0.2260]$ | $45.20$ | $25.2\%$ | $0.551906$ | **Rejected** (higher error due to wider tail spacing) |

---

## 4. Evaluation and Justification of Recommended Scheme

### 1. Normalization Stability
- Under the Three-Zone Hybrid scheme, the bin boundaries are constant across all cosmologies.
- The probability density in bin $i$ is calculated as $P_i = \frac{k_i}{N_{\text{valid}} \cdot \Delta_i}$ where $k_i$ is the count in bin $i$.
- Integration is perfectly stable: $\sum_{i=1}^{100} P_i \Delta_i = \sum_{i=1}^{100} \frac{k_i}{N_{\text{valid}}} = 1.0$.

### 2. Tail Resolution
- By using logarithmic spacing in Zone 3 ($[0.2, 2.5]$), bin widths grow from $0.066$ to $0.182$.
- This prevents a single giant bin from absorbing the entire tail (as happens in the Quantile scheme, where the last bin width is $5.78$, washing out all structure).

### 3. Resolution of the Peak at Low Redshift
- At low redshift ($z \sim 0.5$), the entire PDF is compressed into $[-0.03, 0.11]$.
- A fixed-width binning over $[-0.5, 2.5]$ has bin widths of $0.03$, meaning the entire PDF would span only 4–5 bins.
- The recommended Hybrid scheme has a peak bin width of **$0.0046$** in $[-0.1, 0.2]$, resolving the peak across **65 bins**, ensuring the sharp feature is captured with high fidelity.

---

## 5. Visual Exhibits

Below are the diagnostic figures generated from the dataset.

### 1. Representative PDFs
This figure shows the shape of the magnification PDF across low, intermediate, and high redshifts. Note the extreme sharpness at low $z$ compared to the wider, more skewed shapes at high $z$.

![Representative PDFs](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_target_design/representative_pdfs.png)

### 2. Recommended Hybrid Binning Overlay
This figure shows the aggregate PDF of the entire dataset overlaid with a subset of the recommended Hybrid bin edges. The bins are densest near the peak to capture detail, and wider in the tails.

![Aggregate PDF and Hybrid Bins](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_target_design/mean_pdf.png)

### 3. Tail Zooms (Negative and Positive)
The left panel zooms in on the sharp demagnification cutoff (negative tail). The right panel shows the power-law tail (positive tail) on a logarithmic scale.

![Tails Zoom](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_target_design/tail_zooms.png)

### 4. Redshift Evolution
This plot highlights how the magnification PDF evolves dynamically as a function of redshift, demonstrating the widening of the distribution and the shifting of the peak towards larger demagnification values.

![Redshift Evolution](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_target_design/redshift_evolution.png)

---

## 6. Recommendation for Phase 2A MLP Emulator

For the baseline MLP density emulator, we recommend:

1. **Network Output**:
   - The MLP should predict a $100$-dimensional vector representing the log-density $\ln P_i$ or logit-probabilities $s_i$ for each of the $100$ hybrid bins.
2. **Activation Layer**:
   - Apply a **Softmax** layer to the raw network output vector $\mathbf{s} \in \mathbb{R}^{100}$ to get bin probabilities $p_i$:
     $$p_i = \frac{e^{s_i}}{\sum_{j=1}^{100} e^{s_j}}$$
   - Convert the probabilities to probability densities $P_i$ by dividing by the bin widths $\Delta_i$:
     $$P_i = \frac{p_i}{\Delta_i}$$
   - This guarantees that $\sum P_i \Delta_i = \sum p_i = 1.0$ is strictly satisfied at the hardware level, ensuring physical consistency (flux conservation and total probability normalization).
3. **Loss Function**:
   - Use the **Kullback-Leibler (KL) Divergence** or the **Multinomial Cross-Entropy** between the target probability distribution $\mathbf{p}^{\text{target}}$ and the predicted probability distribution $\mathbf{p}^{\text{pred}}$:
     $$\mathcal{L} = - \sum_{i=1}^{100} p_i^{\text{target}} \ln p_i^{\text{pred}}$$
   - Minimizing this cross-entropy loss is mathematically equivalent to maximizing the log-likelihood of the samples under the binned density model.
4. **Storage Format**:
   - Save the recommended bin edges (a float array of length 101) in a shared configuration file (e.g. `bin_edges.txt` or inside a config module) so it can be loaded identically by both the dataset preprocessing scripts and the PyTorch model definitions.
