# Phase 3D — Posterior Statistics Scaling Study Design

This document details the controlled scaling experiment design to investigate how the Neural Spline Flow (NSF) posterior mismatch scales with statistical support.

## Experiment Sweeps

1. **Event Count Sweep ($N_{\text{events}}$)**:
   - Evaluates statistical support at $N \in \{250, 1000, 5000\}$.
   - Sweep tests if the posterior mismatch shrinks with larger catalog sizes.

2. **Catalog Redshift Composition Types**:
   - **Central Mixed Catalog** (`mixed_uniform`): Discrete redshifts $z \in \{0.5, 1.5, 2.5\}$ with uniform weights $[1/3, 1/3, 1/3]$.
   - **Low-z Dominated Catalog** (`mixed_low_z_dominated`): Discrete redshifts $z \in \{0.5, 1.5, 2.5\}$ with weights $[0.6, 0.3, 0.1]$.
   - **High-z Dominated Catalog** (`mixed_high_z_dominated`): Discrete redshifts $z \in \{0.5, 1.5, 2.5\}$ with weights $[0.1, 0.3, 0.6]$.

3. **Repetitions and Seeds**:
   - Each setting will be repeated **3 times** with independent random seeds (`710001`, `710002`, `710003`) to compute statistical error bars and reduce noise.

4. **Grid Inferences**:
   - Fixed 2D posterior grid of resolution $12 \times 12$.
   - Likelihood evaluation is fixed to `simulator_compatible` mode.

## Deliverables

The scaling study will generate:
- Posterior JSD, TV, overlap, and MAP offset scaling metrics.
- Asymptotic power-law fits ($\text{JSD}(N) = A \cdot N^{-\alpha} + B$) to assess convergence.
- Plot figures showing the scaling curves.
