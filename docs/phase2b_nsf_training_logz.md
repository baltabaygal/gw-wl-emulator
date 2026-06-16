# Phase 2B — NSF Training Diagnostic Report

Generated on: 2026-06-16 11:15:21 UTC

## 1. Model Configuration
- Flow Type: Rational Quadratic Neural Spline Flow (Zuko NSF)
- Transforms: 6
- Hidden Features: 128 x 2
- Spline Bins: 8
- Input Dimension (lnmu): 1
- Context Dimension (z, h, OmegaM, sigma8): 4

## 2. Preprocessing Statistics
- Context Mean: `[4.579125881195068, 0.6853199601173401, 0.30388331413269043, 0.8655007481575012]`
- Context Std: `[2.7103147506713867, 0.049922533333301544, 0.05992034450173378, 0.11461511999368668]`
- lnmu Mean: `0.013335`
- lnmu Std: `0.243234`

## 3. Training Details
- Optimizer: AdamW
- Batch Size: 16384
- Initial Learning Rate: 0.001
- Weight Decay: 1e-05
- Epochs Completed: 47 (Early stopping patience: 20)
- Total Training Time: 2325.28 seconds
- Best Validation NLL: 0.258972

## 4. Diagnostics Curves
![Training Loss](figures/phase2b_nsf_training/training_loss.png)
