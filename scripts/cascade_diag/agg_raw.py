"""Aggregate the certified-core σ check: Δσ_bias sign + composition NA at z_s=1,5,
using the κ_tot≤1 core (certified) vs the lnμ-percentile trim (biased proxy)."""
import os, glob, numpy as np
D = ("/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/"
     "ac612839-196b-4778-8b02-eab1b572a0e1/scratchpad/cascade_data/raw")
OUT = "data/results/cascade_residual"
ZS = [1.0, 5.0]; ARMS = ["base", "sub", "bias", "full"]
rows = ["# Certified κ_tot≤1 core: does clustering Δσ flip sign / is the z=5 interaction real?\n",
        "σ_core = std(anchored lnμ | κ≤1 subcritical core); σ_p999 = lnμ-percentile-trim "
        "(biased proxy). 10 seeds × 100k rays. NA = σ_full−(σ_base+Δσ_sub+Δσ_bias).\n",
        "| z_s | metric | σ_base | Δσ_sub | Δσ_bias | SNR_bias | NA | fracNA | SNR_NA |",
        "|----|----|----|----|----|----|----|----|----|"]
for zs in ZS:
    col = {}
    for a in ARMS:
        fs = sorted(glob.glob(os.path.join(D, f"z{zs}_{a}_s*.npz")))
        col[a] = {m: np.array([np.load(f)[m] for f in fs]) for m in ("sig_core", "sig_p999")}
    for m in ("sig_core", "sig_p999"):
        b, su, bi, fu = (col[a][m] for a in ARMS)
        d_sub, d_bias, d_full = su - b, bi - b, fu - b
        NA = fu - (b + d_sub + d_bias)
        rows.append("| %.1f | %s | %.4f | %+.4f | %+.4f | %.1f | %+.5f | %.2f | %.1f |" % (
            zs, m, b.mean(), d_sub.mean(), d_bias.mean(),
            abs(d_bias.mean()) / d_bias.std(ddof=1),
            NA.mean(), abs(NA.mean()) / abs(d_full.mean()) if d_full.mean() else np.nan,
            abs(NA.mean()) / NA.std(ddof=1)))
open(os.path.join(OUT, "tables_kappacore.md"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
