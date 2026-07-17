"""Diagnose + quantify the NSF high-mu tail cutoff.

Hypothesis: the flow's rational-quadratic spline has a finite domain
[-bound, bound] in STANDARDIZED lnmu; beyond it the density is unreliable and
collapses, producing a fixed-mu cutoff independent of z. Quantify the missing
tail probability mass vs the simulator.
"""
import numpy as np
import torch

from ml.phase3_common import import_gwlensing, load_nsf_model
from ml.run_phase3_posterior_grid import load_bin_edges

theta = (0.72, 0.38, 1.00)
zs = [2.0, 3.5, 5.0, 8.0]
model = load_nsf_model("data/models/conditional_nsf_backend_current.pt")
bin_edges = load_bin_edges()
mean = float(model.lnmu_mean); std = float(model.lnmu_std)
trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
cm = model.context_mean.cpu().numpy()
print(f"lnmu_mean={mean:.4f}  lnmu_std={std:.4f}  context_mean(z,h,Om,s8)={np.round(cm,3)} (z-mean confirms model identity)")
print(f"bin_edges=[{bin_edges[0]:.3f},{bin_edges[-1]:.3f}] (mu=[{np.exp(bin_edges[0]):.2f},{np.exp(bin_edges[-1]):.2f}])")
print(f"spline domain (+-5 std in lnmu): lnmu in [{mean-5*std:.2f},{mean+5*std:.2f}] -> mu in [{np.exp(mean-5*std):.2f},{np.exp(mean+5*std):.2f}]")

# Try to read the spline bound from a zuko transform instance
try:
    ctx0 = ((torch.tensor([[2.0, *theta]], dtype=torch.float32) - model.context_mean) / model.context_std)
    tr = model.flow(ctx0)
    print("zuko transform repr:", repr(model.flow.transform)[:200])
except Exception as e:
    print("bound introspection failed:", e)

# Wide lnmu grid extending well beyond the bin range (mu up to ~90)
lnmu = np.linspace(-2.0, 4.5, 4000)
mu = np.exp(lnmu)

def nsf_p_lnmu(z):
    dev = model.context_mean.device
    x = torch.tensor(lnmu, dtype=torch.float32, device=dev).reshape(-1, 1)
    ctx = torch.tensor([[z, *theta]], dtype=torch.float32, device=dev).repeat(len(lnmu), 1)
    with torch.no_grad():
        xn = (x - model.lnmu_mean) / model.lnmu_std
        cn = (ctx - model.context_mean) / model.context_std
        lp = model.flow(cn).log_prob(xn) - torch.log(model.lnmu_std)
    return np.exp(lp.cpu().numpy().ravel())

thr = [1.5, 2.0, 3.0, 4.0, 5.0]
print(f"\n{'z':>5} | {'src':>4} | " + " ".join(f"P(mu>{t})".rjust(10) for t in thr) + " | NSF cutoff mu (norm)")
print("-" * 90)
for z in zs:
    g = import_gwlensing()
    res = g.sample_lnmu_ml_with_diagnostics(float(z), *[float(x) for x in theta], 400000, 100, False)
    s = np.asarray(res["lnmu"], float); s = s[np.isfinite(s)]; smu = np.exp(s)
    sim_tail = [float(np.mean(smu > t)) for t in thr]
    p = nsf_p_lnmu(z)
    # integrate p_lnmu over lnmu (proper normalization measure) for mu>thr
    nsf_tail = [float(trapz(p[lnmu > np.log(t)], lnmu[lnmu > np.log(t)])) for t in thr]
    # support cutoff: largest mu where density still above 1e-4 of peak
    peak = p.max(); above = mu[p > 1e-4 * peak]
    cut = float(above.max()) if above.size else float("nan")
    cut_norm = (np.log(cut) - mean) / std
    print(f"{z:>5} | {'sim':>4} | " + " ".join(f"{v:10.4f}" for v in sim_tail) + " |")
    print(f"{z:>5} | {'nsf':>4} | " + " ".join(f"{v:10.4f}" for v in nsf_tail) + f" | cutoff mu={cut:.2f} (norm lnmu={cut_norm:.2f})")
