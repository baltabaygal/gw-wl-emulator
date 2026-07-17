"""
Pre-publication invariant checks for the subhalo implementation.

I1. subhalo=OFF is bit-identical to the pre-subhalo baseline (RNG stream untouched):
    compare kappa arrays for subhalo=False vs the same seed. (Self-consistency: two
    calls with subhalo=False must be identical; and kappa_nosub must equal kappa.)
I2. Mass conservation: <Sum m_clump> == f_s_res * M per host bin, measured directly
    from the sampler via the profiling counters is not exposed, so we test the
    analytic identity that the code relies on, independently in numpy.
I3. Flux/normalisation invariant: <1/mu> for the produced P(mu), subhalo on vs off.
I4. Determinism: fixed seed reproduces bit-identical kappa with subhalo on.
"""
import sys, numpy as np
sys.path.insert(0, "build")
import gwlensing as gw

base = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=20000, seed=1234)
raw_base = dict(**base, filaments=False, bias=False, ell=False, Nhalos=100)
ml_base = dict(**base)  # sample_lnmu_ml has no filaments/bias/ell/Nhalos knobs

def raw(**kw):
    r = gw.sample_lensing_raw_ml(**raw_base, **kw)
    return (np.asarray(r["kappa"], float), np.asarray(r["kappa_nosub"], float),
            np.asarray(r["gamma1"], float), np.asarray(r["gamma2"], float))

print("=== I1/I4: determinism + off-path ===")
k_off1, kn_off1, *_ = raw(subhalo=False)
k_off2, kn_off2, *_ = raw(subhalo=False)
print(f"  subhalo=off reproducible:      {np.array_equal(k_off1, k_off2)}")
print(f"  kappa_nosub==kappa when off:   {np.allclose(kn_off1, k_off1, rtol=0, atol=0)}")
k_on1, kn_on1, *_ = raw(subhalo=True, subhalo_model=1, subhalo_factor=1e-5)
k_on2, kn_on2, *_ = raw(subhalo=True, subhalo_model=1, subhalo_factor=1e-5)
print(f"  subhalo=on reproducible:       {np.array_equal(k_on1, k_on2)}")
print(f"  on/off share nosub baseline?   max|kn_on-kn_off| = {np.abs(kn_on1-k_off1).max():.3e}"
      f"  (nonzero expected: on reduces host; nosub tracks FULL host, off IS full host)")
print(f"  var(k_on)-var(kn_on) excess:   {k_on1.var()-kn_on1.var():+.4e}")

print("\n=== I2: mass-conservation identity (analytic, alpha<0 power law) ===")
alpha, psi_max = -0.82, 0.1
def fs_res(g, psi_lo):   # removed fraction (lensing.cpp)
    return g*(psi_max**(1+alpha) - psi_lo**(1+alpha))/(1+alpha)
def Nres(g, psi_lo):     # drawn count (subhalo.cpp)
    return (g/alpha)*(psi_max**alpha - psi_lo**alpha)
def mean_psi(psi_lo):    # <psi> of the sampled clumps (inverse-CDF of psi^(alpha-1))
    num = (psi_max**(1+alpha)-psi_lo**(1+alpha))/(1+alpha)
    den = (psi_max**alpha    -psi_lo**alpha    )/alpha
    return num/den
rng = np.random.default_rng(0)
for g, psi_lo in [(0.5, 1e-3), (1.2, 1e-2), (0.8, 3e-4)]:
    lhs = Nres(g, psi_lo)*mean_psi(psi_lo)   # <Sum psi> = N * <psi>
    rhs = fs_res(g, psi_lo)                  # removed fraction
    # Monte-Carlo the actual sampler: draw N~Poisson(Nres), psi via code's inverse-CDF
    N = rng.poisson(Nres(g, psi_lo), size=200000)
    tot = N.sum()
    u = rng.random(tot)
    pa_lo, pa_hi = psi_lo**alpha, psi_max**alpha
    psi = (pa_lo + u*(pa_hi-pa_lo))**(1/alpha)   # exactly code's log_m mapping
    mc = psi.sum()/200000.0
    print(f"  g={g} psi_lo={psi_lo:.0e}:  <Sum psi> analytic={lhs:.5e}  removed f_s={rhs:.5e}  "
          f"ratio={lhs/rhs:.5f}  MC={mc:.5e} (ratio {mc/rhs:.4f})")

print("\n=== I3: <1/mu> flux invariant (P(mu) from full pipeline) ===")
for label, kw in [("off", dict(subhalo=False)),
                  ("on f=1e-5", dict(subhalo=True, subhalo_model=1, subhalo_factor=1e-5))]:
    out = gw.sample_lnmu_ml(**ml_base, **kw)
    lnmu = np.asarray(out, float)
    inv_mu = np.exp(-lnmu)
    print(f"  {label:12s}: <1/mu>={inv_mu.mean():.5f}  <mu>-ish exp(<lnmu>)={np.exp(lnmu.mean()):.4f}  N={lnmu.size}")
