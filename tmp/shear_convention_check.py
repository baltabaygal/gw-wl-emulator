"""
Ground-truth test of the shear addition convention for host + subhalos.

Ground truth uses NO angle convention: the deflection field of a sum of circular NFW
lenses is exact,  alpha(theta) = sum_i kbar_i(|theta-p_i|) * (theta-p_i),  and the shear
is read off the finite-difference Jacobian:
  gamma1 = (dA11 - dA22)/2,  gamma2 = (dA12 + dA21)/2,  kappa = (dA11 + dA22)/2.
Compare per-realization against
  (A) single-angle sum:  gamma += gt_i * (cos phi_i, sin phi_i)     [old code]
  (B) double-angle sum:  gamma += -gt_i * (cos 2phi_i, sin 2phi_i)  [fix]
where phi_i is the ray->lens... (direction of the ray relative to lens i), gt_i = kbar-kappa.
The kappa column doubles as a check that the finite differencing is trustworthy.
"""
import numpy as np

rng = np.random.default_rng(7)

def fg(x):
    """NFW f(x)=kappa/(2 kappa0), g(x): kbar = 4 kappa0 g/x^2 (Wright & Brainerd 2000)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    f = np.empty_like(x); g = np.empty_like(x)
    hi = x > 1; lo = x < 1; eq = ~hi & ~lo
    t = np.empty_like(x)
    t[hi] = np.arctan(np.sqrt((x[hi]-1)/(1+x[hi])))/np.sqrt(x[hi]**2-1)
    t[lo] = np.arctanh(np.sqrt((1-x[lo])/(1+x[lo])))/np.sqrt(1-x[lo]**2)
    f[hi] = (1-2*t[hi])/(x[hi]**2-1); f[lo] = (1-2*t[lo])/(x[lo]**2-1); f[eq] = 1.0/3.0
    g[hi] = 2*t[hi] + np.log(x[hi]/2); g[lo] = 2*t[lo] + np.log(x[lo]/2); g[eq] = 1+np.log(0.5)
    return f, g

def kappa_of(x, k0):   return 2*k0*fg(x)[0]
def kbar_of(x, k0):    return 4*k0*fg(x)[1]/x**2
def gt_of(x, k0):      return kbar_of(x, k0) - kappa_of(x, k0)

def alpha_field(theta, lenses):
    """theta: (2,), lenses: list of (pos(2,), rs, k0). Returns alpha (2,)."""
    a = np.zeros(2)
    for p, rs, k0 in lenses:
        d = theta - p
        x = np.hypot(*d)/rs
        a += kbar_of(x, k0)*d
    return a

def shear_fd(theta, lenses, h=1e-4):
    """kappa, gamma1, gamma2 from the FD Jacobian of alpha — convention-free."""
    ax_p = alpha_field(theta + [h, 0], lenses); ax_m = alpha_field(theta - [h, 0], lenses)
    ay_p = alpha_field(theta + [0, h], lenses); ay_m = alpha_field(theta - [0, h], lenses)
    A11 = (ax_p[0]-ax_m[0])/(2*h); A21 = (ax_p[1]-ax_m[1])/(2*h)
    A12 = (ay_p[0]-ay_m[0])/(2*h); A22 = (ay_p[1]-ay_m[1])/(2*h)
    return 0.5*(A11+A22), 0.5*(A11-A22), 0.5*(A12+A21)

# ---- toy host + clumps, representative scales (kpc) ----
NREAL = 3000
rs_h, k0_h, r200 = 100.0, 0.05, 300.0
rs_c, k0_c = 5.0, 0.02
res = []
for _ in range(NREAL):
    r = np.sqrt(rng.uniform(0, 1))*2.0*rs_h          # ray-host distance, uniform in area
    phi = rng.uniform(0, 2*np.pi)
    ray = r*np.array([np.cos(phi), np.sin(phi)])
    lenses = [(np.zeros(2), rs_h, k0_h)]
    for _ in range(rng.poisson(3.0)):
        R2d = np.sqrt(rng.uniform(0, 1))*r200        # uniform in projected disc (toy)
        psi = rng.uniform(0, 2*np.pi)                # position angle in halo frame: UNIFORM
        lenses.append((R2d*np.array([np.cos(psi), np.sin(psi)]), rs_c, k0_c))

    # ground truth (no angles anywhere)
    kap_fd, g1_fd, g2_fd = shear_fd(ray, lenses)

    # convention sums
    g1_s = g2_s = g1_d = g2_d = kap_sum = 0.0
    for p, rs, k0 in lenses:
        d = ray - p
        xx = np.hypot(*d)/rs
        gt = float(gt_of(xx, k0)[0]); kap_sum += float(kappa_of(xx, k0)[0])
        c, s = d/np.hypot(*d)
        g1_s += c*gt;                g2_s += s*gt                 # (A) single angle
        g1_d += -(c*c - s*s)*gt;     g2_d += -(2*c*s)*gt          # (B) double angle
    res.append((kap_fd, kap_sum, np.hypot(g1_fd, g2_fd),
                np.hypot(g1_s, g2_s), np.hypot(g1_d, g2_d),
                g1_fd, g2_fd, g1_d, g2_d))

res = np.array(res)
kap_fd, kap_sum, g_fd, g_single, g_double = res[:, 0], res[:, 1], res[:, 2], res[:, 3], res[:, 4]
print(f"N = {NREAL} realizations (host + Poisson(3) clumps)\n")
print(f"FD sanity, kappa:      max|kappa_FD - kappa_sum| = {np.abs(kap_fd-kap_sum).max():.2e}")
print(f"DOUBLE angle vs truth: max| |g|_double - |g|_FD | = {np.abs(g_double-g_fd).max():.2e}")
print(f"  (componentwise:      max|g1_d-g1_FD|, max|g2_d-g2_FD| = "
      f"{np.abs(res[:,7]-res[:,5]).max():.2e}, {np.abs(res[:,8]-res[:,6]).max():.2e})")
print(f"SINGLE angle vs truth: max| |g|_single - |g|_FD | = {np.abs(g_single-g_fd).max():.2e}")
print(f"\n<|gamma|^2>:  truth(FD) = {np.mean(g_fd**2):.5e}")
print(f"              double    = {np.mean(g_double**2):.5e}   "
      f"(ratio {np.mean(g_double**2)/np.mean(g_fd**2):.4f})")
print(f"              single    = {np.mean(g_single**2):.5e}   "
      f"(ratio {np.mean(g_single**2)/np.mean(g_fd**2):.4f})")
q = [0.5, 0.9, 0.99]
print(f"\n|gamma| quantiles {q}:")
print(f"   truth : {np.quantile(g_fd, q)}")
print(f"   double: {np.quantile(g_double, q)}")
print(f"   single: {np.quantile(g_single, q)}")
