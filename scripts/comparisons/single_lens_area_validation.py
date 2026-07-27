# Independent validation of dsigma/dxi for ONE lens, using neither method:
# brute-force area{xi > X} by direct annulus summation on a 4e6-point grid.
# Also splits outer branch (x > tangential critical radius) vs inner branch,
# to quantify exactly what the branch-merging interp drops.
import sys, numpy as np
sys.path.insert(0,'/Users/baltabay/Desktop/gw-wl-emulator/scripts/comparisons')
import analytic_pdf as A
XI, tz = A.XI, A._trapz
pi = np.pi

def brute(M, zl, zs, Xs, N=4_000_000):
    C, rs, ks, fC = A.nfw_params(M, zl, zs)
    x = np.logspace(-9, 3.5, N)
    k, g = A.kappa_gamma(x, ks)
    det = (1-k)**2 - g*g
    xi = np.where(det > 0, -np.log(np.abs(np.where(det > 0, det, 1))), np.nan)
    # annulus areas (trapezoid in x^2), units rs^2
    x2 = x*x
    dA = pi*np.diff(x2)
    xm = 0.5*(xi[1:] + xi[:-1])
    # branch split: tangential critical radius = largest x with det<=0
    neg = np.nonzero(det <= 0)[0]
    xcrit = x[neg[-1]] if neg.size else 0.0
    outer = x[1:] > xcrit
    out = {}
    for X in Xs:
        sel = np.isfinite(xm) & (xm > X)
        out[X] = (rs*rs*np.sum(dA[sel]),                 # total
                  rs*rs*np.sum(dA[sel & outer]),         # outer branch
                  rs*rs*np.sum(dA[sel & ~outer]))        # inner branch
    return out, ks, xcrit, rs

def integ(dsig_per_XI, X):
    m = XI >= X
    return tz(dsig_per_XI[m], XI[m])

print("Single-lens cross-section area{xi>X} [Mpc^2], zl=0.5 zs=2")
for M in (1e12, 1e14, 1e15):
    Xs = [0.1, 0.5, 1.0, 2.0]
    bf, ks, xcrit, rs = brute(M, 0.5, 2.0, Xs)
    dep = A.dsigma_bins(M, 0.5, 2.0)
    xl, dl = A.dsigma_dxi_curve(M, 0.5, 2.0)
    o = np.argsort(xl)
    lnds = np.interp(np.log(XI), np.log(xl[o]), np.log(dl[o]),
                     left=-np.inf, right=-np.inf)
    leg = np.where(np.isfinite(lnds), np.exp(lnds), 0.0)
    print(f"\n M={M:.0e}  kappa_s={ks:.4f}  x_crit={xcrit:.4g}  (Einstein radius exists: {xcrit>0})")
    print("    X    brute_total   brute_outer  brute_inner |   deposit    dep/brute |   legacy   leg/brute")
    for X in Xs:
        tot, ou, inn = bf[X]
        d, l = integ(dep, X), integ(leg, X)
        print(f"  {X:4.1f}  {tot:.4e}  {ou:.4e}  {inn:.4e} | {d:.4e}  {d/tot:6.3f}  | {l:.4e} {l/tot:6.3f}")
print("\nDONE")
