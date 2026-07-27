# Zero-new-MC decomposition of the ~10% MC-vs-analytic sigma_DL residual.
# Uses: analytic R(xi) [already validated], code readouts of kappa_thr and sigma_W,
# and the earlier measured MC source-plane sigma_DL.
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/1678dd4f-8cd2-4d07-b736-8995d89aaa73/scratchpad')
sys.path.insert(0, '/Users/baltabay/Desktop/gw-wl-emulator/build')
import numpy as np
import analytic_pdf as A
import gwlensing as gw

XI, _trapz = A.XI, A._trapz
MC = {0.5:0.0115, 1.0:0.0248, 2.0:0.0453, 5.0:0.0750, 10.0:0.0924}  # source-plane, measured

def kappa2(R, xi_lo=None, xi_hi=None):
    """Second cumulant of xi: integral of xi^2 R_s(xi) over [xi_lo, xi_hi]."""
    Rs = np.exp(-XI)*R
    integ = XI**2 * Rs
    m = np.ones_like(XI, bool)
    if xi_lo is not None: m &= XI >= xi_lo
    if xi_hi is not None: m &= XI <= xi_hi
    return _trapz(integ[m], XI[m])

print(" zs |  kthr     sigW    | k2_tot  Vsub    4sigW^2 | overcarry | "
      "aDL(1e-7) aDL(kthr) MC     | pred_noComp  comp%")
for zs in (0.5, 1.0, 2.0, 5.0, 10.0):
    kthr = gw.get_kappa_threshold(z=zs, h=0.674, OmegaM=0.315, sigma8=0.811, Nhalos=100)
    sigW = gw.get_sigma_background(z=zs, h=0.674, OmegaM=0.315, sigma8=0.811, kappathr=kthr)
    R = A.R_of_xi(zs)
    k2_tot = kappa2(R)                       # xi_min = 1e-7 (analytic full)
    Vsub   = kappa2(R, xi_hi=2*kthr)         # sub-threshold jump band (xi < 2 kthr)
    k2_above = kappa2(R, xi_lo=2*kthr)       # jumps above threshold
    bg_xi = 4*sigW**2                        # background variance in xi (xi ~ 2 kappa)
    overcarry = bg_xi/Vsub
    # sigma_DL^2 ~ kappa2/4  (D_L ~ e^{-xi/2}); background adds sigW^2 to sigma_DL^2
    aDL_full = 0.5*np.sqrt(k2_tot)           # analytic, xi_min=1e-7
    aDL_thr  = 0.5*np.sqrt(k2_above)         # analytic, jumps above kthr only (no bg)
    pred_noComp = np.sqrt(k2_above/4 + sigW**2)   # matched jumps + code background, scalar
    mc = MC[zs]
    comp = (mc**2 - pred_noComp**2)/mc**2     # composition fraction of sigma_DL^2
    print(f"{zs:4.1f}| {kthr:.2e} {sigW:.4f} | {k2_tot:.4f} {Vsub:.4f} {bg_xi:.4f} |"
          f"  {overcarry:5.2f}   | {aDL_full:.4f}   {aDL_thr:.4f}  {mc:.4f} |"
          f"  {pred_noComp:.4f}     {100*comp:+5.1f}")

print("\n== analytic quadrature convergence (sigma_DL, xi_min=1e-7) ==")
print(" zs   Nz40/NM48  Nz80/NM96   d%")
for zs in (2.0, 5.0, 10.0):
    R1 = A.R_of_xi(zs, Nz=40, NM=48)
    R2 = A.R_of_xi(zs, Nz=80, NM=96)
    s1 = 0.5*np.sqrt(kappa2(R1)); s2 = 0.5*np.sqrt(kappa2(R2))
    print(f"{zs:4.1f}  {s1:.4f}    {s2:.4f}   {100*(s2-s1)/s1:+5.2f}")
