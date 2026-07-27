# DIRECT composition (vector vs scalar) measurement, no C++ hook.
#   MC background-removed:  xi_vec = -ln[(1-(kappa-kappa_weak))^2 - gamma^2]  (vector jumps)
#   analytic xi_min=kthr:   scalar compound-Poisson of the same jump content
# Difference = the (Sum kappa_i)^2 + |Sum gamma_i|^2 cross-terms the scalar chain omits.
import sys
sys.path.insert(0,'/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/1678dd4f-8cd2-4d07-b736-8995d89aaa73/scratchpad')
sys.path.insert(0,'/Users/baltabay/Desktop/gw-wl-emulator/build')
import numpy as np, analytic_pdf as A
import gwlensing as gw

XI, _trapz = A.XI, A._trapz
def aDL(zs, xi_lo):
    R = A.R_of_xi(zs); Rs = np.exp(-XI)*R; m = XI>=xi_lo
    k2 = _trapz((XI**2*Rs)[m], XI[m]); return 0.5*np.sqrt(k2)

def sigDL_srcplane(xi):
    xi = xi[np.isfinite(xi)]
    u = np.exp(-0.5*xi); w = np.exp(-xi); w/=w.sum()
    mean = np.sum(w*u); m2 = np.sum(w*u*u)
    return np.sqrt(max(m2/mean**2-1.0,0.0))

V1 = dict(filaments=False, bias=False, ell=False, subhalo=False, Mmin=1e7)
NREAL=400000
print(" zs | MC full  MC bg-removed | analytic(kthr,scalar) | vec/scalar  comp(σ)%")
for zs in (0.5,1.0,2.0,5.0,10.0):
    d = gw.sample_lensing_raw_ml(z=zs, h=0.674, OmegaM=0.315, sigma8=0.811,
                                 nsamples=NREAL, seed=13, **V1)
    kap = np.asarray(d["kappa"]); kw = np.asarray(d["kappa_weak"])
    g = np.sqrt(np.asarray(d["gamma1"])**2 + np.asarray(d["gamma2"])**2)
    kthr = gw.get_kappa_threshold(z=zs,h=0.674,OmegaM=0.315,sigma8=0.811,Nhalos=100)
    # robust anchor <kappa>=0 over kappa<=1 rays (match kappa_anchor=1)
    def anch(k):
        m=k<=1.0; return k - k[m].mean()
    # full (with background)
    kf = anch(kap); detf=(1-kf)**2-g*g; xif=np.where(detf>1e-12,-np.log(np.abs(detf)),np.nan)
    # background removed
    ke = anch(kap-kw); dete=(1-ke)**2-g*g; xie=np.where(dete>1e-12,-np.log(np.abs(dete)),np.nan)
    sf, se = sigDL_srcplane(xif), sigDL_srcplane(xie)
    a = aDL(zs, xi_lo=kthr)     # analytic scalar, matched threshold
    print(f"{zs:4.1f}| {sf:.4f}   {se:.4f}       |   {a:.4f}              |"
          f"  {se/a:5.3f}    {100*(se**2-a**2)/se**2:+5.1f}")
print("DONE")
