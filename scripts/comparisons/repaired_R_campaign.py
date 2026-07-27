# Repaired-R(xi) campaign: (1) spec-table regression, (2) old-vs-new tail,
# (3) MC tail-band ratios vs repaired R, (4) sigma_DL old/new vs MC.
import sys, numpy as np
sys.path.insert(0,'/Users/baltabay/Desktop/gw-wl-emulator/scripts/comparisons')
sys.path.insert(0,'/Users/baltabay/Desktop/gw-wl-emulator/build')
import analytic_pdf as A, gwlensing as gw
XI, tz = A.XI, A._trapz
SCR="/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/1678dd4f-8cd2-4d07-b736-8995d89aaa73/scratchpad"

print("== 1. spec-table regression at zs=2 (repaired R must reproduce xi<=0.8) ==")
R2 = A.R_of_xi(2.0)
sl = -np.gradient(np.log(R2), np.log(XI))
tab = [(1e-4,8.35e6),(1e-3,7.82e4),(1e-2,618),(0.05,15.1),(0.1,2.47),
       (0.2,0.323),(0.4,2.81e-2),(0.8,1.25e-3)]
for xiv, Rsp in tab:
    j = np.argmin(np.abs(XI-xiv))
    print(f"  xi={xiv:7.0e}  spec {Rsp:9.3e}  new {R2[j]:9.3e}  ratio {R2[j]/Rsp:.3f}  slope {sl[j]:.2f}")

print("== 2. the repaired tail (old legacy vs new deposit) at zs=2 ==")
R2L = A.R_of_xi(2.0, legacy=True)
for xiv in (1.0, 2.0, 4.0, 8.0, 16.0):
    j = np.argmin(np.abs(XI-xiv))
    rl = R2L[j] if R2L[j] > 0 else float('nan')
    print(f"  xi={xiv:5.1f}  legacy {rl:9.3e}  new {R2[j]:9.3e}  new slope {sl[j]:.2f}")

print("== 3. MC tail-band vs REPAIRED R (cached xi arrays) ==")
for zs, R in ((2.0, R2), (5.0, A.R_of_xi(5.0))):
    xi = np.load(f"{SCR}/xi_v1_zs{zs:g}.npz")["xi"]
    body = np.median(xi)
    print(f"  zs={zs:g}")
    for X in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0):
        pmc = np.mean(xi - body > X)
        pana = tz(R[XI >= X], XI[XI >= X])
        print(f"    X={X:4.1f}  P_MC {pmc:.3e}  intR {pana:.3e}  ratio {pmc/pana:5.2f}")

print("== 4. sigma_DL: legacy vs repaired vs MC (source-plane) ==")
MC = {0.5:0.0115, 1.0:0.0248, 2.0:0.0453, 5.0:0.0750, 10.0:0.0924}
print("   zs   legacy   repaired   MC      MC/repaired")
for zs in (0.5, 1.0, 2.0, 5.0, 10.0):
    kthr = gw.get_kappa_threshold(z=zs,h=0.674,OmegaM=0.315,sigma8=0.811,Nhalos=100)
    def sdl(R):
        m = XI >= 2*kthr
        Rs = np.exp(-XI)*R
        def Lt(t): return tz((Rs*(np.exp(-t*XI)-1+t*XI))[m], XI[m])
        return np.sqrt(np.exp(Lt(1.0)-2*Lt(0.5))-1.0)
    Rl = A.R_of_xi(zs, legacy=True); Rn = A.R_of_xi(zs) if zs not in (2.0,) else R2
    sl_, sn = sdl(Rl), sdl(Rn)
    print(f"  {zs:4.1f}  {sl_:.4f}   {sn:.4f}    {MC[zs]:.4f}   {MC[zs]/sn:.3f}")
print("DONE")
