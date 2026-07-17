# TOY pre-A0 estimate for the design_new.md review (Claude, 2026-07-17).
# NOT the A0 deliverable: EH98 no-wiggle P_lin, tophat sigma(M), DM14 c(M,z),
# ST(p=0.3,q=0.8) for BOTH f and b (PBS-consistent target state), Delta=200*rhobar_m0,
# truncated-NFW u~ by direct radial quadrature (numpy-only; sandbox has no scipy).
# Question: S(K) = int k dk P_lin(k,z) K(k)^2 / 2pi for K = W_L (derived 2h kernel)
# vs K = disk(R=8.441 Mpc) and Design-B sharp-R_L tophat; report ratios and R_eff.
import numpy as np, math

Om, h, ns, s8, Ob = 0.315, 0.674, 0.965, 0.811, 0.0493
rhob = 2.775e11 * Om * h**2          # Msun / Mpc^3 comoving
dc = 1.686
p_st, q_st = 0.3, 0.8
A_st = 1.0 / (1.0 + 2**(-p_st) * math.gamma(0.5 - p_st) / math.sqrt(math.pi))

# ---- EH98 zero-baryon transfer, k in 1/Mpc ----
th = 2.725 / 2.7
om_h2 = Om * h**2
s_eh = 44.5 * np.log(9.83 / om_h2) / np.sqrt(1 + 10 * (Ob * h**2)**0.75)
a_g = 1 - 0.328 * np.log(431 * om_h2) * Ob / Om + 0.38 * np.log(22.3 * om_h2) * (Ob / Om)**2
def Tk(k):
    G = Om * h * (a_g + (1 - a_g) / (1 + (0.43 * k * s_eh)**4))
    q = k * th**2 / G
    L = np.log(2 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1 + 62.5 * q)
    return L / (L + C * q * q)

kk = np.logspace(-4, 3, 3000)            # main grid for S integrals
Pk_un = kk**ns * Tk(kk)**2
def sigma_R(R):
    x = kk[None, :] * np.atleast_1d(R)[:, None]
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return np.sqrt(np.trapezoid(Pk_un[None, :] * W**2 * kk[None, :]**3, np.log(kk), axis=1) / (2 * np.pi**2))
snorm = s8 / sigma_R(8.0 / h)[0]
Pk0 = Pk_un * snorm**2

def Dgrow(z):
    def g(zz):
        E2 = Om * (1 + zz)**3 + (1 - Om)
        om = Om * (1 + zz)**3 / E2; ol = (1 - Om) / E2
        return 2.5 * om / (om**(4/7.) - ol + (1 + om/2.) * (1 + ol/70.))
    return g(z) / g(0.0) / (1 + z)

# ---- J1 via Bessel integral (x<=25) + asymptotic ----
_thg = np.linspace(0, np.pi, 400)
def J1(x):
    x = np.atleast_1d(x)
    out = np.empty_like(x)
    lo = x <= 25
    if lo.any():
        out[lo] = np.trapezoid(np.cos(_thg[None, :] - x[lo, None] * np.sin(_thg[None, :])), _thg, axis=1) / np.pi
    if (~lo).any():
        xh = x[~lo]
        out[~lo] = np.sqrt(2 / (np.pi * xh)) * np.cos(xh - 0.75 * np.pi)
    return out
def W_disk(k, R):
    x = np.clip(k * R, 1e-8, None)
    return 2 * J1(x) / x

# ---- population on the code grid 1e7..1e17 ----
m = np.logspace(7, 17, 400)
RL = (3 * m / (4 * np.pi * rhob))**(1/3.)
r200 = (3 * m / (4 * np.pi * 200 * rhob))**(1/3.)
sig0 = sigma_R(RL) * snorm
dlnsdlnm = np.gradient(np.log(sig0), np.log(m))

kker = np.logspace(-4, 3, 320)           # kernel grid, interp onto kk
yy = (np.arange(640) + 0.5) / 640        # radial y=r/r200 grid

def report(z):
    D = Dgrow(z)
    sig = sig0 * D
    nu = dc / sig
    f = A_st * np.sqrt(2 * q_st / np.pi) * nu * (1 + (q_st * nu**2)**(-p_st)) * np.exp(-q_st * nu**2 / 2)
    dndlnm = rhob / m * f * np.abs(dlnsdlnm)
    b = 1 + (q_st * nu**2 - 1) / dc + (2 * p_st / dc) / (1 + (q_st * nu**2)**p_st)
    w = dndlnm * b * m
    Ib = np.trapezoid(w, np.log(m)) / rhob
    Im = np.trapezoid(dndlnm * m, np.log(m)) / rhob
    a_c = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b_c = -0.101 + 0.026 * z
    c = 10**(a_c + b_c * np.log10(m * h / 1e12))
    mu = np.log(1 + c) - c / (1 + c)
    # u~(k|m): int_0^1 dy c^2 y/(1+cy)^2 sinc(k r200 y) / mu(c), chunked over m
    num = np.zeros((len(kker), len(m)))
    for i0 in range(0, len(m), 20):
        sl = slice(i0, i0 + 20)
        zarg = kker[:, None, None] * (r200[sl] * 1.0)[None, :, None] * yy[None, None, :]
        snc = np.where(zarg < 1e-6, 1.0, np.sin(zarg) / np.where(zarg < 1e-6, 1.0, zarg))
        prof = (c[sl][:, None]**2 * yy[None, :]) / (1 + c[sl][:, None] * yy[None, :])**2
        num[:, sl] = np.trapezoid(prof[None, :, :] * snc, yy, axis=2)
    U = num / mu[None, :]
    WLk = np.trapezoid(w[None, :] * U, np.log(m), axis=1) / np.trapezoid(w, np.log(m))
    xL = kker[:, None] * RL[None, :]
    Wth = 3 * (np.sin(xL) - xL * np.cos(xL)) / np.clip(xL, 1e-8, None)**3
    WBk = np.trapezoid(w[None, :] * Wth, np.log(m), axis=1) / np.trapezoid(w, np.log(m))
    WL = np.interp(np.log(kk), np.log(kker), WLk)
    WB = np.interp(np.log(kk), np.log(kker), WBk)
    P = Pk0 * D**2
    def S_of(K): return np.trapezoid(kk**2 * P * K**2, np.log(kk)) / (2 * np.pi)
    def S_dsk(R): return S_of(W_disk(kk, R))
    S_wl, S_b, S_d = S_of(WL), S_of(WB), S_dsk(8.441)
    def reff(S_t):
        lo, hi = np.log(0.2), np.log(80.0)
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            if S_dsk(np.exp(mid)) > S_t: lo = mid
            else: hi = mid
        return np.exp(0.5 * (lo + hi))
    mstar = np.exp(np.trapezoid(w * np.log(m), np.log(m)) / np.trapezoid(w, np.log(m)))
    print(f"z={z:4.2f}  Ib={Ib:.3f} Im={Im:.3f}  exp<lnm>_nbm={mstar:.2e} Msun")
    for kq in (0.1, 0.3, 0.5, 1.0, 3.0):
        i = np.argmin(np.abs(kk - kq))
        print(f"    k={kq:>4}: W_L={WL[i]:6.3f}  W_B={WB[i]:6.3f}  W_disk(8.44)={W_disk(np.array([kq]), 8.441)[0]:6.3f}")
    print(f"    S_WL/S_disk(8.44) = {S_wl/S_d:5.2f}   R_eff(WL) = {reff(S_wl):5.2f} Mpc")
    print(f"    S_B /S_disk(8.44) = {S_b/S_d:5.2f}   R_eff(B)  = {reff(S_b):5.2f} Mpc  [Design B sharp-RL]")

for z in (0.35, 1.0, 2.0):
    report(z)
