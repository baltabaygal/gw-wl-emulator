"""
Option (b): is the screen's +3.9% an under-count from its geometry?

The compact screen places clumps UNIFORMLY over the host's kappa_thr disk (pi rmax^2):
    dV2_compact = barNH/(pi rmax^2) * Int[ dN/dm I2(m) ]
The truth places clumps via the anti-biased projected profile u_sub over r200, with the
LoS impact drawn as P(r)=2r/rmax^2:
    dV2_correct = barNH * <p>_LoS * Int[ dN/dm I2(m) ],   <p>_LoS = Int 2r/rmax^2 u_proj(r) dr
We compute both totals and the variance-weighted ratio.  If correct/compact ~ 3-4 the MC
(+~14%) is right and the geometry was the screen's blind spot; if ~1 the MC over-counts.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from scipy.special import gamma as Gamma, gammaincc

# ---- cosmology (code-matched, same as subhalo_screen.py) ----
Om,sig8,h,ns,Ob=0.315,0.811,0.674,0.965,0.0493
zeq=3402.0; OmR=Om/(1+zeq); OmL=1-Om-OmR
H0=0.000102247*h; CH=306.535; rho_c0=277.394*h**2; rho_m0=Om*rho_c0
dc0=3/5*(3*np.pi/2)**(2/3)
def Az(z): return Om*(1+z)**3+OmR*(1+z)**4+OmL
def Hz(z): return H0*np.sqrt(Az(z))
def OmegaMz(z): return Om*(1+z)**3/Az(z)
def OmegaLz(z): return OmL/Az(z)
def Dg(z):
    o,l=OmegaMz(z),OmegaLz(z); return 2.5*o/(o**(4/7)-l+(1+o/2)*(1+l/70))/(1+z)/0.7869370293916
def deltac(z): return dc0/Dg(z)
def rho_cz(z): return Az(z)*rho_c0
def Dc(z): return quad(lambda zp:CH/Hz(zp),0,z)[0]
def DL(z): return (1+z)*Dc(z)
def Scrit(zs,zl):
    DsA=DL(zs)/(1+zs)**2; DlA=DL(zl)/(1+zl)**2; DlsA=DsA-DlA*(1+zl)/(1+zs)
    return 2.08871e16*DsA/(4*np.pi*DlA*DlsA)
def T_EH98(k):
    th=2.728/2.7; Omh2,Obh2=Om*h*h,Ob*h*h
    s=44.5*np.log(9.83/Omh2)/np.sqrt(1+10*Obh2**0.75)
    ag=1-0.328*np.log(431*Omh2)*Ob/Om+0.38*np.log(22.3*Omh2)*(Ob/Om)**2
    G=Om*h*(ag+(1-ag)/(1+(0.43*k*s*h)**4)); q=k/h*th**2/G
    L0=np.log(2*np.e+1.8*q); C0=14.2+731/(1+62.5*q); return L0/(L0+C0*q*q)
def _s2R(R_Mpc):
    R=R_Mpc/h
    f=lambda lk:(np.exp(lk)**3*(np.exp(lk)**ns*T_EH98(np.exp(lk))**2))/(2*np.pi**2)*(3*(np.sin(np.exp(lk)*R)-np.exp(lk)*R*np.cos(np.exp(lk)*R))/(np.exp(lk)*R)**3)**2
    return quad(f,np.log(1e-4),np.log(1e3),limit=200)[0]
_norm=sig8**2/_s2R(8.0)
_lMg=np.linspace(np.log(1e5),np.log(1e17),200)
_sg=np.array([np.sqrt(_norm*_s2R((3*np.exp(l)/(4*np.pi*rho_m0))**(1/3)/1000*h)) for l in _lMg])
_sp=CubicSpline(_lMg,_sg)
def sig(M): return _sp(np.log(M))
def dsigdM(M): return _sp(np.log(M),1)/M
def pFC(d,S):
    p,q=0.3,0.8; A=1/(1+2.0**(-p)*Gamma(0.5-p)/np.sqrt(np.pi)); nu2=d*d/S
    return A*(1+(q*nu2)**(-p))*np.sqrt(q*nu2/(2*np.pi))*np.exp(-q*nu2/2)/S
def dndlnM(M,z): S=sig(M)**2; return rho_m0*pFC(deltac(z),S)*abs(2*sig(M)*dsigdM(M))
def conc(M,z):
    a=0.520+(0.905-0.520)*np.exp(-0.617*z**1.21); b=-0.101+0.026*z
    return 10.0**(a+b*np.log10(M*h/1e12))
def nfw(M,z):
    c=conc(M,z); r200=(3*M/(4*np.pi*200*rho_cz(z)))**(1/3); rs=r200/c
    rhos=200*rho_cz(z)*c**3/(3*(np.log(1+c)-c/(1+c))); return rs,rhos,c,r200
def Fg0(x):
    x=np.asarray(x,float); o=np.empty_like(x)
    lo,hi,eq=x<1-1e-7,x>1+1e-7,np.abs(x-1)<=1e-7; xl,xh=x[lo],x[hi]
    o[lo]=(1-2*np.arctanh(np.sqrt((1-xl)/(1+xl)))/np.sqrt(1-xl**2))/(xl**2-1)
    o[hi]=(1-2*np.arctan(np.sqrt((xh-1)/(1+xh)))/np.sqrt(xh**2-1))/(xh**2-1); o[eq]=1/3
    return o
J2=quad(lambda x:Fg0(np.array([x]))[0]**2*x,0,1,points=[1])[0]+quad(lambda x:Fg0(np.array([x]))[0]**2*x,1,np.inf)[0]
# SHMF
AL,BE,OM,PR=-0.82,50.0,4.0,1e-4
def growthD(z):
    f=lambda zp:(1+zp)/Az(zp)**1.5; return (np.sqrt(Az(z))*quad(f,z,np.inf)[0])/(quad(f,0,np.inf)[0])
def Dvir(z): d=OmegaMz(z)-1; return 18*np.pi**2+82*d-39*d*d
def zform(M0,z0,f=0.5):
    a=0.815*np.exp(-2*f**3)/f**0.707; w=np.sqrt(2*np.log(a+1))
    rhs=1.686/growthD(z0)+w*np.sqrt(sig(f*M0)**2-sig(M0)**2)
    return brentq(lambda zf:1.686/growthD(zf)-rhs,z0,30)
def Ntau(M0,z0):
    zf=zform(M0,z0)
    return quad(lambda z:6.006*np.sqrt(Dvir(z)/178)/(1+z),z0,zf)[0]
def gam_fs(M0,z0):
    Nt=Ntau(M0,z0); fs=0.3563/Nt**0.6-0.075
    if fs<=0: return 0.0,0.0
    s=(1+AL)/OM; den=Gamma(s)*(gammaincc(s,BE*PR**OM)-gammaincc(s,BE))
    return OM*BE**s/den*fs, fs

# ---- grid ----
Mlist=np.logspace(7,17,100); zlist=np.logspace(np.log10(0.01),np.log10(10.01),100)
dlnM=np.log(Mlist[1]/Mlist[0]); ZS=1.0; kthr=1.275e-4; mfloor=1e7
K2_model=7.30e-4

def rmax_host(M,z,Sc):
    rs,rhos,c,r200=nfw(M,z); k0=rs*rhos/Sc
    if 2*k0*Fg0(np.array([1e-6]))[0]<=kthr: return 0.0,rs,k0,r200,c
    lo,hi=np.log(1e-6),np.log(1e7)
    while hi-lo>0.02:
        m=0.5*(lo+hi)
        if 2*k0*Fg0(np.array([np.exp(m)/rs]))[0]>kthr: lo=m
        else: hi=m
    return np.exp(0.5*(lo+hi)),rs,k0,r200,c

def uproj_avg(rmax,r200,c):
    # projected anti-biased clump density u(r), then <p>=Int 2r/rmax^2 u(r) dr
    # n3d(x) ~ B(x)/(1+c x)^2, x=r/r200 ; B(x)=1/sqrt((x/0.54)^-2.5+1)
    def n3d(x): return (1.0/np.sqrt((x/0.54)**(-2.5)+1.0))/(1.0+c*x)**2 if x>0 else 0.0
    Xg=np.linspace(1e-4,1.0,300)
    up=np.array([2*quad(lambda L:n3d(np.sqrt(X*X+L*L)),0,np.sqrt(max(1-X*X,0)))[0] for X in Xg])  # proj (r200 units)
    # normalize Int u_proj 2 pi R dR = 1  (R physical = X r200)
    R=Xg*r200; nrm=np.trapezoid(up*2*np.pi*R,R); up=up/nrm
    # <p>_LoS = Int_0^min(rmax,r200) (2r/rmax^2) u_proj(r) dr
    rl=np.linspace(0,min(rmax,r200),300); upl=np.interp(rl,R,up)
    return np.trapezoid((2*rl/rmax**2)*upl,rl)

comp=corr=0.0; rows=[]
for jz in range(1,len(zlist)):
    z=zlist[jz]; dz=z-zlist[jz-1]
    if z>=ZS: continue
    Sc=Scrit(ZS,z)
    for jM in range(1,len(Mlist)):
        M=Mlist[jM]
        if M<=10*mfloor: continue
        rmax,rsH,k0H,r200,c=rmax_host(M,z,Sc)
        if rmax<=0: continue
        barNH=CH*np.pi*((1+z)*rmax)**2/Hz(z)*dndlnM(M,z)*dlnM*dz
        if barNH<1e-6: continue
        g,fs=gam_fs(M,z)
        if g<=0: continue
        # clump variance integral V = Int dN/dlnm I2(m) dlnm  over [mfloor,0.1M]
        mm=np.exp(np.linspace(np.log(mfloor),np.log(0.1*M),30))
        dN=g*(mm/M)**AL                      # dN/dlnpsi (exp~1)
        rs_c=np.array([nfw(m,z)[0] for m in mm]); rhos_c=np.array([nfw(m,z)[1] for m in mm])
        k0c=rs_c*rhos_c/Sc
        I2=2*np.pi*rs_c**2*(2*k0c)**2*J2
        V=np.trapezoid(dN*I2,np.log(mm))
        comp+=barNH/(np.pi*rmax**2)*V
        corr+=barNH*uproj_avg(rmax,r200,c)*V
        rows.append((z,M,rmax,r200,rmax/r200))

print("=== option (b): geometry-corrected screen (zs=1, kappa_thr=%.2e) ==="%kthr)
print("  compact screen  Delta<k2> = %.3e  -> %+.2f%% of model"%(comp,100*comp/K2_model))
print("  CORRECT geom    Delta<k2> = %.3e  -> %+.2f%% of model"%(corr,100*corr/K2_model))
print("  ratio correct/compact = %.2f"%(corr/comp))
rr=np.array([r[4] for r in rows]); print("  rmax/r200 over bins: median=%.2f  range[%.2f,%.2f]"%(np.median(rr),rr.min(),rr.max()))
print("  (C++ paired MC saturates ~ +13-14%%)")
