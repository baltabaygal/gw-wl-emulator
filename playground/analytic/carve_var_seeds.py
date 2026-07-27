#!/usr/bin/env python
# multi-seed variance check: is the carve's Var reduction real, and does it match carved brute?
import numpy as np, importlib.util, sys
spec=importlib.util.spec_from_file_location("m","/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/97c72c79-5ba7-445c-b872-554425410874/scratchpad/single_host_kappa_pdf.py")
# we re-run the core loop here to control the seed & keep it light (no plotting/JSD).
import numpy as np
from scipy import integrate
Om,OL,h=0.315,0.685,0.674; RHOC0=277.394*h*h; CKMS=2.998e5; G_KPC=4.30091e-6
C2_4piG=CKMS**2/(4*np.pi*G_KPC)
def Ez(z): return np.sqrt(Om*(1+z)**3+OL)
def Dc(z): return (CKMS/(100*h))*1e3*integrate.quad(lambda zz:1/Ez(zz),0,z)[0]
ZL,ZS=0.5,1.0; DcL,DcS=Dc(ZL),Dc(ZS); DAL,DAS=DcL/(1+ZL),DcS/(1+ZS); DALS=(DcS-DcL)/(1+ZS)
SIGMA_C=C2_4piG*DAS/(DAL*DALS); rhoc_zl=RHOC0*Ez(ZL)**2; CH=6.0; M_HOST=1e13
def nfw(M,c=CH):
    r200=(3*M/(4*np.pi*200*rhoc_zl))**(1/3.); rs=r200/c; mc=np.log(1+c)-c/(1+c)
    return rs,(200/3.)*rhoc_zl*c**3/mc,r200
def Fg(x):
    x=np.asarray(x,float);o=np.empty_like(x);lo=x<1-1e-6;hi=x>1+1e-6;mid=~(lo|hi)
    xl,xh=x[lo],x[hi]
    o[lo]=(1-2/np.sqrt(1-xl**2)*np.arctanh(np.sqrt((1-xl)/(1+xl))))/(xl**2-1)
    o[hi]=(1-2/np.sqrt(xh**2-1)*np.arctan(np.sqrt((xh-1)/(xh+1))))/(xh**2-1); o[mid]=1/3.
    return o
def kappa_nfw(M,R,c=CH):
    rs,rhos,_=nfw(M,c); return 2*(rs*rhos/SIGMA_C)*Fg(np.maximum(R/rs,1e-6))
_,_,R200=nfw(M_HOST)
ALPHA,BETA,OMEGA=-0.82,50.,4.; PSI_MAX,PSI_MIN=1.,1e7/M_HOST; PSI_RES=1e-3; FB=0.14
def dN(p): return p**(ALPHA-1)*np.exp(-BETA*p**OMEGA)
GAMMA=FB/integrate.quad(lambda p:p*dN(p),PSI_MIN,PSI_MAX)[0]
def band(a,b): return (integrate.quad(lambda p:GAMMA*dN(p),a,b)[0],integrate.quad(lambda p:GAMMA*p*dN(p),a,b)[0])
N_res,f_res=band(PSI_RES,PSI_MAX); N_unr,f_unr=band(PSI_MIN,PSI_RES); _,f_b=band(PSI_MIN,PSI_MAX)
M_U_MEAN=f_unr*M_HOST; w_u=f_unr/(1-f_res)
def cdf(a,b,n=4000):
    pg=np.logspace(np.log10(a),np.log10(b),n);w=dN(pg)
    c=np.concatenate([[0],np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(pg))]);c/=c[-1];return pg,c
pg_res,cdf_res=cdf(PSI_RES,PSI_MAX); pg_unr,cdf_unr=cdf(PSI_MIN,PSI_RES)

def run(seed,NRAY=40000):
    rng=np.random.default_rng(seed)
    xs=np.linspace(1e-4,1,4000); mcum=np.log(1+CH*xs)-CH*xs/(1+CH*xs); mcum/=mcum[-1]
    def sep(n,r):
        x=np.interp(rng.random(n),mcum,xs); r3=x*R200
        cth=rng.uniform(-1,1,n); az=rng.uniform(0,2*np.pi,n); R2=r3*np.sqrt(1-cth**2)
        return np.sqrt((r-R2*np.cos(az))**2+(R2*np.sin(az))**2)
    # kappa_U moments on r-grid (single-clump MC, same rng-independent pool)
    RG=np.linspace(0.05*R200,1.2*R200,24); NS=120000
    muU=np.zeros(24); s2U=np.zeros(24); cov=np.zeros(24); vM=np.zeros(24); mM=np.zeros(24)
    for i,r in enumerate(RG):
        m=np.interp(rng.random(NS),cdf_unr,pg_unr)*M_HOST; d=sep(NS,r); kc=kappa_nfw(m,d)
        muU[i]=N_unr*kc.mean(); s2U[i]=N_unr*(kc**2).mean(); cov[i]=N_unr*(kc*m).mean()
        vM[i]=N_unr*(m**2).mean(); mM[i]=N_unr*m.mean()
    r_ray=R200*np.sqrt(rng.uniform(0.05**2,1.,NRAY))
    k0=np.empty(NRAY);kA=np.empty(NRAY);kB=np.empty(NRAY);kC=np.empty(NRAY);kD=np.empty(NRAY)
    kBr=np.empty(NRAY);kBrF=np.empty(NRAY)
    negrem=0
    for j in range(NRAY):
        r=r_ray[j]; nR=rng.poisson(N_res)
        if nR:
            mR=np.interp(rng.random(nR),cdf_res,pg_res)*M_HOST; kR=kappa_nfw(mR,sep(nR,r)).sum(); sR=mR.sum()
        else: kR=0.;sR=0.
        d=sR-f_res*M_HOST
        mu=np.interp(r,RG,muU);s2=np.interp(r,RG,s2U);sU=np.sqrt(max(s2,0))
        cv=np.interp(r,RG,cov);vm=np.interp(r,RG,vM);mm=np.interp(r,RG,mM)
        sl=cv/vm; cs=np.sqrt(max(s2-sl*cv,0)); g=rng.standard_normal()
        k0[j]=kappa_nfw((1-f_b)*M_HOST,r)+kR+(mu+sU*g)
        kA[j]=kappa_nfw(max(M_HOST-sR-M_U_MEAN,1.),r)+kR+(mu+sU*g)
        # (B) supervisor scheme: host FIXED at (1-f_b)M; kappa_u conditioned on its mass
        # being exactly the remainder M_rem = f_b M - sum_res (fs never varies).
        Mrem=f_b*M_HOST-sR
        if Mrem<0: negrem+=1
        kB[j]=kappa_nfw((1-f_b)*M_HOST,r)+kR+(mu+sl*(Mrem-mm)+cs*g)
        MUt=M_U_MEAN-w_u*d
        kC[j]=kappa_nfw(max(M_HOST-sR-MUt,1.),r)+kR+(mu+sl*(MUt-mm)+cs*g)
        # (D) sample the unresolved band's OWN total mass, carve host by it, condition kappa_U
        MUd=mm+np.sqrt(max(vm,0.0))*rng.standard_normal()  # Var(M_U)=N_unr E[m^2]=vm (compound Poisson)
        MUd=max(MUd,1.0)
        kD[j]=kappa_nfw(max(M_HOST-sR-MUd,1.),r)+kR+(mu+sl*(MUd-mm)+cs*g)
        nU=rng.poisson(N_unr)
        if nU:
            mU=np.interp(rng.random(nU),cdf_unr,pg_unr)*M_HOST; kUb=kappa_nfw(mU,sep(nU,r)).sum(); sU2=mU.sum()
        else: kUb=0.;sU2=0.
        kBr[j]=kappa_nfw(max(M_HOST-sR-sU2,1.),r)+kR+kUb           # carved brute
        kBrF[j]=kappa_nfw((1-f_b)*M_HOST,r)+kR+kUb                 # fixed-host (uncarved) brute
    return k0.var(),kA.var(),kB.var(),kC.var(),kD.var(),kBr.var(),kBrF.var(),negrem/NRAY

NSEED=12
out=[run(s) for s in range(NSEED)]
res=np.array([o[:7] for o in out]); negfrac=np.mean([o[7] for o in out])
lab=['current','carve A','cond B ','carve C','carve D','bruteCarved','bruteFixed']
BR=5   # carved-brute column index
print(f"Var(kappa) x1e4, {NSEED} seeds (mean +/- sem):")
for i,l in enumerate(lab):
    print(f"  {l:11s} {res[:,i].mean()*1e4:.4f} +/- {res[:,i].std(ddof=1)/np.sqrt(NSEED)*1e4:.4f}")
print(f"\nB: fraction of draws with negative remainder mass = {negfrac*100:.1f}%")
print("\nPaired Var difference vs carved brute (x1e4, mean +/- sem, %):")
for i,l in zip([0,1,2,3,4,6],['current    ','carveA     ','condB      ','carveC     ','carveD     ','bruteFixed ']):
    d=(res[:,i]-res[:,BR])*1e4
    print(f"  {l} - bruteCarved : {d.mean():+.4f} +/- {d.std(ddof=1)/np.sqrt(NSEED):.4f}  ({d.mean()/(res[:,BR].mean()*1e4)*100:+.2f}%)")
