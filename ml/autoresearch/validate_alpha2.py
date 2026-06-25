"""Validate the alpha=2 (image-plane) smooth-blend model against a CLEAN, full-tail
10M-sample reference (no mu<12 truncation). Overlay in log-log; the model tail should
sit on the simulator over the whole range.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ml.autoresearch import train_ar
from ml.autoresearch.train_reparam import g_and_logdet

AR = Path(__file__).resolve().parent; REPO = AR.parents[1]
THETA = (0.72, 0.38, 1.00); ZS = [2.0, 3.5, 5.0, 8.0]
NPER, NWORK = 1_000_000, 10
trapz = np.trapezoid

def _worker(args):
    z, seed = args; sys.path.insert(0, str(REPO/"build")); import gwlensing as gw
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, THETA), int(NPER), int(seed), False)
    x = np.asarray(r["lnmu"], float); return x[np.isfinite(x)]

# --- alpha=2 smooth-blend model on the reparam body flow ---
ck = torch.load(AR/"models/flow_reparam.pt", map_location="cpu", weights_only=False)
cfg, stats = ck["config"], ck["stats"]; flow = train_ar.build_flow(cfg); flow.load_state_dict(ck["state_dict"]); flow.eval()
lmean=float(stats["lnmu_mean"]); lstd=float(stats["lnmu_std"]); t0=cfg["t0"]; kk=cfg["k"]
cmean=torch.tensor(stats["context_mean"]); cstd=torch.tensor(stats["context_std"])
def body(lnmu,z,th):
    lnmu=np.asarray(lnmu,np.float32).reshape(-1,1); y=(torch.from_numpy(lnmu)-lmean)/lstd
    v,ld=g_and_logdet(y.squeeze(-1),t0,kk); ctx=torch.tensor([[z,*th]],dtype=torch.float32).repeat(y.shape[0],1)
    with torch.no_grad(): lp=flow((ctx-cmean)/cstd).log_prob(v.unsqueeze(-1))+ld-np.log(lstd)
    return lp.cpu().numpy().ravel()
def model(lnmu,z,th,alpha=2.0,muc=2.8,width=0.35):
    lnmu=np.asarray(lnmu,float).ravel(); lnc=np.log(muc); lp_c=float(body(np.array([lnc]),z,th)[0])
    pl=np.exp(lp_c-(alpha-1)*(lnmu-lnc)); bd=np.exp(body(lnmu,z,th)); w=0.5*(1+np.tanh((lnmu-lnc)/width))
    p=(1-w)*bd+w*pl
    gb=np.linspace(-4,np.log(400),5000); plg=np.exp(lp_c-(alpha-1)*(gb-lnc)); bdg=np.exp(body(gb,z,th)); wg=0.5*(1+np.tanh((gb-lnc)/width))
    return np.log(np.maximum(p/trapz((1-wg)*bdg+wg*plg,gb),1e-300))

def main():
    fig,ax=plt.subplots(2,2,figsize=(12,8),squeeze=False)
    lg=np.linspace(np.log(0.6),np.log(300),500); mug=np.exp(lg)
    for i,z in enumerate(ZS):
        with Pool(NWORK) as p: parts=p.map(_worker,[(z,100+j) for j in range(NWORK)])
        mu=np.exp(np.concatenate(parts)); N=mu.size
        edges=np.geomspace(1.0,300,40); c,_=np.histogram(mu,bins=edges)
        ctr=np.sqrt(edges[:-1]*edges[1:]); bw=np.diff(edges); m=c>=20
        a=ax[i//2][i%2]
        a.scatter(ctr[m],c[m]/(N*bw[m]),s=16,color="#1f77b4",label="simulator (10M, full tail)")
        a.plot(mug,np.exp(model(lg,z,THETA))/mug,color="#2ca02c",lw=2.2,label=r"model (blend + $\mu^{-2}$)")
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-7,5); a.set_xlim(0.6,300)
        a.set_title(f"z={z}  (max μ={mu.max():.0f})"); a.set_xlabel(r"$\mu$"); a.set_ylabel(r"$dP/d\mu$")
        a.grid(True,which="both",ls=":",alpha=.4)
        if i==0: a.legend(fontsize=8)
        print(f"z={z} done, N={N}, maxmu={mu.max():.0f}")
    fig.suptitle(r"$\alpha=2$ model vs CLEAN 10M full-tail simulator — log-log"); fig.tight_layout()
    fig.savefig(AR/"validate_alpha2.png",dpi=140,bbox_inches="tight"); print("wrote",AR/"validate_alpha2.png")

if __name__=="__main__": main()
