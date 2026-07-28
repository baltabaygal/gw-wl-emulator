
import sys, numpy as np
sys.path.insert(0, "build")
import gwlensing as gw
zs, nray, seed, vir = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
kw = dict(subhalo=True, subhalo_model=5, subhalo_carve=True, m_floor=1e7,
          subhalo_kappathr_factor=0.1,
          bias_model=1, bias_window=1, bias_Rperp=20000.0, bias_weak=True,
          fil_bias=True, kappa_anchor=1, kappa_anchor_cut=1.0,
          subhalo_virial=bool(vir))
# NOTE sample_lnmu positional order is (z, OmegaM, sigma8, h) -- h LAST.
x = np.asarray(gw.sample_lnmu(zs, 0.315, 0.811, 0.674, nray, seed, **kw))
np.save(sys.argv[5], x)
