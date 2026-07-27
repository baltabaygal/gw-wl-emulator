
import sys, json, numpy as np
sys.path.insert(0, "build")
import gwlensing as gw
zs, nray, seed, model, factor = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
kw = dict(subhalo=True, subhalo_carve=True, subhalo_model=model, kappa_anchor=1)
if model == 5:
    kw["subhalo_kappathr_factor"] = factor
x = np.asarray(gw.sample_lnmu(zs, 0.315, 0.811, 0.674, nray, seed, **kw))
np.save(sys.argv[6], x)
