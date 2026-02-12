import gwlensing as gw

lnmu = gw.sample_lnmu(
    z=1.0, OmegaM=0.315, sigma8=0.811, h=0.674,
    N=50_000, seed=123,
    halos=True, bias=True, filaments=True
)

stats = gw.lnmu_stats(
    z=1.0, OmegaM=0.315, sigma8=0.811, h=0.674,
    N=50_000, seed=123,
    halos=True, bias=True, filaments=True,
    fast=True
)

print(stats["mean_lnmu"], stats["var_lnmu"], stats["skew_lnmu"], stats["mean_mu"])
