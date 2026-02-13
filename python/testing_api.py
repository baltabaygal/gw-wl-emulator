import gwlensing as gw

lnmu = gw.sample_lnmu(
    z=1.0,
    OmegaM=0.315,
    sigma8=0.811,
    h=0.674,
    Nreal=50000,
    seed=123,
    filaments=True,
    bias=True,
    ell=True,
    Nhalos=100,   # halos "on" via nonzero count
)

print("lnmu:", lnmu.shape, lnmu[:5])

stats = gw.compute_lnmu_stats(
    z=1.0,
    OmegaM=0.315,
    sigma8=0.811,
    h=0.674,
    Nreal=50000,
    seed=123,
    filaments=True,
    bias=True,
    ell=True,
    Nhalos=100,
    fast=True
)

print("stats:", stats)
