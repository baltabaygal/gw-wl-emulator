import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
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

print("\nTesting ML wrapper...")
lnmu_ml = gw.sample_lnmu_ml(
    z=1.0,
    h=0.674,
    OmegaM=0.315,
    sigma8=0.811,
    nsamples=50000,
    seed=123
)

print("lnmu_ml:", lnmu_ml.shape, lnmu_ml[:5])
