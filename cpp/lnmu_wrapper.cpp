#include "lnmu_wrapper.h"
#include "cosmology.h"
#include "lensing.h"

std::vector<double> sample_lnmu(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp
) {
    cosmology C;

    // cosmology parameters
    C.OmegaM = cp.OmegaM;
    C.sigma8 = cp.sigma8;
    C.h      = cp.h;

    C.OmegaB = cp.OmegaB;
    C.zeq    = cp.zeq;
    C.T0     = cp.T0;
    C.ns     = cp.ns;

    C.Mmin = cp.Mmin;
    C.Mmax = cp.Mmax;
    C.NM   = cp.NM;

    C.zmin = cp.zmin;
    C.zmax = cp.zmax;
    C.Nz   = cp.Nz;

    C.outdir = "dataL";

    int dm = 0;
    C.initialize(dm);

    lensing L;
    rgen mt(sp.seed);

    LensingConfig cfg;
    cfg.Nreal  = sp.Nreal;
    cfg.Nhalos = sp.Nhalos;
    cfg.fil    = sp.fil;
    cfg.bias   = sp.bias;
    cfg.ell    = sp.ell;
    cfg.write  = 0;

    return L.sample_lnmu(C, z, mt, cfg);
}
