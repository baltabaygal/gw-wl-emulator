#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

#include "cosmology.h"
#include "lensing.h"


int main(int argc, char** argv) {
    // Usage:
    // main_sample_lnmu  z  OmegaM  sigma8  h  Nreal  seed  fil  bias  ell  out.txt
    if (argc < 11) {
        std::cerr << "Usage:\n  "
                  << argv[0]
                  << " z OmegaM sigma8 h Nreal seed fil bias ell out.txt\n";
        return 1;
    }

    double z       = atof(argv[1]);
    double OmegaM  = atof(argv[2]);
    double sigma8  = atof(argv[3]);
    double h       = atof(argv[4]);
    int Nreal      = atoi(argv[5]);
    uint64_t seed  = (uint64_t) atoll(argv[6]);
    int fil        = atoi(argv[7]);
    int bias       = atoi(argv[8]);
    int ell        = atoi(argv[9]);
    std::string out = argv[10];

    cosmology C;
    // Keep other defaults as in main_lensing.cpp (you can expose more later)
    C.OmegaM = OmegaM;
    C.sigma8 = sigma8;
    C.h      = h;

    // Important defaults required by C.initialize()
    // Mirror the defaults used in main_lensing.cpp so initialize() has valid ranges
    C.OmegaB = 0.0493;
    C.zeq = 3402.0;
    C.T0 = 2.7255;
    C.ns = 0.965;

    // mass and redshift grids
    C.Mmin = 1.0e7;
    C.Mmax = 1.0e17;
    C.NM = 100;
    C.zmin = 0.01;
    C.zmax = 10.01;
    C.Nz = 100;

    // output directory
    C.outdir = "dataL";


    int dm = 0; // CDM for now
    C.initialize(dm);

    lensing L;
    rgen mt(seed);

    LensingConfig cfg;
    cfg.Nreal = Nreal;
    cfg.Nhalos = L.Nhalos;  // keep class default unless you want CLI arg
    cfg.fil = fil;
    cfg.bias = bias;
    cfg.ell = ell;
    cfg.write = 0;

    std::vector<double> y = L.sample_lnmu(C, z, mt, cfg);

    std::ofstream f(out);
    for (double v : y) f << v << "\n";
    f.close();

    std::cerr << "Wrote " << y.size() << " ln(mu) samples to " << out << "\n";
    return 0;
}
