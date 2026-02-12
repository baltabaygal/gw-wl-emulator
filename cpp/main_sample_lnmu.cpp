#include "lnmu_wrapper.h"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {

    if (argc < 11) {
        std::cerr << "Usage: ...\n";
        return 1;
    }

    CosmologyParams cosmo;
    SamplingParams sampling;

    double z = atof(argv[1]);
    cosmo.OmegaM = atof(argv[2]);
    cosmo.sigma8 = atof(argv[3]);
    cosmo.h      = atof(argv[4]);

    sampling.Nreal = atoi(argv[5]);
    sampling.seed  = atoll(argv[6]);
    sampling.fil   = atoi(argv[7]);
    sampling.bias  = atoi(argv[8]);
    sampling.ell   = atoi(argv[9]);

    std::string out = argv[10];

    auto y = sample_lnmu(z, cosmo, sampling);

    std::ofstream f(out);
    for (double v : y) f << v << "\n";

    std::cerr << "Wrote " << y.size() << " ln(mu) samples\n";
}
