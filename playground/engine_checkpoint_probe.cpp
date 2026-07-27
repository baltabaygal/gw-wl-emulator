// Engine-checkpoint probe (2026-07-24): hold the PRODUCTION C++ cosmology to the
// analytic_chain_spec.md checkpoints DIRECTLY — zero MC. Motivation: the +10%-in-Var
// MC-vs-analytic sigma_DL residual is unexplained; the Python analytic passed every
// spec checkpoint but the ENGINE was never held to one. First miss = the culprit.
//
// Prints "KEY value" lines (engine internal units kpc/Msun converted to the spec's
// Mpc/Msun where noted). Compare side: scripts/comparisons/engine_checkpoint_compare.py.
//
// Build (repo root; same pattern as subhalo_single_host_probe):
//   clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib \
//     playground/engine_checkpoint_probe.cpp cpp/basics.cpp cpp/cosmology.cpp \
//     cpp/lensing.cpp cpp/subhalo.cpp -lgsl -lgslcblas \
//     -o playground/engine_checkpoint_probe
// Run (repo root; uses the dataL/ cosmology cache):
//   ./playground/engine_checkpoint_probe

#include "../cpp/cosmology.h"
#include "../cpp/lensing.h"

#include <cmath>
#include <cstdio>
#include <vector>

double Sigmacf(cosmology &C, double zs, double zl);      // lensing.cpp
array<double,2> FgNFW(double x);                          // lensing.cpp
double kappa0NFW(double rs, double rhos, double Sigmac);  // lensing.cpp

// local log-interpolation of sigmalist rows {M, sigma, dsigma/dM} at exact M
static double sigma_at(cosmology &C, double M) {
    const auto &S = C.sigmalist;
    int n = (int)S.size();
    int j = 0;
    while (j < n-2 && S[j+1][0] < M) j++;
    double lM1 = log(S[j][0]), lM2 = log(S[j+1][0]);
    double ls1 = log(S[j][1]), ls2 = log(S[j+1][1]);
    return exp(ls1 + (ls2-ls1)*(log(M)-lM1)/(lM2-lM1));
}

// engine TOP-HAT sigma(R) from the public Pk0 (kpc^3), R in kpc:
// sigma^2 = (1/2pi^2) int dk k^2 P(k) W_TH^2(kR)
static double sigma_tophat_engine(cosmology &C, double Rkpc) {
    const int N = 4000;
    double lk1 = log(1.0e-8), lk2 = log(1.0);   // kpc^-1  (1e-5..1e3 Mpc^-1)
    double s2 = 0.0, dlk = (lk2-lk1)/N;
    for (int i = 0; i < N; i++) {
        double k = exp(lk1 + (i+0.5)*dlk);
        double x = k*Rkpc;
        double W = 3.0*(sin(x) - x*cos(x))/(x*x*x);
        s2 += k*k*k * C.Pk0(k) * W*W * dlk;     // dk k^2 P = dlnk k^3 P
    }
    return sqrt(s2/(2.0*M_PI*M_PI));
}

static int nearest(const std::vector<double> &v, double x) {
    int j = 0;
    for (int i = 1; i < (int)v.size(); i++)
        if (fabs(log(v[i]/x)) < fabs(log(v[j]/x))) j = i;
    return j;
}

int main() {
    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0;
    C.sigma8 = 0.811; C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    printf("# engine checkpoint probe (fiducial, sigma8-mode)\n");
    printf("sigma8_derived %.6f\n", C.sigma8_derived);
    printf("deltaH8 %.6e\n", C.deltaH8);

    // --- amplitude + window checkpoints ---------------------------------
    double R8 = 8000.0/C.h;                       // 8/h Mpc in kpc
    printf("sigma8_TOPHAT_engine %.6f\n", sigma_tophat_engine(C, R8));
    printf("sigma_1e12_engineWs %.6f\n", sigma_at(C, 1.0e12));
    // top-hat sigma at M=1e12 from engine P(k): R(M) = (3M/4pi rhoM0)^(1/3)
    double rhoM0 = C.OmegaM*277.394*C.h*C.h;      // Msun/kpc^3
    double R12 = pow(3.0e12/(4.0*M_PI*rhoM0), 1.0/3.0);
    printf("sigma_1e12_TOPHAT_engine %.6f\n", sigma_tophat_engine(C, R12));

    // --- growth + collapse threshold ------------------------------------
    printf("Dg_0 %.6f\n", C.Dg(0.0));
    printf("Dg_0.5 %.6f\n", C.Dg(0.5));
    printf("Dg_1 %.6f\n", C.Dg(1.0));
    printf("Dg_2 %.6f\n", C.Dg(2.0));
    printf("deltac_0 %.6f\n", C.deltac(0.0));

    // --- HMF at three grid points (engine units kpc^-3 -> Mpc^-3: x1e9) --
    struct Pt { double M, z; };
    Pt pts[3] = {{1.0e12, 0.01}, {1.0e13, 0.5}, {1.0e14, 1.0}};
    for (auto &p : pts) {
        int jz = nearest(C.zlist, p.z), jM = nearest(C.Mlist, p.M);
        printf("HMF_grid z %.6f M %.6e dndlnM_Mpc3 %.6e\n",
               C.zlist[jz], C.Mlist[jM], C.HMFlist[jz][jM][0]*1.0e9);
    }

    // --- NFW structure at grid host nearest (1e12, z=0.5), source zs=2 ---
    int jz = nearest(C.zlist, 0.5), jM = nearest(C.Mlist, 1.0e12);
    double zl = C.zlist[jz], M = C.Mlist[jM];
    double rs = C.NFWlist[jz][jM][0], rhos = C.NFWlist[jz][jM][1], c200 = C.NFWlist[jz][jM][2];
    double Sc = Sigmacf(C, 2.0, zl);              // Msun/kpc^2
    double k0 = kappa0NFW(rs, rhos, Sc);
    printf("NFW_grid zl %.6f M %.6e\n", zl, M);
    printf("NFW_c200 %.6f\n", c200);
    printf("NFW_rs_kpc %.6f\n", rs);
    printf("NFW_rhos_MsunKpc3 %.6e\n", rhos);
    printf("Sigmac_MsunMpc2 %.6e\n", Sc*1.0e6);   // kpc^-2 -> Mpc^-2
    printf("kappa_s %.6f\n", k0);

    // --- single-halo profile kappa(x), gamma(x) vs Wright-Brainerd ------
    for (double x : {0.1, 1.0, 10.0}) {
        auto Fg = FgNFW(x);
        double kap = 2.0*k0*Fg[0];
        double gam = 2.0*k0*(2.0*Fg[1]/(x*x) - Fg[0]);
        printf("profile x %.3f kappa %.8e gamma %.8e\n", x, kap, gam);
    }
    return 0;
}
