// Sandbox verification harness for the mass-conserving host carve (scheme A).
// Built with the GSL shim (/tmp/gslshim) -> NOT bitwise-comparable to the Mac build,
// but self-consistent within the sandbox (all arms use the same shim), which is exactly
// what the bitwise carve-off / mass-conservation gates need. Authoritative gates run on
// the Mac in the `test` conda env (make build + pytest).
#include "cosmology.h"
#include "lensing.h"
#include "subhalo.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <functional>
#include <algorithm>

double NhfNFW(cosmology &C, double zs, double kappathr);
double findkappathr(int N, std::function<double(double)> Nf);
double Sigmacf(cosmology &C, double zs, double zl);

static cosmology make_cosmo() {
    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0; C.sigma8 = 0.811;
    C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1e7; C.Mmax = 1e17; C.NM = 45; C.zmin = 0.01; C.zmax = 10.01; C.Nz = 45;
    C.outdir = "dataL"; C.initialize(0);
    return C;
}

static void run(cosmology &C, double zs, uint64_t seed, bool carve,
                std::vector<double> &kappa, std::vector<double> &knosub,
                LensingProfile *prof = nullptr) {
    lensing L;
    rgen mt(seed);
    LensingConfig cfg;
    cfg.Nreal = 40000;
    cfg.subhalo = true;
    cfg.subhalo_model = 3;
    cfg.subhalo_factor = 1e-2;
    cfg.subhalo_carve = carve;
    cfg.profile = prof;
    auto raw = L.sample_lnmu_raw(C, zs, mt, cfg);
    kappa.resize(raw.size()); knosub.resize(raw.size());
    for (size_t i = 0; i < raw.size(); i++) { kappa[i] = raw[i].kappa; knosub[i] = raw[i].kappa_nosub; }
}

static void stats(const std::vector<double> &v, double &mean, double &var) {
    long double s = 0; for (double x : v) s += x; mean = (double)(s / v.size());
    long double s2 = 0; for (double x : v) { long double d = x - mean; s2 += d * d; }
    var = (double)(s2 / v.size());
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered so partial output survives a crash
    fprintf(stderr, "start\n");
    cosmology C = make_cosmo();
    fprintf(stderr, "cosmo built\n");
    double zs = 1.0; uint64_t seed = 12345;

    printf("======== SAMPLER-LEVEL (model 3, Nreal=300k, zs=1) ========\n");
    std::vector<double> k_off, kn_off, k_on, kn_on;
    LensingProfile prof; prof.reset(C.NM);
    run(C, zs, seed, false, k_off, kn_off);
    run(C, zs, seed, true,  k_on,  kn_on, &prof);
    { FILE*f=fopen("/tmp/k_off.bin","wb"); fwrite(k_off.data(),sizeof(double),k_off.size(),f); fclose(f);
      fprintf(stderr,"dumped k_off %zu\n", k_off.size()); }

    size_t ndiff = 0; double maxnd = 0;
    for (size_t i = 0; i < kn_off.size(); i++) { double d = fabs(kn_off[i] - kn_on[i]); if (d > 0) { ndiff++; if (d > maxnd) maxnd = d; } }
    printf("[GATE] kappa_nosub identical off-vs-on : %zu/%zu differ (maxabs %.2e)  %s\n",
           ndiff, kn_off.size(), maxnd, ndiff == 0 ? "PASS" : "FAIL");

    double m_off, v_off, m_on, v_on; stats(k_off, m_off, v_off); stats(k_on, m_on, v_on);
    printf("mean kappa  off=%.8e  on=%.8e  frac shift=%+.3e (Jensen, expect small)\n",
           m_off, m_on, (m_on - m_off) / fabs(m_off));
    printf("var  kappa  off=%.8e  on=%.8e  ratio on/off=%.6f (expect <1: anti-corr)\n",
           v_off, v_on, v_on / v_off);
    printf("[GATE] carve_negatives (guard fires)   : %llu  %s\n",
           (unsigned long long)prof.subhalo_carve_negatives,
           prof.subhalo_carve_negatives == 0 ? "PASS" : "WARN");

    double sdelta = 0, maxabs = 0; size_t nnz = 0;
    for (size_t i = 0; i < k_off.size(); i++) { double d = k_on[i] - k_off[i]; sdelta += d; if (fabs(d) > 1e-14) nnz++; if (fabs(d) > maxabs) maxabs = fabs(d); }
    printf("delta(on-off): mean=%+.3e  nonzero(substructure rays)=%zu  maxabs=%.3e\n",
           sdelta / k_off.size(), nnz, maxabs);

    printf("\n======== SINGLE-HOST MASS BOOKKEEPING (direct) ========\n");
    double kappathr = findkappathr(100, [&](double k){ return NhfNFW(C, zs, k); });
    printf("host kappathr(zs=1, N=100) = %.4e\n", kappathr);
    Subhalo sh; sh.m_floor = 1e7;
    sh.precompute(C, zs, 1e-2 * kappathr, kappathr);
    // pick an active bin near M~1e13, z~0.5
    int jM = -1, jz = -1; double best = 1e300;
    for (int a = 0; a < C.Nz; a++) for (int b = 0; b < C.NM; b++) {
        if (sh.gnorm[a][b] <= 0 || sh.fsb[a][b] <= 0) continue;
        double d = fabs(log(C.zlist[a]/0.5)) + fabs(log(C.Mlist[b]/1e13));
        if (d < best) { best = d; jz = a; jM = b; }
    }
    if (jz < 0) { printf("no active bin found\n"); return 1; }
    double zl = C.zlist[jz], M = C.Mlist[jM], Sigmac = Sigmacf(C, zs, zl);
    double f_b = sh.fsb[jz][jM];
    printf("bin: z_l=%.3f  M=%.3e  f_s,b=%.4f\n", zl, M, f_b);
    for (double r : {50.0, 150.0, 400.0}) {
        double M_u = sh.unresolvedMass(C, jz, jM, r, M);
        rgen mt2(777);
        long double sum = 0; int N = 8000; int nneg = 0;
        for (int i = 0; i < N; i++) {
            double kap = 0, g1 = 0, g2 = 0, Msum = 0;
            sh.addClumps(C, jz, jM, zl, M, Sigmac, r, 0.7, mt2, kap, g1, g2, 3, false, 1, 200000, &Msum);
            sum += Msum;
            if (M - Msum - M_u < 0) nneg++;
        }
        double meanMsum = (double)(sum / N);
        double lhs = (meanMsum + M_u) / M;   // (<Sum m_i> + M_u)/M
        printf("r=%6.1f kpc:  <Sum m_i>/M=%.5f  M_u/M=%.5f  sum=%.5f  vs f_s,b=%.5f  "
               "diff=%+.2e  neg-host draws=%d  %s\n",
               r, meanMsum / M, M_u / M, lhs, f_b, lhs - f_b, nneg,
               fabs(lhs - f_b) < 5e-3 ? "PASS" : "CHECK");
    }
    return 0;
}
