// Exact moment decomposition of the full Campbell variance sigma_full(zs).
//
// From lensing.cpp::Sigmacf (flat universe): 1/Sigma_c = A0 * chi_l (chi_s - chi_l)
// / (chi_s (1+z_l)), A0 = 4 pi / 2.08871e16, so ALL z_s dependence of the Campbell
// integrand enters through chi_s. With the (1+z)^2 path factor cancelling the
// (1+z_l)^-2 of Sigma_c^-2:
//
//   sigma^2(zs) = int_0^zs P(z) [chi (1 - chi/chi_s)]^2 dz            (exact)
//               = M2(zs) - 2 M3(zs)/chi_s + M4(zs)/chi_s^2
//   P(z)  = 2 pi C2 * (306.535/Hz(z)) * A0^2 * B(z)
//   B(z)  = int dlnM dn/dlnM [rhos rs^2]^2
//   Mk(zs)= int_0^zs P(z) chi(z)^k dz     (cumulative moments, z_s-independent P)
//
// The probe uses an extended fine grid (zmin=0.002, zmax=30.01) to probe both
// asymptotics: sigma ~ zs^{3/2} as zs->0, and saturation sigma -> sigma_inf as
// chi_s -> chi_horizon (P dies with the HMF, moments converge).
//
// Outputs:
//   playground/campbell_P_of_z.txt      z  chi(z)[kpc]  P(z)  M2(z) M3(z) M4(z)
//   playground/campbell_sigma_dense.txt zs sigma(zs) (dense, from the moment formula)
//   stdout: validation vs sigmakappaW plateau + sigma_inf estimate.
//
// Build (repo root, after `make build`):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/campbell_moments.cpp -Lbuild -lgwcore \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/campbell_moments

#include "cosmology.h"
#include "lensing.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <vector>

double Sigmacf(cosmology &C, double zs, double zl);
double NhfNFW(cosmology &C, double zs, double kappathr);
double findkappathr(int N, std::function<double(double)> Nf);

int main() {
    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0; C.sigma8 = 0.811;
    C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.002; C.zmax = 30.01; C.Nz = 220;   // extended fine grid
    C.outdir = "dataL";
    C.initialize(0);

    // ---- C2 (pure NFW number)
    double C2 = 0.0;
    const double dlnx = 1.0e-3;
    for (double lnx = std::log(1.0e-8); lnx < std::log(1.0e8); lnx += dlnx) {
        const double x = std::exp(lnx);
        const double K = kappagammaNFWeps(0.0, 1.0, x, 0.0)[0];
        C2 += K * K * x * x * dlnx;
    }

    const double A0 = 4.0 * PI / 2.08871e16;

    // ---- verify the Sigmacf factorization: A(zl) = [1/Sigmacf] * chi_s /
    //      (chi_l (chi_s - chi_l)) must be A0/(1+zl), independent of zs.
    //      (chi from the code's own dc().)
    {
        double worst = 0.0;
        for (double zl : {0.1, 0.5, 2.0, 8.0}) {
            for (double zs : {1.0, 5.0, 25.0}) {
                if (zl >= zs) continue;
                const double chil = C.dc(zl), chis = C.dc(zs);
                const double lhs = 1.0 / Sigmacf(C, zs, zl);
                const double rhs = A0 * chil * (chis - chil) / (chis * (1.0 + zl));
                worst = std::max(worst, std::fabs(lhs / rhs - 1.0));
            }
        }
        std::printf("Sigmacf factorization check: max |ratio-1| = %.2e\n", worst);
    }

    // ---- P(z), chi(z), cumulative moments on the grid
    const int Nz = C.Nz;
    std::vector<double> zl(Nz), chi(Nz), P(Nz), M2(Nz, 0.0), M3(Nz, 0.0), M4(Nz, 0.0);
    FILE *fp = std::fopen("playground/campbell_P_of_z.txt", "w");
    std::fprintf(fp, "# C2 = %.8f\n# z   chi[kpc]   P(z)   M2   M3   M4\n", C2);
    for (int jz = 1; jz < Nz; jz++) {
        zl[jz] = C.zlist[jz];
        const double dz = C.zlist[jz] - C.zlist[jz - 1];
        chi[jz] = C.dc(zl[jz]);
        double B = 0.0;
        for (int jM = 1; jM < C.NM; jM++) {
            const double dlnM = std::log(C.Mlist[jM]) - std::log(C.Mlist[jM - 1]);
            const double dndlnM = C.HMFlist[jz][jM][0];
            const double rs = C.NFWlist[jz][jM][0];
            const double rhos = C.NFWlist[jz][jM][1];
            const double s = rhos * rs * rs;
            B += dndlnM * s * s * dlnM;
        }
        P[jz] = 2.0 * PI * C2 * 306.535 / C.Hz(zl[jz]) * A0 * A0 * B;
        const double c2 = chi[jz] * chi[jz];
        M2[jz] = M2[jz - 1] + P[jz] * c2 * dz;
        M3[jz] = M3[jz - 1] + P[jz] * c2 * chi[jz] * dz;
        M4[jz] = M4[jz - 1] + P[jz] * c2 * c2 * dz;
        std::fprintf(fp, "%.6f %.8e %.8e %.8e %.8e %.8e\n",
                     zl[jz], chi[jz], P[jz], M2[jz], M3[jz], M4[jz]);
    }
    std::fclose(fp);

    auto sigma_moments = [&](double zs) {
        const double chis = C.dc(zs);
        // moments up to zs (linear interp of the cumulative arrays in z)
        int j = 1;
        while (j < Nz - 1 && zl[j + 1] < zs) j++;
        const double t = (zs - zl[j]) / (zl[j + 1] - zl[j]);
        const double m2 = M2[j] + t * (M2[j + 1] - M2[j]);
        const double m3 = M3[j] + t * (M3[j + 1] - M3[j]);
        const double m4 = M4[j] + t * (M4[j + 1] - M4[j]);
        return std::sqrt(m2 - 2.0 * m3 / chis + m4 / (chis * chis));
    };

    // ---- validation vs brute sigmakappaW plateau (same cosmology object)
    std::printf("\n%-8s %-14s %-14s %-10s\n", "zs", "sigma_moments", "sigma_brute", "ratio");
    for (double zs : {0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0}) {
        std::function<double(double)> NfNFW = [&C, zs](double k) { return NhfNFW(C, zs, k); };
        const double kfid = findkappathr(100, NfNFW);
        const double brute = sigmakappaW(C, zs, 1.0e3, 1.0e-3 * kfid / 1.0e3);
        const double mom = sigma_moments(zs);
        std::printf("%-8.3g %-14.6e %-14.6e %-10.6f\n", zs, mom, brute, mom / brute);
    }

    // ---- dense sigma(zs) from the moment formula
    FILE *fd = std::fopen("playground/campbell_sigma_dense.txt", "w");
    std::fprintf(fd, "# zs   sigma_moments\n");
    for (int i = 0; i <= 120; i++) {
        const double zs = 0.02 * std::pow(30.0 / 0.02, i / 120.0);
        std::fprintf(fd, "%.6f %.8e\n", zs, sigma_moments(zs));
    }
    std::fclose(fd);

    // ---- sigma_inf: moments frozen at zmax, chi_s -> chi_horizon
    //      chi_hor = chi(zmax) + int_zmax^inf 306.535/Hz dz  (direct sum)
    double chihor = C.dc(30.0);
    {
        const double dlnz = 1.0e-3;
        for (double lnz = std::log(31.0); lnz < std::log(5000.0); lnz += dlnz) {
            const double z = std::exp(lnz);
            chihor += 306.535 / C.Hz(z) * z * dlnz;
        }
        // (z > 5000 contributes < 0.1% of the remainder; radiation era)
    }
    const double m2 = M2[Nz - 1], m3 = M3[Nz - 1], m4 = M4[Nz - 1];
    const double siginf = std::sqrt(m2 - 2.0 * m3 / chihor + m4 / (chihor * chihor));
    std::printf("\nchi(30) = %.6e kpc   chi_hor = %.6e kpc\n", C.dc(30.0), chihor);
    std::printf("sigma(30) = %.6f   sigma_inf = %.6f  (moments frozen at z=30)\n",
                sigma_moments(30.0), siginf);
    std::printf("frozen-moment check: M2/M2(z=20) = %.6f (P(z) support exhausted?)\n",
                M2[Nz - 1] / M2[std::max(1, (int)(std::lower_bound(zl.begin() + 1, zl.end(), 20.0) - zl.begin()) - 1)]);
    // low-z law: sigma^2 -> P(0) [chi'(0) z]^2 * z/30 gives sigma = sqrt(P(0)/30) chi'(0) z^{3/2}
    std::printf("\nC2 = %.8f   A_lowz = sqrt(P(0)/30) * chi'(0) prediction:\n", C2);
    const double chip0 = 306.535 / C.Hz(C.zlist[1]);
    std::printf("  P(z1)=%.6e  chi'(0)=%.6e  A_pred = %.6f  (fit gave 0.0431)\n",
                P[1], chip0, std::sqrt(P[1] / 30.0) * chip0);
    std::printf("\nwrote playground/campbell_P_of_z.txt, campbell_sigma_dense.txt\n");
    return 0;
}
