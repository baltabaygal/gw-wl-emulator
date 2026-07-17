// Standalone driver: kappa_thr under the legacy <N>=Nhalos rule as a function of z_s.
// Copies the NFW-threshold free functions verbatim from cpp/lensing.cpp (no GSL, no
// subhalo path), links only cosmology.cpp + basics.cpp (+ Ci/Si stubs).
#include "cosmology.h"
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>

// ---- verbatim from cpp/lensing.cpp ----
double Sigmacf(cosmology &C, double zs, double zl) {
    double DsA = C.DL(zs)/pow(1+zs,2.0);
    double DlA = C.DL(zl)/pow(1+zl,2.0);
    double DlsA = DsA - DlA*(1+zl)/(1+zs);
    return 2.08871e16*DsA/(4.0*PI*DlA*DlsA);
}
array<double,2> FgNFW(double x) {
    if (x > 1) { double t = atan(sqrt((x-1)/(1+x)))/sqrt(x*x-1); return {(1-2*t)/(x*x-1), 2*t+log(x/2)}; }
    if (x < 1) { double t = atanh(sqrt((1-x)/(1+x)))/sqrt(1-x*x); return {(1-2*t)/(x*x-1), 2*t+log(x/2)}; }
    return {1.0/3.0, 1 + log(0.5)};
}
static inline double safeNFWGammaCore(double x, const array<double,2> &Fg) {
    if (x < 1.0e-4) return 0.5;
    return 2.0*Fg[1]/(x*x) - Fg[0];
}
double kappa0NFW(double rs, double rhos, double Sigmac) { return rs*rhos/Sigmac; }
array<double,2> kappagammaNFWeps(double epsilon, double kappa0, double x, double phi) {
    double x1eps = sqrt(1.0-epsilon)*cos(phi)*x, x2eps = sqrt(1.0+epsilon)*sin(phi)*x;
    double xeps = max(sqrt(pow(x1eps,2.0)+pow(x2eps,2.0)), 1.0e-12);
    double phieps = atan2(x2eps, x1eps);
    auto Fg = FgNFW(xeps);
    double kappaeps0 = 2.0*kappa0*Fg[0];
    double gammaeps0 = 2.0*kappa0*safeNFWGammaCore(xeps, Fg);
    double kappaeps = kappaeps0 + epsilon*cos(2.0*phieps)*gammaeps0;
    double gammaeps2 = pow(gammaeps0,2.0)+2.0*epsilon*cos(2.0*phieps)*gammaeps0*kappaeps0+pow(epsilon,2.0)*(pow(kappaeps0,2.0)-pow(cos(2.0*phieps)*gammaeps0,2.0));
    return {kappaeps, sqrt(max(0.0,gammaeps2))};
}
array<double,2> kappagammaNFW(cosmology &C, double zs, double zl, double r, double M, double phi, double epsilon) {
    double Sigmac = Sigmacf(C, zs, zl);
    vector<double> P = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
    return kappagammaNFWeps(epsilon, kappa0NFW(P[0], P[1], Sigmac), r/P[0], phi);
}
double rmaxfNFW(cosmology &C, double zs, double zl, double M, double kappathr) {
    double rmax, logr1 = log(1.0e-6), logr2 = log(1.0e6);
    if (kappagammaNFW(C, zs, zl, exp(logr1), M, 0.0, 0.0)[0] > kappathr) {
        while (logr2-logr1 > 0.02) {
            rmax = exp((logr2+logr1)/2.0);
            if (kappagammaNFW(C, zs, zl, rmax, M, 0.0, 0.0)[0] > kappathr) logr1 = log(rmax); else logr2 = log(rmax);
        }
        rmax = exp((logr2+logr1)/2.0);
    } else rmax = 0.0;
    return rmax;
}
double NhfNFW(cosmology &C, double zs, double kappathr) {
    double Nh = 0.0, zl, dz, M, dlnM, dndlnM, rmax;
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz]; dz = zl - C.zlist[jz-1];
        if (zl < zs) for (int jM = 1; jM < C.NM; jM++) {
            M = C.Mlist[jM]; dlnM = log(M) - log(C.Mlist[jM-1]); dndlnM = C.HMFlist[jz][jM][0];
            rmax = rmaxfNFW(C, zs, zl, M, kappathr);
            Nh += CLIGHT*PI*pow((1.0+zl)*rmax,2.0)/C.Hz(zl)*dndlnM*dlnM*dz;
        }
    }
    return Nh;
}
double findkappathr(int N, function<double(double)> Nf) {
    double k1 = 1.0e-12, k2 = 1.0, kthr = pow(10.0,(log10(k1)+log10(k2))/2.0);
    while (log10(k2)-log10(k1) > 0.01) {
        if (Nf(kthr) > N) k1 = kthr; else k2 = kthr;
        kthr = pow(10.0,(log10(k1)+log10(k2))/2.0);
    }
    return kthr;
}

int main() {
    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0;
    C.sigma8 = 0.811; C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    const int Nhalos = 100;
    std::vector<double> zs_list = {0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0};
    std::ofstream out("tmp/kthr_driver/kthr_vs_zs.dat");
    out << "# zs   kappathr(N=100)   Ncheck\n";
    std::cout << "zs    kappathr    Ncheck\n";
    for (double zs : zs_list) {
        std::function<double(double)> Nf = [&C, zs](double k){ return NhfNFW(C, zs, k); };
        double kthr = findkappathr(Nhalos, Nf);
        double ncheck = NhfNFW(C, zs, kthr);
        out << zs << "  " << kthr << "  " << ncheck << "\n";
        std::cout << zs << "  " << kthr << "  " << ncheck << std::endl;
    }
    out.close();
    return 0;
}
