// Analytic variance partition vs kappa_thr under the correlated bias field
// (bias_model = 1) WITH the conditional weak arm (bias_weak = true).
//
// Per kappa_thr the weak (sub-threshold) and strong (explicit, kappa > kappa_thr)
// arms each carry
//   Var = shot (Campbell)  +  clustering (2-halo, from the shared per-shell field),
// and the arms are CORRELATED through the same realized field. Total:
//   Var_tot = VarW + VarS + 2 Cov_ws,
// which is kappa_thr-independent: the shot integrals tile kappa in
// (kappa_min, infinity), and the per-shell conditional means obey
// Sw_i(delta) + Ss_i(delta) = S_full,i(delta) for every delta (the per-cell
// means mW + mS = m_full are threshold-independent), so the clustering
// variance of the sum is invariant under the split.
//
// IMPORTANT — why this is NOT the naive log-normal pair sum: the formal
// clustering covariance E[lam_p lam_q] - 1 = exp(a_p a_q C_ij) - 1 DIVERGES
// (overflows double) for the top-of-HMF cells: a = b(M,z) Dg(z) reaches
// O(40-50) at M ~ 1e16-17 where exp(a^2 sig2) outgrows the HMF's
// exp(-q nu^2/2) suppression. Those formal moments are carried by field
// excursions delta ~ a sig2 >> 6 sigma with probability ~ exp(-a^2 sig2/2)
// (never sampled; the same "monster tail" class the standing rules exclude).
// Production's weak arm CLAMPS the field at +-6 sigma in its lnT/lnV tables
// (lensing.cpp BiasField1D::weakSV), so the well-defined object to plot is
// the CLAMPED law — which is exactly what production draws for the weak arm,
// and differs from the production strong arm only through events with
// per-ray probability ~1e-7 (65 shells x P(|delta|>6sig) ~ 2e-9).
//
// Method: per shell, production-convention 193-point log tables over
// +-6 sigma_i of T/V for BOTH arms (weak tables == production's own, strong
// tables the same construction on the explicit-disk moments); the variances
// are then composed from NMC field vectors drawn ONCE through the production
// Cholesky (BiasField1D::build) and REUSED across the whole kappa_thr grid,
// so the MC error is common mode and the conservation check stays sharp:
//   VarX = mean(sum_i Vx_i(delta)) + Var(sum_i Sx_i(delta)),
//   Cov_ws = Cov(sum_i Sw_i, sum_i Ss_i).
//
// Includes cpp/lensing.cpp directly so BiasField1D and weakMomentsNFW are the
// PRODUCTION implementations. The strong-disk moments mirror weakMomentsNFW's
// measure exactly (log annuli, inner-edge evaluation, d(pi r^2) = 2 pi r^2 dlnr),
// stepping INWARD from rmax; sum(vW) == sigmakappaW^2 is gated per point.
// Floor semantics = production: kappa_min held ABSOLUTE at 1e-3*kappa_thr(N=100).
//
// Build (repo root):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/sigma_partition_bias_vs_kthr.cpp cpp/cosmology.cpp cpp/basics.cpp cpp/subhalo.cpp \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/sigma_partition_bias_vs_kthr
// Run:  build/sigma_partition_bias_vs_kthr [zs=1.0] [Rperp=8441.0] [NMC=200000] [kappa_min=-1]
//   kappa_min > 0 overrides the absolute floor (default 1e-3*kappa_thr(N=100)) —
//   the floor-test lever: production reaches the same floor via kappathr_flat
//   (kappa_min = 1e-3*kappathr_flat when custom_kappathr drives the split).
//   Output file gains a _kmin<val> suffix when overridden.
// Writes playground/sigma_partition_bias_vs_kthr_zs<zs>[_kmin<..>].txt, columns:
//   kappa_thr  sigW_shot  sigW_corr  sigW_tot  sigS_shot  sigS_corr  sigS_tot  cov_ws  sig_tot

#include "lensing.cpp"  // production internals: BiasField1D, weakMomentsNFW, sigmakappaW, ...

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

// Strong-disk per-cell Campbell moments: m = int nbar kappa, v = int nbar kappa^2
// over the EXPLICIT disk r in (0, rmax(kappa_thr)]. Same integrand, log-annulus
// measure and inner-edge evaluation as weakMomentsNFW, stepping inward so the
// weak+strong annuli tile the full range with no gap or overlap at rmax.
static void strongMomentsNFW(cosmology &C, double zs, double kappathr,
                             std::vector<std::vector<double> > &m,
                             std::vector<std::vector<double> > &v) {
    m.assign(C.Nz, std::vector<double>(C.NM, 0.0));
    v.assign(C.Nz, std::vector<double>(C.NM, 0.0));
    const double dlnr = 0.01;
    const double Edlnr_in = exp(-dlnr);
    for (int jz = 1; jz < C.Nz; jz++) {
        const double zl = C.zlist[jz];
        const double dz = zl - C.zlist[jz-1];
        if (zl >= zs) continue;
        for (int jM = 1; jM < C.NM; jM++) {
            const double M = C.Mlist[jM];
            const double dlnM = log(M) - log(C.Mlist[jM-1]);
            const double dndlnM = C.HMFlist[jz][jM][0];

            const double rmax = rmaxfNFW(C, zs, zl, M, kappathr);
            if (rmax <= 0.0) continue;  // halo never exceeds kappa_thr

            const double Sigmac = Sigmacf(C, zs, zl);
            std::vector<double> NFWp = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
            const double rs = NFWp[0];
            const double kappa0 = kappa0NFW(rs, NFWp[1], Sigmac);
            const double pref0 = CLIGHT*2.0*PI*pow(1.0+zl,2.0)/C.Hz(zl)*dndlnM*dlnr*dlnM*dz;

            double r = rmax;
            for (int k = 0; k < 8000; k++) {
                r *= Edlnr_in;  // inner edge of the annulus [r, r/Edlnr_in]
                const double kap = kappagammaNFWeps(0.0, kappa0, r/rs, 0.0)[0];
                const double w = pref0*r*r;
                const double dm = w*kap, dv = w*kap*kap;
                m[jz][jM] += dm;
                v[jz][jM] += dv;
                if (dm < 1.0e-12*m[jz][jM] && dv < 1.0e-12*v[jz][jM]) break;
            }
        }
    }
}

// Per-shell production-convention tables (BiasField1D::buildWeak conventions:
// NGRID_W=193 points over +-6 sigma_i, log tables, clamped linear interp) of
// T(delta) = sum_M m_c lam_c(delta) and V(delta) = sum_M v_c lam_c(delta),
// lam_c = exp(a_c delta - a_c^2 sig2_i / 2), a_c = Dg(z_l) b(M, z_l).
struct ArmTables {
    int n = 0;
    std::vector<double> sig;             // per-shell field std
    std::vector<double> msum;            // per-shell sum_M m_c
    std::vector<double> lnT, lnV;        // row-major n * NG
    static constexpr int NG = BiasField1D::NGRID_W;      // 193
    static constexpr double DS = BiasField1D::DGRID_SIG; // 6.0

    void build(cosmology &C, double zs, const BiasField1D &bf,
               const std::vector<std::vector<double> > &m,
               const std::vector<std::vector<double> > &v) {
        n = bf.n;
        sig.assign(n, 0.0);
        msum.assign(n, 0.0);
        lnT.assign(static_cast<size_t>(n)*NG, 0.0);
        lnV.assign(static_cast<size_t>(n)*NG, 0.0);
        for (int i = 0; i < n; i++) {
            const int jz = i + 1;
            const double zl = C.zlist[jz];
            const double Dgz = C.Dg(zl);
            sig[i] = sqrt(bf.sig2[i]);
            for (int g = 0; g < NG; g++) {
                const double delta = (-DS + 2.0*DS*g/(NG - 1))*sig[i];
                double T = 0.0, V = 0.0;
                for (int jM = 1; jM < C.NM; jM++) {
                    const double mm = m[jz][jM], vv = v[jz][jM];
                    if (mm <= 0.0 && vv <= 0.0) continue;
                    const double a = Dgz*C.halobias(zl, C.sigmalist[jM][1]);
                    const double lam = exp(a*delta - 0.5*a*a*bf.sig2[i]);
                    T += mm*lam;
                    V += vv*lam;
                }
                lnT[static_cast<size_t>(i)*NG + g] = log(std::max(T, 1.0e-300));
                lnV[static_cast<size_t>(i)*NG + g] = log(std::max(V, 1.0e-300));
            }
            for (int jM = 1; jM < C.NM; jM++) msum[i] += m[jz][jM];
        }
    }

    // production weakSV: clamped log-linear interp; S = T - msum
    inline void SV(int i, double delta, double &S, double &V) const {
        const double half = DS*sig[i];
        const double x = std::min(std::max(delta, -half), half);
        const double t = (x + half)/(2.0*half)*(NG - 1);
        const int g = std::min(static_cast<int>(t), NG - 2);
        const double f = t - g;
        const double *rT = &lnT[static_cast<size_t>(i)*NG];
        const double *rV = &lnV[static_cast<size_t>(i)*NG];
        S = exp((1.0 - f)*rT[g] + f*rT[g+1]) - msum[i];
        V = exp((1.0 - f)*rV[g] + f*rV[g+1]);
    }
};

int main(int argc, char **argv) {
    const double zs = (argc > 1) ? std::atof(argv[1]) : 1.0;
    const double Rperp = (argc > 2) ? std::atof(argv[2]) : 8441.0;  // production default
    const long NMC = (argc > 3) ? std::atol(argv[3]) : 200000L;
    const double kmin_override = (argc > 4) ? std::atof(argv[4]) : -1.0;

    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0; C.sigma8 = 0.811;
    C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    std::function<double(double)> NfNFW = [&C, zs](double k) { return NhfNFW(C, zs, k); };
    const double kthr_fid = findkappathr(100, NfNFW);   // production <N>=100 threshold
    const double kappa_min = (kmin_override > 0.0)
        ? kmin_override
        : 1.0e-3*kthr_fid;                              // production absolute floor
    std::printf("zs = %g   Rperp = %g kpc   kappa_thr(N=100) = %.6e   kappa_min = %.3e%s   NMC = %ld\n",
                zs, Rperp, kthr_fid, kappa_min,
                (kmin_override > 0.0) ? " (OVERRIDE)" : "", NMC);

    // production field (shell segment covariance, exact Cholesky)
    BiasField1D bf;
    bf.build(C, zs, Rperp);
    const int n = bf.n;
    std::printf("field: n_shells = %d   Nmax = %ld modes   L = %.4e kpc\n", n, bf.Nmax, bf.L);

    // one fixed set of field vectors, reused for every kappa_thr (common-mode MC error)
    std::vector<float> fvals(static_cast<size_t>(NMC)*n);
    {
        std::mt19937_64 mt(20260716ULL);
        std::normal_distribution<double> G(0.0, 1.0);
        std::vector<double> g(n);
        for (long j = 0; j < NMC; j++) {
            for (int i = 0; i < n; i++) g[i] = G(mt);
            for (int i = 0; i < n; i++) {
                double s = 0.0;
                const double *row = &bf.chol[static_cast<size_t>(i)*n];
                for (int k = 0; k <= i; k++) s += row[k]*g[k];
                fvals[static_cast<size_t>(j)*n + i] = static_cast<float>(s);
            }
        }
    }

    // quarter-decade grid; extends BELOW the absolute floor kappa_min = 1.28e-7,
    // where the weak arm is empty by construction (eps_floor > 1) and the explicit
    // arm additionally picks up the sub-floor band the default path drops — the
    // conservation plateau is only claimed for kt >= kappa_min. NB below
    // kt ~ 3e-8 the rmaxfNFW bisection ceiling (r <= 1e6 kpc) truncates the
    // largest cells' discs (identically in production — consistency preserved).
    std::vector<double> kthr;
    for (int i = 0; i <= 44; i++) kthr.push_back(std::pow(10.0, -8.0 + 11.0*i/44.0));

    char fn[256];
    if (kmin_override > 0.0) {
        std::snprintf(fn, sizeof(fn), "playground/sigma_partition_bias_vs_kthr_zs%g_kmin%g.txt",
                      zs, kmin_override);
    } else {
        std::snprintf(fn, sizeof(fn), "playground/sigma_partition_bias_vs_kthr_zs%g.txt", zs);
    }
    FILE *fp = std::fopen(fn, "w");
    std::fprintf(fp, "# bias_model=1 Rperp=%g bias_weak partition (6-sigma-clamped tables, "
                 "production convention); kappa_min=%.6e (absolute); NMC=%ld\n",
                 Rperp, kappa_min, NMC);
    std::fprintf(fp, "# kappa_thr   sigW_shot   sigW_corr   sigW_tot   sigS_shot   sigS_corr   "
                 "sigS_tot   cov_ws   sig_tot\n");

    std::printf("%-11s %-11s %-11s %-11s %-11s %-11s %-11s %-12s %-11s\n",
                "kappa_thr", "sigW_shot", "sigW_corr", "sigW_tot",
                "sigS_shot", "sigS_corr", "sigS_tot", "cov_ws", "sig_tot");

    std::vector<double> shot_all, varT_all, kt_all;
    for (double kt : kthr) {
        const double eps = kappa_min/kt;   // floor-consistent (production) semantics

        std::vector<std::vector<double> > mW, vW, mS, vS;
        weakMomentsNFW(C, zs, kt, eps, mW, vW);
        strongMomentsNFW(C, zs, kt, mS, vS);

        double sumVW = 0.0, sumVS = 0.0;
        for (int jz = 0; jz < C.Nz; jz++)
            for (int jM = 0; jM < C.NM; jM++) { sumVW += vW[jz][jM]; sumVS += vS[jz][jM]; }

        // gate: weak shot moments == sigmakappaW^2 (same code path, roundoff only)
        const double sW2 = pow(sigmakappaW(C, zs, kt, eps), 2.0);
        if (std::fabs(sumVW - sW2) > 1.0e-9*sW2) {
            std::fprintf(stderr, "GATE FAIL: sum vW = %.12e vs sigmakappaW^2 = %.12e at kt=%.3e\n",
                         sumVW, sW2, kt);
            return 1;
        }

        ArmTables TW, TS;
        TW.build(C, zs, bf, mW, vW);
        TS.build(C, zs, bf, mS, vS);

        // compose the variances from the shared field draws; clamp_c < 6
        // additionally pre-clamps the field at +-clamp_c sigma_i (diagnostic:
        // locates which field excursions carry the clustering variance).
        // Outputs: total arm variances, the clustering (corr) parts Var(S),
        // the cross covariance, and the conditional-shot means E[V].
        auto compose = [&](double clamp_c, double &varW_, double &varS_,
                           double &varSW_, double &varSS_, double &covWS_,
                           double &vwbar_, double &vsbar_) {
            double s1w = 0.0, s2w = 0.0, s1s = 0.0, s2s = 0.0, sws = 0.0;
            double vw = 0.0, vs = 0.0;
            for (long j = 0; j < NMC; j++) {
                const float *fj = &fvals[static_cast<size_t>(j)*n];
                double SWj = 0.0, SSj = 0.0, VWj = 0.0, VSj = 0.0, S, V;
                for (int i = 0; i < n; i++) {
                    double d = static_cast<double>(fj[i]);
                    const double h = clamp_c*TW.sig[i];
                    d = std::min(std::max(d, -h), h);
                    TW.SV(i, d, S, V); SWj += S; VWj += V;
                    TS.SV(i, d, S, V); SSj += S; VSj += V;
                }
                s1w += SWj; s2w += SWj*SWj;
                s1s += SSj; s2s += SSj*SSj;
                sws += SWj*SSj;
                vw += VWj; vs += VSj;
            }
            const double mw = s1w/NMC, ms = s1s/NMC;
            varSW_ = s2w/NMC - mw*mw;
            varSS_ = s2s/NMC - ms*ms;
            varW_ = vw/NMC + varSW_;
            varS_ = vs/NMC + varSS_;
            covWS_ = sws/NMC - mw*ms;
            vwbar_ = vw/NMC; vsbar_ = vs/NMC;
        };

        double varW, varS, varSW, varSS, covWS, vwbar, vsbar;
        compose(ArmTables::DS, varW, varS, varSW, varSS, covWS, vwbar, vsbar);

        // clamp-level scan at the grid ends: where does the clustering variance live?
        if (kt == kthr.front() || kt == kthr.back()) {
            std::printf("  [diag kt=%.3e] field-clamp scan (c: varW varS cov varT):\n", kt);
            for (double c : {2.0, 3.0, 4.0, 5.0, 6.0}) {
                double w_, S_, cw_, cs_, x_, vw_, vs_;
                compose(c, w_, S_, cw_, cs_, x_, vw_, vs_);
                std::printf("    c=%.0f: %.4e %.4e %.4e %.4e\n",
                            c, w_, S_, x_, w_ + S_ + 2.0*x_);
            }
        }

        const double varT = varW + varS + 2.0*covWS;
        const double shot = sumVW + sumVS;
        kt_all.push_back(kt);
        shot_all.push_back(shot);
        varT_all.push_back(varT);

        std::fprintf(fp, "%.6e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                     kt, sqrt(sumVW), sqrt(varSW), sqrt(varW),
                     sqrt(sumVS), sqrt(varSS), sqrt(varS), covWS, sqrt(varT));
        std::printf("%-11.3e %-11.6f %-11.6f %-11.6f %-11.6f %-11.6f %-11.6f %-12.4e %-11.6f\n",
                    kt, sqrt(sumVW), sqrt(varSW), sqrt(varW),
                    sqrt(sumVS), sqrt(varSS), sqrt(varS), covWS, sqrt(varT));
        std::fflush(fp);
        std::fflush(stdout);
    }
    std::fclose(fp);

    // conservation is claimed for kt >= kappa_min only (below the absolute floor
    // the weak arm is empty and the explicit arm includes the sub-floor band the
    // default path drops — a different, larger total by construction)
    double sref = -1.0, tref = -1.0, sdev = 0.0, tmin = 1e30, tmax = 0.0;
    double t_deep_max = 0.0;
    for (size_t i = 0; i < kt_all.size(); i++) {
        if (kt_all[i] >= kappa_min) {
            if (sref < 0.0) { sref = shot_all[i]; tref = varT_all[i]; }
            sdev = std::max(sdev, std::fabs(shot_all[i]/sref - 1.0));
            tmin = std::min(tmin, varT_all[i]);
            tmax = std::max(tmax, varT_all[i]);
        } else {
            t_deep_max = std::max(t_deep_max, varT_all[i]);
        }
    }
    std::printf("\nconservation (kt >= kappa_min): max |shot/ref - 1| = %.3e   "
                "varT spread (max-min)/mid = %.3e\n", sdev, (tmax - tmin)/(0.5*(tmax + tmin)));
    if (t_deep_max > 0.0) {
        std::printf("below-floor (kt < kappa_min): max varT = %.4e = %.3f x plateau "
                    "(sub-floor band re-enters the explicit arm)\n",
                    t_deep_max, t_deep_max/(0.5*(tmax + tmin)));
    }
    std::printf("wrote %s\n", fn);
    return 0;
}
