// Minimal replacements for the only two GSL symbols cosmology.cpp needs
// (gsl_sf_Ci, gsl_sf_Si, used inside rhokNFW — off the lensing-threshold path).
// Abramowitz & Stegun 5.2.38/5.2.39 auxiliary-function approximations (err < 5e-7),
// series for x<1. C linkage to match the GSL headers' extern "C" declarations.
#include <cmath>

static double si_series(double x) {          // Si(x), x small
    double term = x, sum = x, x2 = x*x;
    for (int k = 1; k < 30; k++) {
        term *= -x2 / ((2*k)*(2*k+1));
        sum += term / (2*k+1);
        if (std::fabs(term) < 1e-18) break;
    }
    return sum;
}
static double ci_series(double x) {          // Ci(x), x small
    const double euler = 0.5772156649015328606;
    double sum = euler + std::log(x), term = 1.0, x2 = x*x;
    for (int k = 1; k < 30; k++) {
        term *= -x2 / ((2*k)*(2*k-1));       // (-1)^k x^{2k}/((2k)(2k)!) built incrementally
        sum += term / (2*k);
        if (std::fabs(term) < 1e-18) break;
    }
    return sum;
}
static void fg(double x, double &f, double &g) {
    double x2=x*x, x4=x2*x2, x6=x4*x2, x8=x4*x4;
    f = (1.0/x) * (x8 + 38.027264*x6 + 265.187033*x4 + 335.677320*x2 + 38.102495) /
                  (x8 + 40.021433*x6 + 322.624911*x4 + 570.236280*x2 + 157.105423);
    g = (1.0/x2)* (x8 + 42.242855*x6 + 302.757865*x4 + 352.018498*x2 + 21.821899) /
                  (x8 + 48.196927*x6 + 482.485984*x4 + 1114.978885*x2 + 449.690326);
}
extern "C" double gsl_sf_Si(const double x) {
    double ax = std::fabs(x), s;
    if (ax < 1.0) s = si_series(ax);
    else { double f,g; fg(ax,f,g); s = M_PI/2.0 - f*std::cos(ax) - g*std::sin(ax); }
    return (x < 0) ? -s : s;
}
extern "C" double gsl_sf_Ci(const double x) {
    double ax = std::fabs(x);
    if (ax < 1.0) return ci_series(ax);
    double f,g; fg(ax,f,g);
    return f*std::sin(ax) - g*std::cos(ax);
}
