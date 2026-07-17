#ifndef STUB_GSL_SF_EXPINT_H
#define STUB_GSL_SF_EXPINT_H
/* Minimal stand-in for GSL's expint header: only the two special functions
   cosmology.cpp actually references (definitions provided in gsl_stubs.cpp). */
#ifdef __cplusplus
extern "C" {
#endif
double gsl_sf_Ci(const double x);
double gsl_sf_Si(const double x);
#ifdef __cplusplus
}
#endif
#endif
