#include <complex>
#include <array>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <functional>
#include <gsl/gsl_sf_expint.h>


using namespace std;
namespace fs = filesystem;

typedef mt19937_64 rgen;

const double PI = 3.141592653589793238463;
// Speed of light in internal units (kpc/Gyr); gives c/H(z) = CLIGHT/Hz(z) kpc comoving.
const double CLIGHT = 306.535;

string to_string_prec(const double a, const int n);
bool fileExists(const string& filename);

vector<double> linlist(double xmin, double xmax, int Nx);
vector<double> loglist(double xmin, double xmax, int Nx);

inline double randomreal(double x1, double x2, rgen &mt) {
    double r01 = (double)mt() * (1.0 / (double)mt.max());
    return (x1 + (x2-x1)*r01);
}
inline double randomreal(double x1, double x2) {
    double r01 = (double)rand() * (1.0 / (double)RAND_MAX);
    return (x1 + (x2-x1)*r01);
}

double NPDF(double x, double mu, double sigma);
double logNPDF(double x, double mu, double sigma);

double interpolate(double X, vector<vector<double> > &xy);
vector<double> interpolaten(double X, vector<vector<double> > &xy);
double interpolate2(double x0, double y0, vector<double> &x, vector<double> &y, vector<vector<double> > &f);
vector<double> interpolate2(double x0, double y0, vector<double> &x, vector<double> &y, vector<vector<vector<double> > > &f);

double variance(vector<double> &sample);

vector<double> findMinMax(vector<double> &sample);
vector<double> findMinMax(vector<vector<double> > &sample2, int j);

vector<vector<double> > binSample(vector<double> &sample, int N);
vector<double> confidenceInterval(vector<double> &sample, double P);

vector<double> sampleFromCDF(vector<vector<double> > &cdf, int N, rgen &mt);
vector<double> sampleFromPDF(vector<vector<double> > &pdf, int N, rgen &mt);
    
void writeToFile(vector<double> &row, const string &filename);
void writeToFile(vector<vector<double> > &matrix, const string &filename);
void writeToFile(vector<double> &x, vector<vector<vector<double> > > &cubic, const string &filename);
void writeToFile(vector<double> &x, vector<double> &y, vector<vector<double> > &f, const string &filename);
void writeToFile(vector<double> &x, vector<double> &y, vector<vector<vector<double> > > &f, const string &filename);
void writeToFile(vector<double> &x, vector<double> &y, vector<double> &z, vector<vector<vector<double> > > &f, const string &filename);
void writeToFile(vector<double> &x, vector<double> &y, vector<double> &z, vector<vector<vector<vector<double> > > > &f, const string &filename);

vector<double> readdata(fs::path filename);
vector<vector<double> > readdata(fs::path filename, int N);
vector<vector<double> > readdataCSV(fs::path filename);

vector<vector<double> > MCMC_sampling(int N, int Nburnin, function<double(vector<double>&)> logpdf, vector<double> &initial, vector<double> &steps, vector<vector<double> > &priors, function<double(vector<double>&)> cut, rgen &mt, int print, int printL, fs::path filename);

