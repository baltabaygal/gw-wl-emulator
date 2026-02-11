#include <complex>
#include <vector>
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <gsl/gsl_sf_expint.h>


using namespace std;
namespace fs = filesystem;

typedef mt19937_64 rgen;

const double PI = 3.141592653589793238463;

string to_string_prec(const double a, const int n);
bool fileExists(const string& filename);

vector<double> linlist(double xmin, double xmax, int Nx);
vector<double> loglist(double xmin, double xmax, int Nx);

double randomreal(double x1, double x2, rgen &mt);
double randomreal(double x1, double x2);

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
