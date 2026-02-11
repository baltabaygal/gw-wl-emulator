#include "basics.h"

string to_string_prec(double a, const int n) {
    ostringstream out;
    out.precision(n);
    out << fixed << a;
    return out.str();
}

bool fileExists(const string& filename) {
    ifstream file(filename);
    return file.good();
}

// generates a list in linear scale
vector<double> linlist(double xmin, double xmax, int Nx) {
    if (Nx > 1) {
        double dx = (xmax-xmin)/(1.0*(Nx-1));
        double x = xmin;
        vector<double> tmp(Nx,0.0);
        for (int j = 0; j < Nx; j++) {
            tmp[j] = x;
            x += dx;
        }
        return tmp;
    } else {
        vector<double> tmp;
        return tmp;
    }
}

// generates a list in log scale
vector<double> loglist(double xmin, double xmax, int Nx) {
    if (Nx > 1) {
        double dlogx = (log(xmax)-log(xmin))/(1.0*(Nx-1));
        double x = xmin;
        vector<double> tmp(Nx,0.0);
        for (int j = 0; j < Nx; j++) {
            tmp[j] = x;
            x = exp(log(x) + dlogx);
        }
        return tmp;
    } else {
        vector<double> tmp;
        return tmp;
    }
}

// random real number in the range (x_1,x_2)
double randomreal(double x1, double x2, rgen &mt) {
    long double r01 = mt()/(1.0*mt.max());
    return (x1 + (x2-x1)*r01);
}
double randomreal(double x1, double x2) {
    long double r01 = rand()/(1.0*RAND_MAX);
    return (x1 + (x2-x1)*r01);
}

// normal distribution
double NPDF(double x, double mu, double sigma) {
    return exp(-pow((x-mu)/sigma,2.0)/2.0)/(sqrt(2*PI)*sigma);
}

// logarithm of a normal distribution
double logNPDF(double x, double mu, double sigma) {
    return (-pow((x-mu)/sigma,2.0) - log(2*PI) - 2.0*log(sigma))/2.0;
}


// linear interpolation
// xy[all][0] must be strictly increasing or strictly decreasing
double interpolate(double X, vector<vector<double> > &xy) {
    
    int n = xy.size();
    auto comp0 = [&](vector<double> &XY, double val) {
      return XY[0] < val;
    };
    
    int jx;
    if (xy[0][0] < xy[n-1][0]) {
        jx = lower_bound(xy.begin(), xy.end(), X, comp0) - xy.begin();
    } else {
        jx = xy.rend() - lower_bound(xy.rbegin(), xy.rend(), X, comp0);
    }
    
    double Y;
    if (jx <= 0) {
        Y = xy[0][1];
    } else if (jx >= n) {
        Y = xy[n-1][1];
    } else {
        Y = xy[jx-1][1] + (X-xy[jx-1][0])/(xy[jx][0]-xy[jx-1][0])*(xy[jx][1]-xy[jx-1][1]);
    }
    return Y;
}

// linear interpolation of vector of form {{x_1,y_11,y_21,...},{x_2,y_12,y_22,...},...}
// xy[all][0] must be strictly increasing or strictly decreasing
vector<double> interpolaten(double X, vector<vector<double> > &xy) {
    
    int n = xy.size();
    auto comp0 = [&](vector<double> &XY, double val) {
      return XY[0] < val;
    };
    
    int jx;
    if (xy[0][0] < xy[n-1][0]) {
        jx = lower_bound(xy.begin(), xy.end(), X, comp0) - xy.begin();
    } else {
        jx = xy.rend() - lower_bound(xy.rbegin(), xy.rend(), X, comp0);
    }
    
    vector<double> tmp(xy[0].size(),0.0);
    if (jx <= 0) {
        tmp = xy[0];
    } else if (jx >= n) {
        tmp = xy[n-1];
    } else {
        tmp[0] = X;
        for (int j = 1; j < tmp.size(); j++) {
            tmp[j] = xy[jx-1][j] + (X-xy[jx-1][0])/(xy[jx][0]-xy[jx-1][0])*(xy[jx][j]-xy[jx-1][j]);
        }
    }
    return tmp;
}

// linear interpolation
double lin(double y1, double y2, double x1, double x2, double x) {
    return y1 + (x-x1)/(x2-x1)*(y2 - y1);
}

vector<double> interpolate2(double x0, double y0, vector<double> &x, vector<double> &y, vector<vector<vector<double> > > &f) {
    
    int nx = x.size();
    int ny = y.size();
    int nf = f[0][0].size();
    
    // find the closest corner where x>x0 and y>y0
    int jx = lower_bound(x.begin(), x.end(), x0) - x.begin();
    int jy = lower_bound(y.begin(), y.end(), y0) - y.begin();

    vector<double> tmp(nf, 0.0);
    if (jx <= 0) {
        if (jy <= 0) {
            tmp = f[0][0];
        } else if (jy >= ny) {
            tmp = f[0][ny-1];
        } else {
            for (int j = 0; j < nf; j++) {
                tmp[j] = lin(f[0][jy-1][j], f[0][jy][j], y[jy-1], y[jy], y0);
            }
        }
    } else if (jx >= nx) {
        if (jy <= 0) {
            tmp = f[nx-1][0];
        } else if (jy >= ny) {
            tmp = f[nx-1][ny-1];
        } else {
            for (int j = 0; j < nf; j++) {
                tmp[j] = lin(f[nx-1][jy-1][j], f[nx-1][jy][j], y[jy-1], y[jy], y0);
            }
        }
    } else {
        if (jy <= 0) {
            for (int j = 0; j < nf; j++) {
                tmp[j] = lin(f[jx-1][0][j], f[jx][0][j], x[jx-1], x[jx], x0);
            }
        } else if (jy >= ny) {
            for (int j = 0; j < nf; j++) {
                tmp[j] = lin(f[jx-1][ny-1][j], f[jx][ny-1][j], x[jx-1], x[jx], x0);
            }
        }
        else {
            double low, up;
            for (int j = 0; j < nf; j++) {
                low = lin(f[jx-1][jy-1][j], f[jx][jy-1][j], x[jx-1], x[jx], x0);
                up = lin(f[jx-1][jy][j], f[jx][jy][j], x[jx-1], x[jx], x0);
                tmp[j] = lin(low, up, y[jy-1], y[jy], y0);
            }
        }
    }
    return tmp;
}

double interpolate2(double x0, double y0, vector<double> &x, vector<double> &y, vector<vector<double> > &f) {
    
    int nx = x.size();
    int ny = y.size();
    
    // find the closest corner where x>x0 and y>y0
    int jx = lower_bound(x.begin(), x.end(), x0) - x.begin();
    int jy = lower_bound(y.begin(), y.end(), y0) - y.begin();

    double tmp = 0.0;
    if (jx <= 0) {
        if (jy <= 0) {
            tmp = f[0][0];
        } else if (jy >= ny) {
            tmp = f[0][ny-1];
        } else {
            tmp = lin(f[0][jy-1], f[0][jy], y[jy-1], y[jy], y0);
        }
    } else if (jx >= nx) {
        if (jy <= 0) {
            tmp = f[nx-1][0];
        } else if (jy >= ny) {
            tmp = f[nx-1][ny-1];
        } else {
            tmp = lin(f[nx-1][jy-1], f[nx-1][jy], y[jy-1], y[jy], y0);
        }
    } else {
        if (jy <= 0) {
            tmp = lin(f[jx-1][0], f[jx][0], x[jx-1], x[jx], x0);
        } else if (jy >= ny) {
            tmp = lin(f[jx-1][ny-1], f[jx][ny-1], x[jx-1], x[jx], x0);
        }
        else {
            double low = lin(f[jx-1][jy-1], f[jx][jy-1], x[jx-1], x[jx], x0);
            double up = lin(f[jx-1][jy], f[jx][jy], x[jx-1], x[jx], x0);
            tmp = lin(low, up, y[jy-1], y[jy], y0);
        }
    }
    return tmp;
}

// variance of a sample
double variance(vector<double> &sample) {
    double mean = accumulate(sample.begin(), sample.end(), 0.0)/(1.0*sample.size());
    double var = 0.0;
    for (double x : sample) {
        var += pow(x - mean,2.0);
    }
    return var/(1.0*sample.size());
}

// minimal and maximal elemnets of a sample
vector<double> findMinMax(vector<double> &sample) {
    auto result = minmax_element(sample.begin(), sample.end());
    return {*result.first, *result.second};
}
vector<double> findMinMax(vector<vector<double> > &sample2, int j) {
    vector<double> sample;
    for (auto& vec : sample2) {
        sample.push_back(vec[j]);
    }
    
    return findMinMax(sample);
}


// bin a sample
vector<vector<double> > binSample(vector<double> &sample, int N) {
    vector<double> minmax = findMinMax(sample);
    double var = variance(sample);
    
    if (var <= 0.0 || minmax[0] >= minmax[1]) {
        cout << "bad sample" << endl;
    }
    
    double dx = sqrt(var)/(1.0*(N-1));
    int Nbins = (int) ceil((minmax[1]-minmax[0])/dx + 1.0);
    vector<vector<double> > P(Nbins, vector<double> (2, 0.0));
    
    for (int j = 0; j < Nbins; j++) {
        P[j][0] = minmax[0] + j*dx;
    }
    int jb;
    for (double x : sample) {
        jb = (int) round((Nbins-1)*(x-minmax[0])/(minmax[1]-minmax[0]));
        if (jb >= 0 && jb < Nbins) {
            P[jb][1] += 1.0;
        }
    }
    return P;
}

// find the mean and P CL interval from a sample
vector<double> confidenceInterval(vector<double> &sample, double P) {
    // sort the sample
    sort(sample.begin(), sample.end());
    int n = sample.size();

    // compute indices for (1-P)/2% and (1-(1-P)/2)% percentiles
    int lowerIndex = static_cast<int>(0.5*(1-P) * n);
    int upperIndex = static_cast<int>((1-0.5*(1-P)) * n);

    // clamp indices within bounds
    lowerIndex = max(0, min(n - 1, lowerIndex));
    upperIndex = max(0, min(n - 1, upperIndex));
    
    double mean = 0.0;
    for (double x : sample) {
        mean += x;
    }
    mean = mean/(1.0*n);

    return {mean, sample[lowerIndex], sample[upperIndex]};
}

// sample N values from a cumulative density function
vector<double> sampleFromCDF(vector<vector<double> > &cdf, int N, rgen &mt) {
    double sum = cdf.back()[1];
    vector<double> ret(N);
    double r;
    for (int j = 0; j < N; j++) {
        r = sum*randomreal(0.0,1.0,mt);
        for (int i = 0; i < cdf.size(); i++) {
            if (r <= cdf[i][1]) {
                if (i == 0) {
                    ret[j] = cdf[i][0];
                } else {
                    ret[j] = cdf[i-1][0] + (cdf[i][1] - r)/(cdf[i][1] - cdf[i-1][1])*(cdf[i][0] - cdf[i-1][0]);
                }
                i = cdf.size();
            }
        }
    }
    return ret;
}

// sample N values from a probability density function
vector<double> sampleFromPDF(vector<vector<double> > &pdf, int N, rgen &mt) {
    vector<vector<double> > cdf(pdf.size(), vector<double> (2, 0.0));
    cdf[0][0] = pdf[0][0];
    double sum = 0.0;
    for (int i = 1; i < pdf.size(); i++) {
        cdf[i][0] = pdf[i][0];
        sum += pdf[i][1];
        cdf[i][1] = sum;
    }
    return sampleFromCDF(cdf, N, mt);
}

void writeToFile(vector<double> &row, const string &filename) {
    ofstream outFile(filename);
    for (int j = 0; j < row.size(); j++) {
        outFile << row[j];
        if (j < row.size() - 1) {
            outFile << endl;
        }
    }
    outFile.close();
}

void writeToFile(vector<vector<double> > &matrix, const string &filename) {
    ofstream outFile(filename);
    for (auto& row : matrix) {
        for (int j = 0; j < row.size(); j++) {
            outFile << row[j];
            if (j < row.size() - 1) {
                outFile << "   ";
            }
        }
        outFile << endl;
    }
    outFile.close();
}

void writeToFile(vector<double> &x, vector<vector<vector<double> > > &cubic, const string &filename) {
    ofstream outFile(filename);
    vector<vector<double> > matrix;
    for (int jx = 0; jx < x.size(); jx++) {
        matrix = cubic[jx];
        for (auto& row : matrix) {
            outFile << x[jx] << "   ";
            for (int j = 0; j < row.size(); j++) {
                outFile << row[j];
                if (j < row.size() - 1) {
                    outFile << "   ";
                }
            }
            outFile << endl;
        }
    }
    outFile.close();
}

void writeToFile(vector<double> &x, vector<double> &y, vector<vector<double> > &f, const string &filename) {
    ofstream outFile(filename);
    for (int jx = 0; jx < x.size(); jx++) {
        for (int jy = 0; jy < y.size(); jy++) {
            outFile << x[jx] << "   " << y[jy] << "   " << f[jx][jy] << "   " << endl;
        }
    }
    outFile.close();
}

void writeToFile(vector<double> &x, vector<double> &y, vector<vector<vector<double> > > &f, const string &filename) {
    ofstream outFile(filename);
    for (int jx = 0; jx < x.size(); jx++) {
        for (int jy = 0; jy < y.size(); jy++) {
            outFile << x[jx] << "   " << y[jy] << "   ";
            for (int j = 0; j < f[jx][jy].size(); j++) {
                outFile << f[jx][jy][j];
                outFile << "   ";
            }
            outFile << endl;
        }
    }
    outFile.close();
}

void writeToFile(vector<double> &x, vector<double> &y, vector<double> &z, vector<vector<vector<double> > > &f, const string &filename) {
    ofstream outFile(filename);
    for (int jx = 0; jx < x.size(); jx++) {
        for (int jy = 0; jy < y.size(); jy++) {
            for (int jz = 0; jz < z.size(); jz++) {
                outFile << x[jx] << "   " << y[jy] << "   " << z[jz] << "   " << f[jx][jy][jz] << endl;
            }
        }
    }
    outFile.close();
}

void writeToFile(vector<double> &x, vector<double> &y, vector<double> &z, vector<vector<vector<vector<double> > > > &f, const string &filename) {
    ofstream outFile(filename);
    for (int jx = 0; jx < x.size(); jx++) {
        for (int jy = 0; jy < y.size(); jy++) {
            for (int jz = 0; jz < z.size(); jz++) {
                outFile << x[jx] << "   " << y[jy] << "   " << z[jz] << "   ";
                for (int j = 0; j < f[jx][jy][jz].size(); j++) {
                    outFile << f[jx][jy][jz][j];
                    outFile << "   ";
                }
                outFile << endl;
            }
        }
    }
    outFile.close();
}

// read list
vector<double> readdata(fs::path filename) {
    vector<double> data;
    int jrow = 0;
    double A;
    
    ifstream infile;
    infile.open(filename);
    if (infile) {
        while (infile >> A) {
            data.push_back(A);
        }
    } else {
        cout << "couldn't open " << filename << endl;
    }
    infile.close();
    return data;
}

// read data
vector<vector<double> > readdata(fs::path filename, int N) {
    vector<vector<double> > data;
    vector<double> row(N,0.0);
    int jrow = 0;
    double A;
    
    ifstream infile;
    infile.open(filename);
    if (infile) {
        while (infile >> A) {
            row[jrow] = A;
            jrow++;
            if (jrow == N) {
                jrow = 0;
                data.push_back(row);
            }
        }
    } else {
        cout << "couldn't open " << filename << endl;
    }
    infile.close();
    return data;
}

// read data
vector<vector<double> > readdataCSV(fs::path filename) {
    vector<vector<double> > data;
    string line;
    int jrow = 0;
    double A;
    
    ifstream infile;
    infile.open(filename);
    if (infile) {
        while (getline(infile, line)) {
            stringstream ss(line);
            string token;
            vector<double> row;
            
            while (std::getline(ss, token, ',')) {
                row.push_back(stod(token));
            }
            data.push_back(row);
        }
    } else {
        cout << "couldn't open " << filename << endl;
    }
    infile.close();
    return data;
}
