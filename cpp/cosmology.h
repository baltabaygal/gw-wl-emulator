#include "basics.h"

class cosmology {

public:
    fs::path outdir;
    
    int Nk = 1000;
    int NM, Nz;
    double Mmin, Mmax, zmin, zmax;
    
    int Nm22 = 0;
    double m22min, m22max;
    
    int Nm3 = 0;
    double m3min, m3max;
    
    int Nkc = 0;
    double kcmin, kcmax;
    
    // cosmological parameters
    double OmegaM;
    double OmegaB;
    double fB;
    double zeq;
    double sigma8;
    double h;
    double T0;
    double ns;
    // primordial amplitude mode: As > 0 sets the P(k) normalization directly
    // (sigma8 is then ignored); As <= 0 keeps the original sigma8 normalization.
    double As = -1.0;
    double kpivot = 5.0e-5;      // pivot scale, comoving kpc^-1 (= 0.05 Mpc^-1)
    // derived amplitudes, filled by initialize_normalization() for either mode
    double sigma8_derived = 0.0; // via the code's smooth window Ws, not a tophat
    double As_derived = 0.0;
    double OmegaR;
    double OmegaL;
    double OmegaC;
    double H0;
    double rhoc;
    double rhoM0;
    
    double M8;
    double deltaH8;
    
    // z dependencies
    double Az(double z) {
        return OmegaM*pow(1+z,3.0) + OmegaR*pow(1+z,4.0) + OmegaL;
    }
    double Hz(double z) {
        return H0*sqrt(Az(z));
    }
    double OmegaMz(double z) {
        return OmegaM*pow(1+z,3.0)/Az(z);
    }
    double OmegaRz(double z) {
        return OmegaR*pow(1+z,4.0)/Az(z);
    }
    double OmegaLz(double z) {
        return OmegaL/Az(z);
    }
    
    // growth function (Carroll-Press-Turner). gfid = g_CPT(z=0) at the fiducial
    // cosmology (OmegaM=0.315), so Dg(0)=1 there; the same constant enters the
    // As -> deltaH8 mapping in initialize_normalization() to cancel this convention.
    static constexpr double gfid = 0.7869370293916;
    double Dg(double z) {
        return 5.0/2.0*OmegaMz(z)/(pow(OmegaMz(z),4.0/7.0) - OmegaLz(z) + (1+OmegaMz(z)/2.0)*(1+OmegaLz(z)/70.0))/(1+z)/gfid;
    }
    
    // spherical collapse threshold
    double deltac0 = 3.0/5.0*pow(3.0*PI/2.0,2.0/3.0);
    double deltac(double z) {
        return deltac0/Dg(z);
    }
    
    // elliptical collapse threshold (Eq. (1) of MNRAS 329, 61–75 (2002)), S>0
    double deltaell(double z, double S) {
        double alpha = 0.615;
        double beta = 0.485;
        double a = 0.707;
        double delta = deltac(z);

        return sqrt(a)*delta*(1.0 + beta*pow(a*pow(delta,2.0)/S,-alpha));
    }
    
private:
    
    // CDM matter transfer function (astro-ph/9709112)
    double TM(double k);
    
    // FDM matter transfer function (2201.10238, old from astro-ph/0003365)
    double kJ(double m22) {
        return 66.5e-3*pow(OmegaC*h*h/0.12/(1+zeq),1.0/4.0)*sqrt(m22);
    }
    double km22(double m22) {
        double A = 2.22*pow(m22,1.0/25.0 - 0.001*log(m22));
        return kJ(m22)/A;
    }
    double TF(double k, double m22) {
        double n = 5.0/2.0;
        double x = k/km22(m22);
        double B = 0.16*pow(m22,-1.0/20.0);
        return sin(pow(x,n))/pow(x,n)/(1+B*pow(x,6-n));
    }
    double TMF(double k, double m22) {
        return TF(k,m22)*TM(k);
    }
    
    // WDM matter transfer function (astro-ph/0501562)
    double km3(double m3) {
        double fix = 3.3*0.43; // fixes the problem with M_hm (see 1801.02547)
        return 1.0/(fix*49.0/h*pow(OmegaC/0.25,0.11)*pow(h/0.7,1.22)*pow(m3,-1.11));
    }
    double TW(double k, double m3) {
        double mu = 1.12;
        return pow(1.0+pow(k/km3(m3),2.0*mu),-5.0/mu);
        
    }
    double TMW(double k, double m3) {
        return TW(k,m3)*TM(k);
        
    }
    
    // CDM matter power spectrum
    double Deltak(double k, double deltaH) {
        return sqrt(pow(306.535*k/H0,3.0+ns)*pow(deltaH*TM(k),2.0));
    }
    double Plin(double z, double k, double deltaH) {
        return (2.0*pow(PI,2.0))*pow(Deltak(k, deltaH)*Dg(z), 2.0)/(pow(k,3.0));
    }
    
    // FDM matter power spectrum
    double DeltakF(double k, double deltaH, double m22) {
        return sqrt(pow(306.535*k/H0,3.0+ns)*pow(deltaH*TMF(k,m22),2.0));
    }
    double PlinF(double z, double k, double deltaH, double m22) {
        return (2.0*pow(PI,2.0))*pow(DeltakF(k, deltaH, m22)*Dg(z), 2.0)/(pow(k,3.0));
    }
    
    // WDM matter power spectrum
    double DeltakW(double k, double deltaH, double m3) {
        return sqrt(pow(306.535*k/H0,3.0+ns)*pow(deltaH*TMW(k,m3),2.0));
    }
    double PlinW(double z, double k, double deltaH, double m3) {
        return (2.0*pow(PI,2.0))*pow(DeltakW(k, deltaH, m3)*Dg(z), 2.0)/(pow(k,3.0));
    }
    
    // white noise enhanced matter power spectrum
    double DeltakE(double k, double deltaH, double kc) {
        return Deltak(k, deltaH) + pow(k/kc,3.0)*Deltak(kc, deltaH);
    }
    double PlinE(double z, double k, double deltaH, double kc) {
        return (2.0*pow(PI,2.0))*pow(DeltakE(k, deltaH, kc)*Dg(z), 2.0)/(pow(k,3.0));
    }
    
    vector<double> kBlist;
    vector<vector<double> > logDeltaBlist;
    
    // magnetic field enhanced matter power spectrum
    double DeltakB(double k, double deltaH, double B) {
        double kmin = kBlist.front()*h/1000.0;
        double kmax = kBlist.back()*h/1000.0;
        if (k > kmin && k < kmax) {
            return sqrt(pow(10.0, 2.0*interpolate2(B, k/h*1000.0, Blist, kBlist, logDeltaBlist)) + pow(Deltak(k, deltaH),2.0));
        }
        return Deltak(k, deltaH);
    }
    double PlinB(double z, double k, double deltaH, double B) {
        return pow(DeltakB(k, deltaH, B)*Dg(z), 2.0)/(pow(k,3.0)/(2.0*pow(PI,2.0)));
    }
    
    // real space top hat window function
    double W(double x) {
        return 3.0*(x*cos(x)-sin(x))/pow(x,3.0);
    }
    double DW(double x) {
        return -9.0*cos(x)/pow(x,3.0) + 9.0*sin(x)/pow(x,4.0) - 3.0*sin(x)/pow(x,2.0);
    }
    
    // smooth k space window function
    double Ws(double x) {
        double c = 1.0/0.43;
        double b = 6.0;
        return 1.0/(1.0+pow(x/c,b));
    }
    double DWs(double x) {
        double c = 1.0/0.43;
        double b = 6.0;
        return -b*pow(x/c,b)/(x*pow(1.0+pow(x/c,b),2.0));
    }
    
    // variance of the matter fluctuations, m22: FDM mass in 10^-22 eV, m3: WDM mass in keV
    vector<double> sigmaC(double M, double deltaH);
    vector<double> sigmaF(double M, double deltaH, double m22);
    vector<double> sigmaW(double M, double deltaH, double m3);
    vector<double> sigmaE(double M, double deltaH, double kc);
    vector<double> sigmaB(double M, double deltaH, double B);

    // variance of matter fluctuations, {M,sigma(M),sigma'(M)}
    vector<vector<double> > sigmalistf(double m22, double m3, double kc, double B);

    // CDM halo consentration parameter (1601.02624), {jz,jM} -> {c, dc/dM}
    double cons14(double z, double M);
    double cons16(double z, double sigma);
    vector<vector<vector<double> > > conslistf();
    vector<vector<vector<double> > > conslist;
    
    // NFW scale radius and density and their derivatives, {jz,jM} -> {r_s, dr_s/dM, rho_s, drho_s/dM}
    vector<vector<vector<double> > > NFWlistf();
    
    // first crossing probability with ellipsoidal collapse
    double pFC(double delta, double S);
    double pFC(double deltap, double Sp, double delta, double S);
    
    // transition rate S->S0
    double pTR(double delta, double Ddeltaell, double S, double S0);

    
    // first crossing probability for filaments
    double pFCfil(double delta, double S);

    // halo mass function and growth rate, {jz,jM} -> {dn/dlnM, dotM, dotM/dM}
    vector<vector<vector<double> > > HMFlistf();
    
    // halo merger rate, {jz,jM1,jM2} -> dR/dlnM1 dlnM2
    vector<vector<vector<double> > > HMRlistf();
    
    // halo bias, {jz,jM} -> b
    vector<vector<double> > halobiaslistf();
    vector<vector<double> > PQlistf(double z);
    
    // filament mass function and growth rate, {jz,jM} -> dn_fil/dlnM
    vector<vector<double> > FMFlistf();
    
    // characteristic mass such that sigma(M_char) = delta_c(z)
    vector<vector<double> > logMcharlistf();
    
    vector<vector<double> > dclist();
    vector<vector<double> > tlist();
    
    vector<vector<double> > zdc;
    vector<vector<double> > zt;
    
public:
    
    // halo bias b(M,z)
    double halobias(double z, double sigma);
    
    // star formation rate
    double fstar(double z, double M, double Mc, double Mt, double epsilon, double alpha, double beta);

    // derivative of the star formation rate, df_*/dM
    double Dfstarperfstar(double z, double M, double Mc, double Mt, double epsilon, double alpha, double beta);
    
    vector<vector<double> > evolvestellarmass(double Mc, double Mt, double epsilon, double alpha, double beta);
    vector<vector<double> > evolveBHmass(double Mc, double Mt, double epsilon, double alpha, double beta, double fEdd, double facc1, double facc2);
        
    // list of redshifts and halo masses
    vector<double> zlist;
    vector<double>  Mlist;
    
    vector<vector<double> > sigmalist;
    vector<vector<vector<double> > > HMFlist;
    vector<vector<vector<double> > > HMRlist;
    vector<vector<double> > halobiaslist;
    vector<vector<double> > FMFlist;
    vector<vector<double> > logMcharlist;
    vector<vector<vector<double> > > NFWlist;
    
    // list of redshifts, same as for P(lnmu,z) list
    vector<double> Zlist;
    vector<vector<vector<double> > > Plnmuz;
    
    vector<double> m22list;
    vector<vector<vector<double> > > FDMsigmalist;
    vector<vector<vector<vector<double> > > > FDMHMFlist;
    vector<vector<vector<double> > > FDMFMFlist;
    
    vector<double> m3list;
    vector<vector<vector<double> > > WDMsigmalist;
    vector<vector<vector<vector<double> > > > WDMHMFlist;
    vector<vector<vector<double> > > WDMFMFlist;
    
    vector<double> kclist;
    vector<vector<vector<double> > > EDMsigmalist;
    vector<vector<vector<vector<double> > > > EDMHMFlist;
    vector<vector<vector<double> > > EDMFMFlist;
    
    vector<double> Blist;
    vector<vector<vector<double> > > BDMsigmalist;
    vector<vector<vector<vector<double> > > > BDMHMFlist;
    vector<vector<vector<double> > > BDMFMFlist;
    
    // background quantities + power spectrum normalization only (no sigma(M)/HMF
    // tables). Cheap enough for diagnostics: at most two sigmaC integrals.
    void initialize_normalization() {
        OmegaR = OmegaM/(1+zeq);
        OmegaL = 1.0 - OmegaM - OmegaR;
        OmegaC = OmegaM - OmegaB;
        fB = OmegaB/OmegaM;

        H0 = 0.000102247*h;
        rhoc = 277.394*pow(h,2.0);
        rhoM0 = OmegaM*rhoc;
        M8 = 4.0*PI/3.0*pow(8000.0/h,3.0)*rhoM0;

        // P(k) amplitude. The code's convention is Delta^2(k,z) =
        // (ck/H0)^(3+ns) (deltaH8 TM(k))^2 Dg(z)^2 with Dg = [g_CPT/(1+z)]/gfid.
        // As-mode maps the primordial amplitude Delta_R^2 = As (k/kpivot)^(ns-1)
        // through the standard relation Delta_m^2 = (4/25) As (ck/H0)^4
        // (k/kpivot)^(ns-1) T^2 D^2/OmegaM^2 (D -> a in matter domination), giving
        //   deltaH8 = (2/5) gfid sqrt(As) (c kpivot/H0)^((1-ns)/2) / OmegaM.
        // sigma8-mode inverts sigmaC as before. NOTE sigmaC uses the smooth window
        // Ws, not a tophat, so sigma8_derived vs CAMB agrees only to a few %.
        if (As > 0.0) {
            deltaH8 = 0.4*gfid*sqrt(As)*pow(306.535*kpivot/H0,(1.0-ns)/2.0)/OmegaM;
        } else {
            deltaH8 = sigma8/sigmaC(M8, 1.0)[0];
        }
        sigma8_derived = deltaH8*sigmaC(M8, 1.0)[0];
        As_derived = pow(deltaH8*OmegaM/(0.4*gfid), 2.0)*pow(306.535*kpivot/H0, ns-1.0);
    }

    void initialize0() {

        // directory for output files
        if (!fs::exists(outdir)) {
            fs::create_directories(outdir);
        }

        initialize_normalization();

        zlist = loglist(zmin,zmax,Nz);
        Mlist = loglist(Mmin,Mmax,NM);

        zdc = dclist();
        zt = tlist();

        // NFW halo parameters, computed with CDM sigma
        sigmalist = sigmalistf(0.0, 0.0, 0.0, 0.0);
        logMcharlist = logMcharlistf();
        conslist = conslistf();
        NFWlist = NFWlistf();
        
        writeToFile(logMcharlist, outdir/"logMcharlist.dat");
        
        /*
        vector<vector<double> > Deltalist(100, vector<double> (2, 0.0));
        double dlnk = (log(1000.0)-log(0.001))/99.0;
        double ki;
        for (int jk = 0; jk < Deltalist.size(); jk++) {
            ki = exp(log(0.001) + jk*dlnk);
            Deltalist[jk][0] = ki;
            Deltalist[jk][1] = Deltak(h*ki/1000.0, deltaH8);
        }
        
        writeToFile(Deltalist, "Deltak_CDM.dat");
        */
    }
    
    // initialize for multiple beyond CDM parameter values and output the halo mass functions
    void initialize(int dm) {
        initialize0();
        
        if (dm == 0) {
            // halo mass function and halo growth rate
            HMFlist = HMFlistf();
            FMFlist = FMFlistf();
            halobiaslist = halobiaslistf();
            
            // uncomment to compute and export halo merger rate
            //HMRlist = HMRlistf();
            //writeToFile(zlist, Mlist, Mlist, HMRlist, "HMR_CDM.dat");
            
            writeToFile(sigmalist, outdir/"sigma_CDM.dat");
            writeToFile(zlist, Mlist, HMFlist, outdir/"HMF_CDM.dat");
            writeToFile(zlist, Mlist, FMFlist, outdir/"FMF_CDM.dat");
            writeToFile(zlist, Mlist, halobiaslist, outdir/"halobiaslist_CDM.dat");
        }
        if (dm == 1) {
            m22list = loglist(m22min,m22max,Nm22);
            
            for (double m22 : m22list) {
                // fix deltaH to match the input sigma8
                deltaH8 = sigma8/sigmaF(M8, 1.0, m22)[0];
                            
                // halo mass function and halo growth rate
                sigmalist = sigmalistf(m22, 0.0, 0.0, 0.0);
                HMFlist = HMFlistf();
                halobiaslist = halobiaslistf();

                FDMsigmalist.push_back(sigmalist);
                FDMHMFlist.push_back(HMFlist);
            }
            writeToFile(m22list, FDMsigmalist, outdir/"sigma_FDM.dat");
            writeToFile(m22list, zlist, Mlist, FDMHMFlist, outdir/"HMF_FDM.dat");
        }
        if (dm == 2) {
            m3list = loglist(m3min,m3max,Nm3);
            
            for (double m3 : m3list) {
                // fix deltaH to match the input sigma8
                deltaH8 = sigma8/sigmaW(M8, 1.0, m3)[0];
                            
                // halo mass function and halo growth rate
                sigmalist = sigmalistf(0.0, m3, 0.0, 0.0);
                HMFlist = HMFlistf();
                halobiaslist = halobiaslistf();

                WDMsigmalist.push_back(sigmalist);
                WDMHMFlist.push_back(HMFlist);
            }
            
            writeToFile(m3list, WDMsigmalist, outdir/"sigma_WDM.dat");
            writeToFile(m3list, zlist, Mlist, WDMHMFlist, outdir/"HMF_WDM.dat");
        }
        if (dm == 3) {
            kclist = loglist(kcmin,kcmax,Nkc);
            
            for (double kc : kclist) {
                // fix deltaH to match the input sigma8
                deltaH8 = sigma8/sigmaE(M8, 1.0, kc)[0];
                            
                // halo mass function and halo growth rate
                sigmalist = sigmalistf(0.0, 0.0, kc, 0.0);
                HMFlist = HMFlistf();
                halobiaslist = halobiaslistf();

                EDMsigmalist.push_back(sigmalist);
                EDMHMFlist.push_back(HMFlist);
            }
            
            writeToFile(kclist, EDMsigmalist, outdir/"sigma_EDM.dat");
            writeToFile(kclist, zlist, Mlist, EDMHMFlist, outdir/"HMF_EDM.dat");
        }
        if (dm == 4 || dm == 5) {
            // read Delta spectra for different B values and extract list of B and k values
            vector<vector<double> > tmp  = readdata(outdir/"DeltaBinf.dat",3);
            if (dm == 5) {
                tmp  = readdata(outdir/"DeltaBpt.dat",3);
            }
            
            set<double> xs, ys;
            for (const auto &row : tmp) {
                xs.insert(row[0]);
                ys.insert(row[1]);
            }
            Blist.assign(xs.begin(), xs.end());
            kBlist.assign(ys.begin(), ys.end());
            
            xs.clear(); ys.clear();
            map<double,int> x_index, y_index;
            for (int i = 0; i < Blist.size(); ++i) {
                x_index[Blist[i]] = i;
            }
            for (int j = 0; j < kBlist.size(); ++j) {
                y_index[kBlist[j]] = j;
            }

            vector<vector<double> > tmp2(Blist.size(), vector<double>(kBlist.size()));
            for (const auto &row : tmp) {
                double x = row[0];
                double y = row[1];
                double z = row[2];
                
                int jx = x_index[x];
                int jy = y_index[y];

                tmp2[jx][jy] = log10(z);
            }
            logDeltaBlist = tmp2;
            tmp.clear(); tmp2.clear();
            
            for (double B : Blist) {
                // fix deltaH to match the input sigma8
                deltaH8 = sigma8/sigmaC(M8, 1.0)[0];
                                
                // halo mass function and halo growth rate
                sigmalist = sigmalistf(0.0, 0.0, 0.0, B);
                HMFlist = HMFlistf();
                halobiaslist = halobiaslistf();
                
                BDMsigmalist.push_back(sigmalist);
                BDMHMFlist.push_back(HMFlist);
            }
            
            if (dm == 4) {
                writeToFile(Blist, BDMsigmalist, outdir/"sigma_Binf.dat");
                writeToFile(Blist, zlist, Mlist, BDMHMFlist, outdir/"HMF_Binf.dat");
            }
            if (dm == 5) {
                writeToFile(Blist, BDMsigmalist, outdir/"sigma_Bpt.dat");
                writeToFile(Blist, zlist, Mlist, BDMHMFlist, outdir/"HMF_Bpt.dat");
            }
        }
    }
    
    // initialize for single beyond CDM parameter value, x = m_WDM, m_FDM or k_EDM
    void initialize(int dm, double x) {
        initialize0();
        
        // compute halo mass function as a function of M and z
        if (dm == 0) {
            HMFlist = HMFlistf();
            FMFlist = FMFlistf();
            halobiaslist = halobiaslistf();
        }
        if (dm == 1) {
            deltaH8 = sigma8/sigmaF(M8, 1.0, x)[0];
            sigmalist = sigmalistf(x, 0.0, 0.0, 0.0);
            HMFlist = HMFlistf();
            halobiaslist = halobiaslistf();
        }
        if (dm == 2) {
            deltaH8 = sigma8/sigmaW(M8, 1.0, x)[0];
            sigmalist = sigmalistf(0.0, x, 0.0, 0.0);
            HMFlist = HMFlistf();
            halobiaslist = halobiaslistf();
        }
        if (dm == 3) {
            deltaH8 = sigma8/sigmaE(M8, 1.0, x)[0];
            sigmalist = sigmalistf(0.0, 0.0, x, 0.0);
            HMFlist = HMFlistf();
            halobiaslist = halobiaslistf();
        }
        if (dm == 4 || dm == 5) {
            cout << "not implemented..." << endl;
        }
    }
    
    // comoving distance
    double dc(double z) {
        return interpolate(z, zdc);
    }
    
    // comoving volume
    double Vc(double z) {
        double dc = interpolate(z, zdc);
        return 4.0*PI/3.0*pow(dc,3.0);
    }
    
    // derivative of the comoving volume
    double DVc(double z) {
        double dc = interpolate(z, zdc);
        return 4.0*PI*pow(dc,2.0)/Hz(z);
    }
    
    // luminosity distance
    double DL(double z) {
        return (1+z)*interpolate(z, zdc);
    }
    
    // age of the universe
    double age(double z) {
        return interpolate(z, zt);
    }
};
