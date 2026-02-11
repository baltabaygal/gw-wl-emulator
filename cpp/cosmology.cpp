#include "cosmology.h"

// CDM matter transfer function (astro-ph/9709112)
double cosmology::TM (double k) {
    double keq = 0.00326227*Hz(zeq)/(1.0+zeq);
    double ksilk = 0.0016*pow(OmegaB*pow(h,2.0),0.52)*pow(OmegaM*pow(h,2.0),0.73)*(1+pow(10.4*OmegaM*pow(h,2.0),-0.95));
    double a1 = pow(46.9*OmegaM*pow(h,2.0),0.67)*(1+pow(32.1*OmegaM*pow(h,2.0),-0.532));
    double a2 = pow(12.0*OmegaM*pow(h,2.0),0.424)*(1+pow(45.0*OmegaM*pow(h,2.0),-0.582));
    double b1 = 0.944/(1+pow(458*OmegaM*pow(h,2.0),-0.708));
    double b2 = pow(0.395*OmegaM*pow(h,2.0),-0.026);
    double acnum = pow(a1,-OmegaB/OmegaM)*pow(a2,-pow(OmegaB/OmegaM,3.0));
    double bcnum = 1.0/(1+b1*(pow(OmegaC/OmegaM,b2)-1));
    double s2 = 44.5*1000*log(9.83/(OmegaM*pow(h,2.0)))/sqrt(1+10*pow(OmegaB*pow(h,2.0),3.0/4.0));
    double b3 = 0.313*pow(OmegaM*pow(h,2.0),-0.419)*(1+0.607*pow(OmegaM*pow(h,2.0),0.674));
    double b4 = 0.238*pow(OmegaM*pow(h,2.0),0.223);
    double zd = 1291*pow(OmegaM*pow(h,2.0),0.251)/(1+0.659*pow(OmegaM*pow(h,2.0),0.828))*(1+b3*pow(OmegaB*pow(h,2.0),b4));
    double Rd = 31.5*OmegaB*pow(h,2.0)*pow(T0/2.7,-4.0)/(zd/1000);
    function<double(double)> g2 = [&](double y) {
        return y*(-6*sqrt(1+y) + (2.0+3.0*y)*log((sqrt(1+y)+1)/(sqrt(1+y)-1)));
    };
    double ab = 2.07*keq*s2*pow(1+Rd,-3.0/4.0)*g2((1+zeq)/(1+zd));
    double bb = 0.5 + OmegaB/OmegaM + (3.0-2.0*OmegaB/OmegaM)*sqrt(1+pow(17.2*OmegaM*pow(h,2.0),2.0));
    double bnode = 8.41*pow(OmegaM*pow(h,2.0),0.435);
    function<double(double)> qk = [&](double k) {
        return k/(13.41*keq);
    };
    function<double(double)> fk = [&](double k) {
        return 1.0/(1+pow(k*s2/5.4,4.0));
    };
    function<double(double, double)> C1 = [&](double k, double ac) {
        return 14.2/ac + 386.0/(1+69.9*pow(qk(k),1.08));
    };
    function<double(double, double, double)> To1 = [&](double k, double ac, double bc) {
        return log(exp(1.0)+1.8*bc*qk(k))/(log(exp(1.0)+1.8*bc*qk(k))+C1(k,ac)*pow(qk(k),2.0));;
    };
    function<double(double)> jo = [&](double x) {
        return sin(x)/x;
    };
    function<double(double)> s3 = [&](double k) {
        return s2/pow(1+pow(bnode/(k*s2),3.0),1.0/3.0);
    };
    function<double(double)> TC = [&](double k) {
        return fk(k)*To1(k,1.0,bcnum)+(1-fk(k))*To1(k,acnum,bcnum);
    };
    function<double(double)> TB = [&](double k) {
        return (To1(k,1.0,1.0)/(1+pow(k*s2/5.2,2.0)) + ab/(1+pow(bb/(k*s2),3.0))*exp(-pow(k/ksilk,1.4)))*jo(k*s3(k));
    };
    return OmegaB/OmegaM*TB(k) + OmegaC/OmegaM*TC(k);
}


// variance of the CDM matter fluctuations
vector<double> cosmology::sigmaC(double M, double deltaH) {
    double RM = pow(3.0*M/(4.0*PI*rhoM0),1.0/3.0);
    double DRM = RM/(3.0*M);
    
    double kmax = 1000.0/RM;
    double kmin = 1.0e-6*kmax;
    double dlogk = (log(kmax)-log(kmin))/(1.0*(Nk-1));
    
    double sigma2 = 0.0, dsigma2 = 0.0;
    double k1, k2 = kmin;
    for (int jk = 0; jk < Nk; jk++) {
        k1 = k2;
        k2 = exp(log(k2)+dlogk);
        sigma2 += (k2-k1)*(pow(Ws(k1*RM)*Deltak(k1,deltaH),2.0)/k1 + pow(Ws(k2*RM)*Deltak(k2,deltaH),2.0)/k2)/2.0;
        dsigma2 += (k2-k1)*(2.0*k2*DRM*DWs(k2*RM)*Ws(k2*RM)*pow(Deltak(k2,deltaH),2.0)/k2 + 2.0*k1*DRM*DWs(k1*RM)*Ws(k1*RM)*pow(Deltak(k1,deltaH),2.0)/k1)/2.0;
    }
    return {sqrt(sigma2), dsigma2/(2.0*sqrt(sigma2))};
}


// variance of the FDM matter fluctuations
vector<double> cosmology::sigmaF(double M, double deltaH, double m22) {
    double RM = pow(3.0*M/(4.0*PI*rhoM0),1.0/3.0);
    double DRM = RM/(3.0*M);
    
    double kmax = min(1000.0/RM, 10.0*km22(m22));
    double kmin = 1.0e-6*kmax;
    double dlogk = (log(kmax)-log(kmin))/(1.0*(Nk-1));
    
    double sigma2 = 0.0, dsigma2 = 0.0;
    double k1, k2 = kmin;
    for (int jk = 0; jk < Nk; jk++) {
        k1 = k2;
        k2 = exp(log(k2)+dlogk);
        sigma2 += (k2-k1)*(pow(Ws(k1*RM)*DeltakF(k1,deltaH,m22),2.0)/k1 + pow(Ws(k2*RM)*DeltakF(k2,deltaH,m22),2.0)/k2)/2.0;
        dsigma2 += (k2-k1)*(2.0*k2*DRM*DWs(k2*RM)*Ws(k2*RM)*pow(DeltakF(k2,deltaH,m22),2.0)/k2 + 2.0*k1*DRM*DWs(k1*RM)*Ws(k1*RM)*pow(DeltakF(k1,deltaH,m22),2.0)/k1)/2.0;
    }
    return {sqrt(sigma2), dsigma2/(2.0*sqrt(sigma2))};
}


// variance of the WDM matter fluctuations
vector<double> cosmology::sigmaW(double M, double deltaH, double m3) {
    double RM = pow(3.0*M/(4.0*PI*rhoM0),1.0/3.0);
    double DRM = RM/(3.0*M);
    
    double kmax = min(1000.0/RM, 10.0*km3(m3));
    double kmin = 1.0e-6*kmax;
    double dlogk = (log(kmax)-log(kmin))/(1.0*(Nk-1));
    
    double sigma2 = 0.0, dsigma2 = 0.0;
    double k1, k2 = kmin;
    for (int jk = 0; jk < Nk; jk++) {
        k1 = k2;
        k2 = exp(log(k2)+dlogk);
        sigma2 += (k2-k1)*(pow(Ws(k1*RM)*DeltakW(k1,deltaH,m3),2.0)/k1 + pow(Ws(k2*RM)*DeltakW(k2,deltaH,m3),2.0)/k2)/2.0;
        dsigma2 += (k2-k1)*(2.0*k2*DRM*DWs(k2*RM)*Ws(k2*RM)*pow(DeltakW(k2,deltaH,m3),2.0)/k2 + 2.0*k1*DRM*DWs(k1*RM)*Ws(k1*RM)*pow(DeltakW(k1,deltaH,m3),2.0)/k1)/2.0;
    }
    return {sqrt(sigma2), dsigma2/(2.0*sqrt(sigma2))};
}

// variance of the white noise enhanced matter fluctuations
vector<double> cosmology::sigmaE(double M, double deltaH, double kc) {
    double RM = pow(3.0*M/(4.0*PI*rhoM0),1.0/3.0);
    double DRM = RM/(3.0*M);
    
    double kmax = 1000.0/RM;
    double kmin = 1.0e-6*kmax;
    double dlogk = (log(kmax)-log(kmin))/(1.0*(Nk-1));
    
    double sigma2 = 0.0, dsigma2 = 0.0;
    double k1, k2 = kmin;
    for (int jk = 0; jk < Nk; jk++) {
        k1 = k2;
        k2 = exp(log(k2)+dlogk);
        sigma2 += (k2-k1)*(pow(Ws(k1*RM)*DeltakE(k1,deltaH,kc),2.0)/k1 + pow(Ws(k2*RM)*DeltakE(k2,deltaH,kc),2.0)/k2)/2.0;
        dsigma2 += (k2-k1)*(2.0*k2*DRM*DWs(k2*RM)*Ws(k2*RM)*pow(DeltakE(k2,deltaH,kc),2.0)/k2 + 2.0*k1*DRM*DWs(k1*RM)*Ws(k1*RM)*pow(DeltakE(k1,deltaH,kc),2.0)/k1)/2.0;
    }
    return {sqrt(sigma2), dsigma2/(2.0*sqrt(sigma2))};
}

// variance of the magnetic field enhanced matter fluctuations
vector<double> cosmology::sigmaB(double M, double deltaH, double B) {
    double RM = pow(3.0*M/(4.0*PI*rhoM0),1.0/3.0);
    double DRM = RM/(3.0*M);
    
    double kmax = 1000.0/RM;
    double kmin = 1.0e-6*kmax;
    double dlogk = (log(kmax)-log(kmin))/(1.0*(Nk-1));
    
    double sigma2 = 0.0, dsigma2 = 0.0;
    double k1, k2 = kmin;
    for (int jk = 0; jk < Nk; jk++) {
        k1 = k2;
        k2 = exp(log(k2)+dlogk);
        sigma2 += (k2-k1)*(pow(Ws(k1*RM)*DeltakB(k1,deltaH,B),2.0)/k1 + pow(Ws(k2*RM)*DeltakB(k2,deltaH,B),2.0)/k2)/2.0;
        dsigma2 += (k2-k1)*(2.0*k2*DRM*DWs(k2*RM)*Ws(k2*RM)*pow(DeltakB(k2,deltaH,B),2.0)/k2 + 2.0*k1*DRM*DWs(k1*RM)*Ws(k1*RM)*pow(DeltakB(k1,deltaH,B),2.0)/k1)/2.0;
    }
    return {sqrt(sigma2), dsigma2/(2.0*sqrt(sigma2))};
}


// list of matter fluctuation variances, {M, sigma, dsigma/dM}
vector<vector<double> > cosmology::sigmalistf(double m22, double m3, double kc, double B) {
    int Nextra = (int) ceil((NM-1)*log(3.0)/log(Mmax/Mmin)); // extend the M range for computation of dotM
    vector<vector<double> > Ms(NM+Nextra, vector<double> (3,0.0));
    
    double dlogM = (log(Mmax)-log(Mmin))/(1.0*(NM-1));
    double M = exp(log(Mmin) - dlogM);
        
    vector<double> sigma;
    for (int jM = 0; jM < NM+Nextra; jM++) {
        M = exp(log(M) + dlogM);
        if (m22 == 0.0 && m3 == 0.0) {
            sigma = sigmaC(M, deltaH8);
        }
        if (m22 > 0.0) {
            sigma = sigmaF(M, deltaH8, m22);
        }
        if (m3 > 0.0) {
            sigma = sigmaW(M, deltaH8, m3);
        }
        if (kc > 0.0) {
            sigma = sigmaE(M, deltaH8, kc);
        }
        if (B > 0.0) {
            sigma = sigmaB(M, deltaH8, B);
        }
        Ms[jM][0] = M;
        Ms[jM][1] = sigma[0];
        Ms[jM][2] = sigma[1];
    }
    return Ms;
}

// first crossing probability with ellipsoidal collapse
double cosmology::pFC(double delta, double S) {
    double p = 0.3;
    double q = 0.8;
    double A = 1.0/(1+pow(2.0,-p)*tgammaf(0.5-p)/sqrt(PI));
    
    double nu2 = 0.0;
    if (S > 0.0) {
        nu2 = pow(delta,2.0)/S;
    }
    if (nu2 > 0.0) {
        return A*(1+pow(q*nu2,-p))*sqrt(q*nu2/(2.0*PI))*exp(-q*nu2/2.0)/S;
    }
    return 0.0;
}

// halo mass function and growth rate, {jz,jM} -> {dn/dlnM, dotM, dotM/dM}
vector<vector<vector<double> > > cosmology::HMFlistf() {
    vector<vector<vector<double> > > dndlnM(Nz, vector<vector<double> > (NM, vector<double> (3,0.0)));
    
    double dz = 0.01; // z step over which DeltaM is computed
    double z, M, Mp1, S, S2, Ddeltaell, dMdt, dMdtp1;
    vector<double> sigma, sigma2;
    for (int jz = 0; jz < Nz; jz++) {
        z = zlist[jz];
        
        sigma = sigmalist[0];
        S = pow(sigma[1],2.0);
        Ddeltaell = (deltaell(z+dz,S)-deltaell(z,S))/dz;
        M = sigma[0];
        sigma2 = interpolaten(2.0*M, sigmalist);
        S2 = pow(sigma2[1],2.0);
        dMdt = (1+z)*Hz(z)*sqrt(2.0/PI)*Ddeltaell/abs(2.0*sigma[1]*sigma[2])*sqrt(S-S2);
        
        for (int jM = 0; jM < NM; jM++) {
            dndlnM[jz][jM][0] = -rhoM0*pFC(deltac(z),pow(sigma[1],2.0))*2.0*sigma[1]*sigma[2];
            
            sigma = sigmalist[jM+1];
            S = pow(sigma[1],2.0);
            
            Ddeltaell = (deltaell(z+dz,S)-deltaell(z,S))/dz;
            
            Mp1 = sigma[0];
            sigma2 = interpolaten(2.0*Mp1, sigmalist);
            S2 = pow(sigma2[1],2.0);
            dMdtp1 = (1+z)*Hz(z)*sqrt(2.0/PI)*Ddeltaell/abs(2.0*sigma[1]*sigma[2])*sqrt(S-S2);
            
            dndlnM[jz][jM][1] = dMdt;
            dndlnM[jz][jM][2] = (dMdtp1 - dMdt)/(Mp1 - M);
            
            M = Mp1;
            dMdt = dMdtp1;
        }
    }
    return dndlnM;
}

// transition rate S->S0
double cosmology::pTR(double delta, double Ddeltaell, double S, double S0) {
    double p = 0.3;
    double q = 0.8;
    double a = 0.707;
    
    double nu = pow(delta,2.0)/S;
    double nu0 = pow(delta,2.0)/S0;
    
    double Ap = 1.0/(1+pow(a*nu0,-p));
    
    return Ap/sqrt(2.0*PI)*(1.0+pow(a*nu,-p))*(1.0+pow(q*nu0,-p))/(1.0+pow(q*nu,-p))*pow(S/(S0*(S-S0)),3.0/2.0)*exp(-q*(nu0-nu)/2.0);
}

// halo merger rate, {jz,jM1,jM2} -> dR/dlnM1 dlnM2
vector<vector<vector<double> > > cosmology::HMRlistf() {
    vector<vector<vector<double> > > dRdlnM1dlnM2(Nz, vector<vector<double> > (NM, vector<double> (NM, 0.0)));
    
    double dz = 0.01; // z step over which DeltaM is computed
    double z, M1, M2, Mb, Mf, S1, S2, Sb, Sf, dndlnMb, Ddeltaell;
    int jMb;
    vector<double> sigma1, sigma2, sigmaf;
    for (int jz = 0; jz < Nz; jz++) {
        z = zlist[jz];

        for (int jM1 = 0; jM1 < NM; jM1++) {
            sigma1 = sigmalist[jM1];
            M1 = sigma1[0];
            S1 = pow(sigma1[1],2.0);
            
            for (int jM2 = 0; jM2 < NM; jM2++) {
                sigma2 = sigmalist[jM2];
                M2 = sigma2[0];
                S2 = pow(sigma2[1],2.0);
                
                jMb = min(jM1, jM2);
                Mb = min(M1, M2);
                Sb = max(S1, S2);
                dndlnMb = HMFlist[jz][jMb][0];
                
                Mf = M1 + M2;
                sigmaf = interpolaten(Mf, sigmalist);
                Sf = pow(sigmaf[1],2.0);
                Ddeltaell = (deltaell(z+dz,Sf)-deltaell(z,Sf))/dz;
                
                dRdlnM1dlnM2[jz][jM1][jM2] = dndlnMb*(1+z)*Hz(z)*abs(2.0*sigmaf[1]*sigmaf[2])*Mf*pTR(deltac(z), Ddeltaell, Sb, Sf);
            }
        }
    }
    return dRdlnM1dlnM2;
}

// first crossing probability for filaments
double cosmology::pFCfil(double delta, double S) {
    double p = 0.0;
    double q = 0.7;
    double A = 1.0/(1+pow(2.0,-p)*tgammaf(0.5-p)/sqrt(PI));

    double nu2 = 0.0;
    if (S > 0.0) {
        nu2 = pow(delta,2.0)/S;
    }
    if (nu2 > 0.0) {
        return A*(1+pow(q*nu2,-p))*sqrt(q*nu2/(2.0*PI))*exp(-q*nu2/2.0)/S;
    }
    return 0.0;
}

// filament mass function, {jz,jM} -> dn_fil/dlnM
vector<vector<double> > cosmology::FMFlistf() {
    
    double dz = 0.01; // z step over which DeltaM is computed
    
    double dlogz = (log(zmax)-log(zmin))/(1.0*(Nz-1));
    vector<vector<double> > dndlnM(Nz, vector<double> (NM, 0.0));
    
    double z, M;
    vector<double> sigma, sigma2;
    for (int jz = 0; jz < Nz; jz++) {
        z = zlist[jz];
        sigma = sigmalist[0];
        for (int jM = 0; jM < NM; jM++) {
            dndlnM[jz][jM] = -rhoM0*pFCfil(deltac(z),pow(sigma[1],2.0))*2.0*sigma[1]*sigma[2];
            sigma = sigmalist[jM+1];
        }
    }
    return dndlnM;
}


/*
double Q(double S, double St, double z) {
    double dz;
    double Ddeltac = (deltaell(z+dz,S)-deltaell(z,S))/dz;
    return (pow(2.0,-p)*exp(q*pow(deltac(z),2.0)/(2.0*S))*pow(q,-0.5+p)*pow(S/(S-St),1.5)*(pow(2.0,p)*sqrt(PI)+Gammaf(0.5-p))*Mt*pow(deltac(z),-1.0+2.0*p)*(1.0+pow(a*pow(deltac(z),2.0)/S,p))*Ddeltac/(sqrt(PI)*rhoM0*(pow(S,p)+pow(a*pow(deltac(z),2.0),p))*(1.0+(q*pow(deltac(z),2.0),p))));
}
*/

// star formation rate f_*(M) and its derivative df_*/dM
double cosmology::fstar(double z, double M, double Mc, double Mt, double epsilon, double alpha, double beta) {
    double Mtz = Mt*pow((1.0+z)/10.0,-3.0/2.0);
    if (alpha>0.0 && beta>0.0) {
        return epsilon*(alpha+beta)/(beta*pow(M/Mc,-alpha) + alpha*pow(M/Mc,beta))*exp(-Mtz/M);
    }
    return epsilon*exp(-Mtz/M);
}
double cosmology::Dfstarperfstar(double z, double M, double Mc, double Mt, double epsilon, double alpha, double beta) {
    double Mtz = Mt*pow((1.0+z)/10.0,-3.0/2.0);
    if (alpha>0.0 && beta>0.0) {
        return beta*((alpha+beta)/(alpha*pow(M/Mc,alpha+beta)+beta) - 1.0)/M + Mtz/pow(M,2.0);
    }
    return Mtz/pow(M,2.0);
}

// halo concentration parameter (1402.7073)
double cosmology::cons14(double z0, double M) {
    double a = 0.520 + (0.905-0.520)*exp(-0.617*pow(z0,1.21));
    double b = -0.101 + 0.026*z0;
    
    return pow(10.0, a + b*log10(M/(1.0e12/h)));
}

// halo concentration parameter (1601.02624)
double cosmology::cons16(double z0, double sigma) {
    
    double z = z0;
    if (z0 > 7) {
        z = 7.0;
    }
    
    double c0 = 3.395*pow(1+z,-0.215);
    double beta = 0.307*pow(1+z,0.540);
    double gamma1 = 0.628*pow(1+z,-0.047);
    double gamma2 = 0.317*pow(1+z,-0.893);
    double nu0 = 4.135 - 0.564*(1+z) - 0.210*pow(1+z,2.0) + 0.0557*pow(1+z,3.0) - 0.00348*pow(1+z,4.0);
    
    double nu = deltac(z)/sigma;
    
    return c0*pow(nu/nu0,-gamma1)*pow(1+pow(nu/nu0,1.0/beta),-beta*(gamma2-gamma1));
}

// halo concentration parameter, {jz,jM} -> {c, dc/dM}
vector<vector<vector<double> > > cosmology::conslistf() {
    vector<vector<vector<double> > > zMc(Nz, vector<vector<double> > (NM, vector<double> (2,0.0)));
    double z, c, cn, M, Mn;
    for (int jz = 0; jz < Nz; jz++) {
        z = zlist[jz];
        M = sigmalist[0][0];
        
        c = cons14(z, M);
        //c = cons16(z, sigmalist[0][1]);
        for (int jM = 0; jM < NM; jM++) {
            Mn = sigmalist[jM+1][0];
            
            cn = cons14(z, Mn);
            // cn = cons16(z, sigmalist[jM+1][1]);
            
            zMc[jz][jM][0] = c;
            zMc[jz][jM][1] = (cn - c)/(Mn - M);
            
            M = Mn;
            c = cn;
        }
    }
    return zMc;
}


// NFW scale radius and density and their derivatives, {jz, jM} -> {r_s, rho_s}
vector<vector<vector<double> > > cosmology::NFWlistf() {
    vector<vector<vector<double> > > NFWparams(Nz, vector<vector<double> > (NM, vector<double> (3,0.0)));
    
    double z, M, c, Dc, r200, rs, rhos;
    vector<double> zMc(2,0.0);
    for (int jz = 0; jz < Nz; jz++) {
        z = zlist[jz];
        for (int jM = 0; jM < NM; jM++) {
            M = Mlist[jM];
            
            zMc = conslist[jz][jM];
            c = zMc[0];
            
            r200 = pow(3.0*M/(4.0*PI*200*rhoc),1.0/3.0);
            rs =  r200/c;
            rhos = 200*rhoc*pow(c,3.0)*(1+c)/(3.0*((1+c)*log(1+c) - c));
            
            /*
            Dc = zMc[1];
            Dr200 = 3.0/(4.0*PI*200*rhoc)*pow(3.0*M/(4.0*PI*200*rhoc),-2.0/3.0);
            Drs =  Dr200/c - r200*Dc/pow(c,2.0);
            Drhos = Dc*pow(c,2.0)*(-c*(3+4*c) + 3*pow(1+c,2.0)*log(1+c))/(3.0*pow(c-(1+c)*log(1+c),2.0));
            */
            
            NFWparams[jz][jM][0] = rs;
            NFWparams[jz][jM][1] = rhos;
            NFWparams[jz][jM][2] = c;
        }
    }
    return NFWparams;
}


// Fourier transform of the NFW density profile
double rhokNFW(double k, double rs, double rhos, double c) {
    double x = rs*k;
    return 4*PI*pow(rs, 3.0)*rhos*(cos(x)*(-gsl_sf_Ci(x) + gsl_sf_Ci(x + c*x)) + sin(x)*(-gsl_sf_Si(x) + gsl_sf_Si(x + c*x)) - sin(c*x)/(x + c*x));
}


// halo bias, see Baumann (5.132)
double cosmology::halobias(double z, double sigma) {

    double p = 0.3;
    double q = 0.75;
    double qnu2 = q*pow(deltac(z)/sigma,2.0);
    
    return 1.0 + (qnu2-1.0)/deltac0 + 2.0*p/(deltac0*(1.0+pow(qnu2, p)));
}

// halo bias, {jz,jM} -> b
vector<vector<double> > cosmology::halobiaslistf() {
    vector<vector<double> > B(Nz, vector<double> (NM, 0.0));

    double z;
    for (int jz = 0; jz < Nz; jz++) {
        z = zlist[jz];
        for (int jM = 0; jM < NM; jM++) {
            B[jz][jM] = halobias(z, sigmalist[jM][1]);
        }
    }
    return B;
}

vector<vector<double> > cosmology::PQlistf(double z) {
    double kmax = 1.0;
    double kmin = 1.0e-6;
    double dlogk = (log(kmax)-log(kmin))/(1.0*(Nk-1));
    vector<vector<double> > PQ(Nk, vector<double> (3, 0.0));
    
    int jz = lower_bound(zlist.begin(), zlist.end(), z)- zlist.begin();
    if (jz > 0 && zlist[jz]-z > z-zlist[jz-1]) {
        jz--;
    }
    
    double dlogM = log(Mlist[1]) - log(Mlist[0]);
    
    vector<double> NFWpar;
    double M, k, rs, rhos, c, HMF, B, bnrho, bnM;
    for (int jk = 0.0; jk < Nk; jk++) {
        k = exp(log(kmin) + jk*dlogk);
        
        bnrho = 0.0;
        bnM = 0.0;
        for (int jM = 0; jM < NM; jM++) {
            M = Mlist[jM];
            
            NFWpar = NFWlist[jz][jM];
            rs = NFWpar[0];
            rhos = NFWpar[1];
            c = NFWpar[2];
            
            HMF = HMFlist[jz][jM][0];
            B = halobiaslist[jz][jM];
                                    
            bnrho += HMF*B*rhokNFW(k, rs, rhos, c)*dlogM;
            bnM += HMF*B*M*dlogM;
        }
        PQ[jk][0] = k;
        PQ[jk][1] = Plin(z, k, deltaH8)*pow(bnrho/bnM, 2.0);
        PQ[jk][2] = Plin(z, k, deltaH8);
    }
    
    return PQ;
}


// comoving distance, {z,d_c}
vector<vector<double> > cosmology::dclist() {
    int Nz2 = 100*Nz;
    
    double dlogz = (log(zmax)-log(zmin))/(1.0*(Nz2-2));
    double z1 = 0.0, z2 = zmin;
    vector<double> d0 = {0.0, 0.0};
    
    vector<vector<double> > dlist(Nz2, vector<double> (2));
    dlist[0] = d0;
    
    for (int jz = 1; jz < Nz2; jz++) {
        d0[0] = z2;
        d0[1] += (z2-z1)*306.535*exp((log(1.0/Hz(z2))+log(1.0/Hz(z1)))/2.0);
        dlist[jz] = d0;
        
        z1 = z2;
        z2 = exp(log(z2) + dlogz);
    }
    return dlist;
}


// age of the Universe, {z,t}
vector<vector<double> > cosmology::tlist() {
    int Nz2 = 100*Nz;
    
    double dlogz = (log(zmax)-log(zmin))/(1.0*(Nz2-2));
    double z1 = 0.0, z2 = zmin;
    vector<double> d0 = {0.0, 0.0};
    
    vector<vector<double> > tlist(Nz2, vector<double> (2));
    tlist[0] = d0;
    
    for (int jz = 1; jz < Nz2; jz++) {
        d0[0] = z2;
        d0[1] += (z2-z1)*exp((log(1.0/((1+z2)*Hz(z2)))+log(1.0/((1+z1)*Hz(z1))))/2.0);
        
        tlist[jz] = d0;
        
        z1 = z2;
        z2 = exp(log(z2)+dlogz);
    }
    
    double tmax = tlist[Nz2-1][1];
    for (int jz = 0; jz < Nz2; jz++) {
        tlist[jz][1] = tmax - tlist[jz][1];
    }
    
    return tlist;
}


// characteristic mass such that sigma(M_char) = delta_c(z)
vector<vector<double> > cosmology::logMcharlistf() {
    vector<vector<double> > logMcharlist(Nz);
    
    double z, deltaz, M, logMmin, logMmax;
    for (int jz = 0; jz < Nz; jz++) {
        z = zlist[jz];
        deltaz = deltac(z);
        
        logMmin = -1.0;
        logMmax = 15.0;
        M = pow(10.0, (logMmax+logMmin)/2.0);
        while (logMmax - logMmin > 0.01) {
            if (sigmaC(M, deltaH8)[0] > deltaz) {
                logMmin = (logMmax+logMmin)/2.0;
            } else {
                logMmax = (logMmax+logMmin)/2.0;
            }
            M = pow(10.0, (logMmax+logMmin)/2.0);
        }
        
        logMcharlist[jz] = {z, log(M)};
    }
    return logMcharlist;
}
