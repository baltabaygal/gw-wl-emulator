# Deconstruction of Methods to Derive One-Point Lensing Statistics: Markdown Extraction

## 1. Exact Definition of the Correlated Weak Component $\kappa_L$

### Lensing Convergence (Born Approximation)
* **Equation (Page 3, Eq. 6):**
  $$\kappa(z_s) = \int_0^{r_s} dr \rho_{M0} G(r, r_s) \delta_M(\mathbf{r}, t(r))$$
* **Symbols definition:**
  * $\kappa$: Lensing convergence
  * $z_s$: Redshift of the source
  * $r$: Comoving radius along the line of sight
  * $r_s = r(z_s)$: Comoving position of the source at redshift $z_s$
  * $\rho_{M0}$: Matter density today
  * $G(r, r_s)$: Lensing efficiency of inhomogeneity at the comoving radius $r$
  * $\delta_M(\mathbf{r}, t(r))$: Matter density contrast at position $\mathbf{r}$ and time $t(r)$
  * $t(r)$: Geodesic time for the background FLRW model

### Lensing Efficiency Kernel
* **Equation (Page 3, Eq. 7):**
  $$G(r, r_s) = \frac{4\pi G}{c^2 a} \frac{f_k(r) f_k(r_s - r)}{f_k(r_s)}$$
* **Symbols definition:**
  * $G$: Gravitational constant (in numerator of the fraction)
  * $c$: Speed of light
  * $a$: Scale factor
  * $f_k(r)$: Comoving angular diameter distance, defined as $\sin(r\sqrt{k})/\sqrt{k}$, $r$, or $\sinh(r\sqrt{-k})/\sqrt{-k}$ depending on whether the curvature is $k > 0$, $k = 0$, or $k < 0$, respectively.

### Density Field Fourier Decomposition
* **Equation (Page 3, Eq. 8):**
  $$\delta_M(\mathbf{r}, t) = \int \frac{d^3k}{(2\pi)^3} e^{i\mathbf{k}\cdot\mathbf{r}} \delta_M(\mathbf{k}, t)$$
* **Symbols definition:**
  * $\mathbf{k}$: Wavenumber vector
  * $k = |\mathbf{k}|$: Wavenumber
  * $\delta_M(\mathbf{k}, t)$: Fourier mode of the density contrast field at time $t$

### Convergence Decomposition
* **Equation (Page 3, Eq. 9):**
  $$\kappa(z) = \kappa_L(z) + \kappa_H(z)$$
* **Symbols definition:**
  * $\kappa_L(z)$: Correlated weak (2-halo/large-scale) component of convergence
  * $\kappa_H(z)$: Halo/Poisson (1-halo/small-scale) component of convergence

### Variance of the 2-Halo Convergence PDF ($\sigma_{\kappa_L}^2$)
* **Equation (Page 3, Eq. 10):**
  $$\sigma_{\kappa_L}^2 = \int_0^{r_s} dr \rho_{M0}^2 G^2(r, r_s) \int_0^{k_L} \frac{k \, dk}{2\pi} P_L(k, z(r))$$
* **Symbols definition:**
  * $\sigma_{\kappa_L}^2$: Variance of the 2-halo (correlated weak) convergence component
  * $k_L$: Cut-off scale separating the 2-halo contribution (larger scales) from the 1-halo term (smaller scales)
  * $P_L(k, z(r))$: Linear power spectrum at wavenumber $k$ and redshift $z(r)$ corresponding to comoving distance $r$

### Linear Power Spectrum
* **Equation (Page 3, Eq. 2):**
  $$P_L(k, z) = \frac{2\pi^2}{k^3} \delta_{H0}^2 \left( \frac{ck}{H_0} \right)^{3+n_s} T^2(k) D^2(z)$$
* **Symbols definition:**
  * $H_0$: Present-day Hubble parameter
  * $n_s$: Spectral index
  * $\delta_{H0}$: Amplitude of perturbations on the horizon scale today
  * $T(k)$: Transfer function
  * $D(z)$: Growth function

---

## 2. Calibration of $k_L$

* **Formula (Page 3, Section 2.1.3):**
  $$k_L = \exp(3.9 - 4.6z) \, \text{Mpc}^{-1}$$
* **Fitted Values:**
  * $a = 3.9$
  * $b = 4.6$
  (in the parameterized form $k_L = \exp(a - bz)$)
* **Calibration Data:**
  The cutoff scale $k_L$ was established using a pragmatic fitting approach. It was adjusted so that the lensing PDF derived from the `turboGL` (`tgl`) code matches the PDF obtained from $\text{PINOCCHIO}_{\text{halos}}$ (which runs ray-tracing on the standard PINOCCHIO past light cone but randomizes all particles outside of halos). 
  
  The underlying cosmology for these simulations is a flat $\Lambda$CDM model calibrated against DES Y1 + Planck 2015 + JLA SNe + BAO (Abbott et al. 2018), with parameters:
  $$\Omega_{m0} = 0.301, \quad \Omega_{b0} = 0.048, \quad \Omega_{\Lambda0} = 0.699, \quad h = 0.682, \quad \sigma_8 = 0.798, \quad n_s = 0.973$$
  The simulations use a comoving box size of $L_{\text{box}} = 150 \, \text{Mpc}/h$ with $N_{\text{part}} = 1024^3$ dark matter particles (individual particle mass $m_{\text{DM}} = 2.6 \times 10^8 \, M_{\odot}/h$).

---

## 3. Statistical Relation Between $\kappa_L$ and $\kappa_H$

### Treatment of the Relation
The paper assumes statistical independence between the correlated weak component ($\kappa_L$) and the halo component ($\kappa_H$). This independence is implemented by convolving the 1-halo and 2-halo PDFs to obtain the final convergence PDF.

### Quote on Convolution Method
* **Page 2, Section 1:**
  > "This method calculates the 1-halo term of the lensing PDF using stochastic configurations of inhomogeneities based on a halo model and subsequently incorporates the 2-halo term through convolution."

* **Page 4, Section 2.1.5:**
  > "The full convergence PDF is finally obtained by convolving the 1- and 2-halo PDFs. The former is obtained via a histogram of a (Poissonian) realization; the latter is based on the log-normal template previously discussed."

### Quantifying Neglected Count-Environment Covariance
The paper does not discuss or quantify the neglected count-environment covariance (the covariance between the halo numbers/counts and the background large-scale environment). It assumes complete independence between $\kappa_L$ and $\kappa_H$ through convolution, and no quantification of the coupling or covariance is provided.

---

## 4. Per-Mass and Per-Redshift Decomposition

### Key Physical Insights (Page 8, Section 3.3)
* **Mass Dominance:** Halos with mass $M_{\text{halo}} \approx 10^{13} h^{-1} M_{\odot}$ contribute to most of the convergence variance. Halos with masses below $10^9 h^{-1} M_{\odot}$ or above $10^{15} h^{-1} M_{\odot}$ have a negligible impact.
* **Redshift Scaling:** The convergence variance for sources at $z = 3$ is approximately 5 times greater than that for sources at $z = 1$.
* **Skewness Drivers:** More massive halos contribute significantly to the overall asymmetry (skewness) of the PDF. Less massive halos are much more numerous, distributing the mass over many objects, which leads to a higher number of interactions (hits) per ray and drives the PDF towards a Gaussian distribution.

### Figure 6 Decomposition (Page 8 - Contribution to Variance and Skewness vs. Halo Mass)
* **Variance ($10^3 \times \mu_2$):**
  * **At $z = 1$:** Rises from $\approx 0.005$ at $M_{\text{halo}} = 10^9 h^{-1} M_{\odot}$ to a peak of $\approx 0.10$ at $M_{\text{halo}} \approx 10^{13} h^{-1} M_{\odot}$, then falls to $\approx 0.00$ at $M_{\text{halo}} = 10^{15} h^{-1} M_{\odot}$.
  * **At $z = 3$:** Rises from $\approx 0.02$ at $M_{\text{halo}} = 10^9 h^{-1} M_{\odot}$ to a peak of $\approx 0.60$ in the range $M_{\text{halo}} \in [10^{13}, 10^{14}] h^{-1} M_{\odot}$, then declines to $\approx 0.05$ at $M_{\text{halo}} = 10^{15} h^{-1} M_{\odot}$.
* **Standardized Skewness ($\tilde{\mu}_3$):**
  * **At $z = 1$:** Starts at $\approx 1.5$ for $M_{\text{halo}} = 10^9 h^{-1} M_{\odot}$ and increases to $\approx 30$ at $10^{15} h^{-1} M_{\odot}$.
  * **At $z = 3$:** Starts at $\approx 0.6$ for $M_{\text{halo}} = 10^9 h^{-1} M_{\odot}$ and increases to $\approx 40$ at $10^{15} h^{-1} M_{\odot}$.
* **Average Halo Hits per Light Ray:**
  * **At $z = 1$:** Decreases from $\approx 20$ hits at $M_{\text{halo}} = 10^9 h^{-1} M_{\odot}$ down to $\approx 0.5$ at $10^{12} h^{-1} M_{\odot}$, and vanishes below $0.02$ at $10^{15} h^{-1} M_{\odot}$.
  * **At $z = 3$:** Decreases from $\approx 50$ hits at $M_{\text{halo}} = 10^9 h^{-1} M_{\odot}$ down to $\approx 2$ at $10^{12} h^{-1} M_{\odot}$, and drops to $\approx 0.02$ at $10^{15} h^{-1} M_{\odot}$.

### Figure 7 Decomposition (Page 9 - Contribution to Variance and Skewness vs. Lens Redshift)
* **Variance ($10^3 \times \mu_2$):**
  * **For Source $z_s = 1$:** Rises from $z = 0$ to a peak contribution of $\approx 0.10$ at lens redshift $z \approx 0.4$, before falling back to $0$ at $z = 1.0$.
  * **For Source $z_s = 3$:** Rises from $z = 0$ to a peak contribution of $\approx 0.30$ at lens redshift $z \approx 0.8$, before falling back to $0$ at $z = 3.0$.
* **Standardized Skewness ($\tilde{\mu}_3$):**
  * The skewness curves for source redshifts $z_s = 1$ and $z_s = 3$ fall exactly on top of each other, starting at $\approx 11$ at $z = 0$ and decreasing monotonically to $\approx 4.5$ at $z = 3$.
* **Average Halo Hits per Light Ray (per redshift bin $\Delta z$):**
  * **For Source $z_s = 1$:** Increases from $\approx 2$ hits at $z = 0$ to $\approx 4.5$ hits at $z = 1.0$.
  * **For Source $z_s = 3$:** Increases from $\approx 2$ hits at $z = 0$ to $\approx 8.5$ hits at $z = 2.5$.

---

## 5. Conclusions on PDF Tails and Filamentary Physics

### (a) PDF Tails Requiring N-Body Input
* **Page 10-11, Section 4 ("Conclusions", point 1):**
  > "Approximate methods are effective primarily within the weak and medium-lensing region, which mainly arises from mildly nonlinear matter structures. Accurately modeling the tails of the PDFs necessitates N-body simulations, as demonstrated by our analysis of Types I, II, and III lensing PDFs."

### (b) Filaments vs. Halo Clustering as Next-Order Physics
* **Page 3, Section 2.1.3:**
  > "This aligns with the fact that the halo model does not represent low-density filamentary structures, which become relatively more important at lower redshifts. Addressing this discrepancy necessitates the inclusion of higher mode wavenumbers in calculating the 2-halo variance, thereby compensating for the model’s lack of power in these regions."

* **Page 6, Section 3.1:**
  > "Since these plots probe weak convergences and, therefore, mildly nonlinear structures, the slight discrepancies should be attributed to a lack of modeling precision at the scale of filaments, which will be discussed subsequently."

* **Page 10, Section 3.5:**
  > "Here, the filaments represented in PINOCCHIO appear more pronounced than those in the N-body simulation, highlighting the limits of agreement. This finding is consistent with the analysis of the PDFs detailed in Sections 3.1 and 3.2, which highlighted disagreement for $\gamma \gtrsim 0.1$. In particular, we can infer that the correct modeling of filaments is important for type II and III images."

* **Page 11, Section 4 ("Conclusions", point 6):**
  > "6. The analyses of the moments in Figs. 6 and 7 and the maps (Figs. 9–11) underscore the need for precise modeling of mass distribution and non-linear clustering of halos with $M > 10^{12} M_{\odot}$. The challenges are twofold: (i) while there is a moderate consensus on the impact of baryons at the scale of galaxy clusters, their effects in individual galaxies and galaxy groups remain far less understood and more uncertainties (see, e.g., Euclid Collaboration: Castro, T., et al. 2023); (ii) lighter objects in filaments require both accurate modeling of their individual profiles and a detailed understanding of their interconnected mass distributions."
