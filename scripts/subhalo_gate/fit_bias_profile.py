import numpy as np
import matplotlib.pyplot as plt

# Digitized data points from Figure 7 (top-right panel: bias function log10(dN_sub/dN_NFW) vs log10(x))
# We extract points for the Bolshoi simulation (black squares) and the green curve (Withering + disruption)
bolshoi_logx = np.array([-1.5, -1.4, -1.3, -1.25, -1.15, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
bolshoi_logy = np.array([-1.52, -1.50, -1.33, -1.28, -1.30, -1.25, -1.10, -0.97, -0.85, -0.73, -0.58, -0.48, -0.38, -0.25, -0.15, -0.06, 0.0])

# Green line (Withering + disruption) points
green_logx = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
green_logy = np.array([-1.55, -1.43, -1.29, -1.15, -1.01, -0.88, -0.75, -0.62, -0.50, -0.39, -0.29, -0.21, -0.13, -0.07, -0.03, 0.0])

# Convert to linear x and y (bias B)
x_bolshoi = 10**bolshoi_logx
B_bolshoi = 10**bolshoi_logy

x_green = 10**green_logx
B_green = 10**green_logy

# 1. User's fit: 1/((x/0.6)**-3 + 1)**0.5
def B_user(x):
    # Normalized so that B(1) = 1
    raw = 1.0 / np.sqrt((x / 0.6)**(-3.0) + 1.0)
    norm = 1.0 / np.sqrt((1.0 / 0.6)**(-3.0) + 1.0)
    return raw / norm

# 2. Power law model: B(x) = x^alpha
def B_power(x, alpha):
    return x**alpha

# 3. Generalized transition model: B(x) = ( (x/x0)**-alpha + 1 )**-beta
def B_gen(x, x0, alpha, beta):
    raw = ((x / x0)**(-alpha) + 1.0)**(-beta)
    norm = ((1.0 / x0)**(-alpha) + 1.0)**(-beta)
    return raw / norm

# Fit models using scipy optimize curve_fit
from scipy.optimize import curve_fit

# Fit power law
popt_pow_b, _ = curve_fit(B_power, x_bolshoi, B_bolshoi, p0=[1.0])
popt_pow_g, _ = curve_fit(B_power, x_green, B_green, p0=[1.0])

# Fit generalized model (let's fix beta=1/2 or beta=1 for stability, or fit all)
# Let's fit all three: x0, alpha, beta
popt_gen_b, _ = curve_fit(lambda x, x0, alpha, beta: B_gen(x, x0, alpha, beta), 
                          x_bolshoi, B_bolshoi, p0=[0.5, 2.0, 0.5],
                          bounds=((1e-3, 0.1, 0.1), (2.0, 10.0, 5.0)))
popt_gen_g, _ = curve_fit(lambda x, x0, alpha, beta: B_gen(x, x0, alpha, beta), 
                          x_green, B_green, p0=[0.5, 2.0, 0.5],
                          bounds=((1e-3, 0.1, 0.1), (2.0, 10.0, 5.0)))

# Let's also fit a modified user model where beta=0.5, and we fit x0 and alpha
popt_user_like_b, _ = curve_fit(lambda x, x0, alpha: B_gen(x, x0, alpha, 0.5), 
                               x_bolshoi, B_bolshoi, p0=[0.6, 3.0])
popt_user_like_g, _ = curve_fit(lambda x, x0, alpha: B_gen(x, x0, alpha, 0.5), 
                               x_green, B_green, p0=[0.6, 3.0])

# Let's print out the best fit parameters
print("Power law fits:")
print(f"  Bolshoi: alpha = {popt_pow_b[0]:.3f}")
print(f"  Green:   alpha = {popt_pow_g[0]:.3f}")

print("\nUser-like fits (beta = 0.5 fixed):")
print(f"  Bolshoi: x0 = {popt_user_like_b[0]:.3f}, alpha = {popt_user_like_b[1]:.3f}")
print(f"  Green:   x0 = {popt_user_like_g[0]:.3f}, alpha = {popt_user_like_g[1]:.3f}")

print("\nGeneralized fits (x0, alpha, beta free):")
print(f"  Bolshoi: x0 = {popt_gen_b[0]:.3f}, alpha = {popt_gen_b[1]:.3f}, beta = {popt_gen_b[2]:.3f}")
print(f"  Green:   x0 = {popt_gen_g[0]:.3f}, alpha = {popt_gen_g[1]:.3f}, beta = {popt_gen_g[2]:.3f}")

# Plotting the comparison
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(bolshoi_logx, bolshoi_logy, color='k', marker='s', s=40, label='Bolshoi (Figure 7)')
ax.plot(green_logx, green_logy, 'g-', lw=2, label='Green et al. 2021 (Withering+disruption)')

# Plot models
x_eval = np.logspace(-2, 0, 200)
logx_eval = np.log10(x_eval)

ax.plot(logx_eval, np.log10(B_user(x_eval)), 'r--', lw=1.5, label=r"User's eye-fit: $[(x/0.6)^{-3} + 1]^{-1/2}$")
ax.plot(logx_eval, np.log10(B_power(x_eval, popt_pow_g[0])), 'm-.', lw=1.8, label=f"Power-law fit (Green): $x^{{{popt_pow_g[0]:.2f}}}$")
ax.plot(logx_eval, np.log10(B_power(x_eval, popt_pow_b[0])), 'k-.', lw=1.8, label=f"Power-law fit (Bolshoi): $x^{{{popt_pow_b[0]:.2f}}}$")
ax.plot(logx_eval, np.log10(B_gen(x_eval, popt_user_like_g[0], popt_user_like_g[1], 0.5)), 'b-', lw=2, label=f"Transition fit (Green): $[(x/{popt_user_like_g[0]:.2f})^{{-{popt_user_like_g[1]:.2f}}} + 1]^{{-1/2}}$")
ax.plot(logx_eval, np.log10(B_gen(x_eval, popt_user_like_b[0], popt_user_like_b[1], 0.5)), 'c-', lw=2, label=f"Transition fit (Bolshoi): $[(x/{popt_user_like_b[0]:.2f})^{{-{popt_user_like_b[1]:.2f}}} + 1]^{{-1/2}}$")




ax.set_xlim(-1.6, 0.05)
ax.set_ylim(-2.1, 0.1)
ax.set_xlabel(r'$\log_{10}(x \equiv r/r_{\rm vir})$')
ax.set_ylabel(r'$\log_{10}[{\rm Bias\ Function}\ B(x)]$')
ax.set_title('Subhalo Radial Bias Function Fitting')
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('plots/bias_fit_comparison.png', dpi=150)
print("\nWrote plots/bias_fit_comparison.png")
