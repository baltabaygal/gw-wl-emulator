import numpy as np
import matplotlib.pyplot as plt
import os

# Load samples
x = np.loadtxt("../lnmu_5k.txt")

print("N =", len(x))
print("mean(lnmu) =", np.mean(x))
print("var(lnmu) =", np.var(x))
print("min, max =", np.min(x), np.max(x))

print("mean(mu) =", np.mean(np.exp(x)))
print("skewness approx =", np.mean((x - np.mean(x))**3) / np.std(x)**3)

print("fraction with mu > 2 =", np.mean(np.exp(x) > 2))

# Histogram normalized as PDF
plt.hist(x, bins=80, density=True, alpha=0.7)

plt.xlabel(r"$\ln \mu$")
plt.ylabel("PDF")
plt.title("PDF of ln(mu)")
plt.grid(True)
plt.savefig("lnmu_histogram.pdf")
plt.show()
