import numpy as np
import matplotlib.pyplot as plt

# Load samples
x = np.loadtxt("lnmu.txt")

print("N =", len(x))
print("mean(lnmu) =", np.mean(x))
print("var(lnmu) =", np.var(x))
print("min, max =", np.min(x), np.max(x))

# Histogram normalized as PDF
plt.hist(x, bins=80, density=True, alpha=0.7)

plt.xlabel(r"$\ln \mu$")
plt.ylabel("PDF")
plt.title("PDF of ln(mu)")
plt.grid(True)

plt.show()
