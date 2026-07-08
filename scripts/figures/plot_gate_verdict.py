"""Section-4 gate verdict: substructure vs the systematics it competes with."""
import numpy as np, matplotlib.pyplot as plt

# fractional effect on the convergence moments (%), zs=1
effects = [
    ("subhalo substructure\n(clumps, Poisson)",        3.9,   1.8,  'tab:red'),
    ("host $M_{\\min}$ shift\n($\\pm$1 dex @ $10^{11}$)", 8.1,   1.7,  'tab:orange'),
    ("host features\n(filaments+bias+ellipt.)",        64.0,  248.0, 'tab:purple'),
    ("ACE $-$ Vaskonen\n(model vs model)",             50.0,  86.0,  'tab:blue'),
]
labels = [e[0] for e in effects]
k2 = [e[1] for e in effects]; k3 = [e[2] for e in effects]; cols = [e[3] for e in effects]
y = np.arange(len(effects)); hbar = 0.36

fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.barh(y+hbar/2, k2, hbar, color=cols, alpha=0.95, label=r'effect on $\langle\kappa^2\rangle$')
ax.barh(y-hbar/2, k3, hbar, color=cols, alpha=0.5, hatch='///',
        label=r'effect on $\langle\kappa^3\rangle$')
for yi, v in zip(y+hbar/2, k2): ax.text(v*1.08, yi, f'{v:.0f}%' if v>=10 else f'{v:.1f}%',
                                         va='center', fontsize=8)
for yi, v in zip(y-hbar/2, k3): ax.text(v*1.08, yi, f'{v:.0f}%' if v>=10 else f'{v:.1f}%',
                                         va='center', fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
ax.set_xscale('log'); ax.set_xlim(1, 500)
ax.axvspan(1, 4.0, color='green', alpha=0.07)
ax.set_xlabel('fractional effect on the convergence moments  [%]  (log scale)')
ax.set_title('§4 gate verdict ($z_s=1$): the subhalo signal is the SMALLEST systematic\n'
             'substructure (top, green zone) is swamped by host-model and model-vs-model uncertainty',
             fontsize=11)
ax.legend(loc='lower right', fontsize=9); ax.grid(axis='x', alpha=0.3, which='both')
ax.invert_yaxis()
fig.tight_layout()
fig.savefig('plots/gate_verdict.png', dpi=140, bbox_inches='tight')
print('wrote plots/gate_verdict.png')
