"""Stage 4 -- Metropolis-Hastings + Gelman-Rubin, matching Vaskonen's setup.

Deliberately plain MH (Gaussian proposals, per-chain) rather than emcee: it is
what he uses (`basics.cpp::MCMC_sampling`), it has no new dependency, and with a
DETERMINISTIC likelihood (the emulator's key upgrade over the stochastic C++) it
is perfectly adequate at 3-6 parameters.

His settings, reproduced as defaults: 4 chains, 2600 samples, 200 burn-in,
proposals tuned to 10-28% acceptance, convergence at Gelman-Rubin R < 1.05.

⚠ With the STOCHASTIC C++ likelihood, MH is a pseudo-marginal sampler and its
stationary distribution is not exactly the posterior. Here the likelihood is
deterministic, so that caveat does not apply -- but it is the reason not to port
his acceptance-rate tuning uncritically to a noisy likelihood.
"""
from __future__ import annotations

import numpy as np


def run_chain(logpost, x0, steps, n_samples, rng, n_burn=200):
    """One MH chain. Returns (samples, acceptance_rate)."""
    x = np.array(x0, dtype=float)
    lp = logpost(x)
    if not np.isfinite(lp):
        raise ValueError(f"initial point has non-finite log-posterior: {x0}")
    out = np.empty((n_samples, x.size))
    n_acc = 0
    total = n_samples + n_burn
    for i in range(total):
        prop = x + rng.normal(0.0, steps)
        lp_prop = logpost(prop)
        if np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop
            if i >= n_burn:
                n_acc += 1
        if i >= n_burn:
            out[i - n_burn] = x
    return out, n_acc / max(n_samples, 1)


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """R-hat per parameter. chains: (n_chains, n_samples, n_par)."""
    m, n, _ = chains.shape
    if m < 2:
        return np.full(chains.shape[2], np.nan)
    means = chains.mean(axis=1)                 # (m, npar)
    W = chains.var(axis=1, ddof=1).mean(axis=0)
    B = n * means.var(axis=0, ddof=1)
    var_hat = (n - 1) / n * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_hat / W)


def run_mcmc(
    logpost,
    x0,
    *,
    prior_ranges,
    n_chains: int = 4,
    n_samples: int = 2400,
    n_burn: int = 200,
    f_step: float = 0.06,
    seed: int = 0,
    scatter: float = 0.02,
    verbose: bool = True,
):
    """4 chains from scattered starts. Steps are f_step x prior width, the same
    parameterization as the C++ driver (`main_lensing.cpp` :222-225).

    f_step default 0.06 tuned 2026-07-29 on the 300-event ET arm (mock P(mu),
    3d): measured acceptance 43.8 / 26.8 / 18.1 / 11.2 % at f_step =
    0.03 / 0.05 / 0.07 / 0.09, so 0.06 sits mid-band (~22%) in Vaskonen's
    10-28% target and mixes better (max R-hat 1.105 -> ~1.02). ⚠ The optimum
    depends on the catalogue size and parameter count -- retune for the 6d mode
    and for the LISA arm rather than assuming this transfers; the runner warns
    when the measured acceptance leaves the band.

    Returns dict with samples (n_chains, n_samples, npar), rhat, acceptance.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float)
    widths = np.array([hi - lo for lo, hi in prior_ranges], float)
    steps = f_step * widths

    chains, accs = [], []
    for c in range(n_chains):
        start = x0 + rng.normal(0.0, scatter * widths)
        start = np.clip(
            start,
            [lo + 1e-6 * (hi - lo) for lo, hi in prior_ranges],
            [hi - 1e-6 * (hi - lo) for lo, hi in prior_ranges],
        )
        s, a = run_chain(logpost, start, steps, n_samples, rng, n_burn=n_burn)
        chains.append(s)
        accs.append(a)
        if verbose:
            print(f"  chain {c}: acceptance {a:.1%}")
    chains = np.array(chains)
    rhat = gelman_rubin(chains)
    if verbose:
        print(f"  R-hat: {np.array2string(rhat, precision=4)}")
        if np.any(rhat > 1.05):
            print("  ⚠ R-hat > 1.05: NOT converged, do not quote these numbers")
        mean_acc = float(np.mean(accs))
        if not (0.10 <= mean_acc <= 0.28):
            print(
                f"  ⚠ mean acceptance {mean_acc:.1%} outside Vaskonen's 10-28% "
                f"band; retune f_step (currently {f_step})"
            )
    return dict(
        samples=chains,
        flat=chains.reshape(-1, chains.shape[2]),
        rhat=rhat,
        acceptance=np.array(accs),
        f_step=f_step,
    )


def summarize(flat: np.ndarray, keys, truth=None) -> str:
    """Mean +- sd and relative precision per parameter -- the '10% on sigma_8'
    style number. If truth is given, also the pull (bias in units of sd)."""
    lines = []
    for i, k in enumerate(keys):
        x = flat[:, i]
        m, s = float(x.mean()), float(x.std(ddof=1))
        line = f"  {k:>8s} = {m:.4f} +- {s:.4f}  ({100 * s / abs(m):.1f}%)"
        if truth is not None and k in truth:
            line += f"   pull = {(m - truth[k]) / s:+.2f} sd"
        lines.append(line)
    return "\n".join(lines)
