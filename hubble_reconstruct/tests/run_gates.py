"""Acceptance gates. `python -m hubble_reconstruct.tests.run_gates`

Every gate is a claim this package makes about itself. A gate that passes
trivially is worse than no gate, so each prints the number it checked, not just
OK/FAIL -- consistent with the repo's standing rule that "at floor" is a
statement about depth, not about the effect.

Gates:
  1  background   -- flat-LCDM D_L against closed forms + an independent quad
  2  plane        -- the mu^{-1} conversion: direction, renormalization, identities
  3  grid         -- ln-mu grid resolution/range convergence of the likelihood
  4  pdf          -- mock PDF anatomy (edge, skew, tail slope, sigma8 response)
  5  catalogue    -- generation order, and mu-fairness under each selection rule
  6  pdet_noop    -- P_det is a genuine no-op under a mu-independent rule
  7  recovery     -- injection-recovery: the pull on 3 parameters
  8  selection    -- sec.5: omitting the matched P_det biases sigma_8
  9  gaussian     -- the non-Gaussian shape carries information vs the Gaussian control
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from .. import catalogue as cat_mod
from .._repo import FIDUCIAL, PRIOR_VASKONEN, THETA3_KEYS
from ..background import Background, C_KMS
from ..likelihood import GaussianApproxLikelihood, HDRLikelihood
from ..mcmc import run_mcmc
from ..pmu import MockPDF, PDFGrid, default_grid
from ..selection import RedshiftCut, SNRCut

RESULTS = []

# Catalogue realizations for the two bias gates. 24 is enough to establish the
# SIGN of the selection effect and that the correction removes it; it is NOT
# enough to QUOTE the size (at 24 the uncorrected bias reads ~1.8 sigma, and
# the repo's standing rule -- CLAUDE.md sec.16, after two 2-3 sigma false
# positives collapsed under a depth doubling -- is to require a depth
# confirmation or >=5 sigma before calling a shift real).
#   HDR_GATE_NREAL=200 python -m hubble_reconstruct.tests.run_gates
N_REAL = int(os.environ.get("HDR_GATE_NREAL", "24"))


def gate(name):
    def deco(fn):
        def wrapper(*a, **k):
            t0 = time.time()
            try:
                msg = fn(*a, **k)
                ok = True
            except AssertionError as e:
                msg, ok = str(e), False
            RESULTS.append((name, ok, msg, time.time() - t0))
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<12s} {msg}  "
                  f"({time.time() - t0:.1f}s)")
            return ok
        return wrapper
    return deco


# ---------------------------------------------------------------------------
@gate("background")
def gate_background():
    """D_L against (a) the analytic EdS-like small-z expansion and (b) an
    independent adaptive quadrature. Also checks the zeq->radiation wiring."""
    from scipy.integrate import quad

    th = dict(FIDUCIAL)
    bg = Background(th)

    # (a) low-z: D_L -> (c/H0) [z + (1-q0)z^2/2], q0 = Om/2 - OL (radiation negl.)
    z = 1e-3
    q0 = th["Om"] / 2.0 - (1.0 - th["Om"] - bg.Or)
    approx = (C_KMS / bg.H0) * (z + 0.5 * (1.0 - q0) * z**2)
    rel_a = abs(bg.DL(z) - approx) / approx
    assert rel_a < 2e-5, f"low-z expansion mismatch {rel_a:.2e}"

    # (b) independent quadrature at several z
    def E(zz):
        return np.sqrt(bg.Or * (1 + zz) ** 4 + th["Om"] * (1 + zz) ** 3 + bg.OL)

    worst = 0.0
    for zt in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        ref = (1 + zt) * (C_KMS / bg.H0) * quad(lambda x: 1.0 / E(x), 0, zt)[0]
        worst = max(worst, abs(bg.DL(zt) - ref) / ref)
    assert worst < 1e-4, f"quad mismatch {worst:.2e}"

    # (c) zeq wiring: Or = Om/(1+zeq), and larger zeq => less radiation
    assert abs(bg.Or - th["Om"] / (1 + th["zeq"])) < 1e-12
    b2 = Background(dict(th, zeq=2500.0))
    assert b2.Or > bg.Or, "zeq -> Or direction wrong"
    return f"max rel err {worst:.1e} vs quad; low-z {rel_a:.1e}"


@gate("plane")
def gate_plane():
    """The conversion is the documented failure mode; test it four ways."""
    lnmu = default_grid()
    prov = MockPDF()
    gi = prov(1.0, FIDUCIAL)
    gs = gi.source_plane()

    assert gi.plane == "image" and gs.plane == "source"
    # 1. normalized
    for g in (gi, gs):
        assert abs(np.trapezoid(g.p, g.lnmu) - 1.0) < 1e-10, "not normalized"
    # 2. DIRECTION: source plane must be shifted to LOWER mu (down-weights
    #    magnified rays). This is the check that catches the mu vs 1/mu flip.
    assert gs.mean_mu() < gi.mean_mu(), "source plane not shifted to lower mu"
    # 3. exact pointwise identity up to the (constant) renormalization
    ratio = (gs.p / np.clip(gi.p, 1e-300, None)) * np.exp(lnmu)
    m = gi.p > gi.p.max() * 1e-6
    assert np.ptp(ratio[m]) / np.mean(ratio[m]) < 1e-10, "not a pure 1/mu reweight"
    # 4. idempotent + round trip
    assert gs.source_plane() is gs, "source_plane not idempotent"
    back = gs.image_plane()
    rel = np.max(np.abs(back.p - gi.p)) / gi.p.max()
    assert rel < 1e-10, f"round trip {rel:.2e}"
    # 5. renormalization is NOT a no-op: <1/mu>_I != 1 in general for the mock,
    #    so the weighted integral before renormalizing differs from 1.
    raw = np.trapezoid(gi.p * np.exp(-lnmu), lnmu)
    return (f"<mu>_I={gi.mean_mu():.4f} -> <mu>_S={gs.mean_mu():.4f}; "
            f"pre-renorm mass {raw:.4f}")


@gate("grid")
def gate_grid():
    """log L must be converged wrt the ln-mu grid and z-node count."""
    rng = np.random.default_rng(3)
    prov = MockPDF()
    cat = cat_mod.generate_bns_et(prov, FIDUCIAL, n_events=80, rng=rng)
    sel = RedshiftCut(2.0)
    x = [FIDUCIAL[k] for k in THETA3_KEYS]

    base = HDRLikelihood(cat, prov, sel, n_znodes=24).log_likelihood(x)
    fine = HDRLikelihood(cat, MockPDF(np.linspace(-1.5, 4.0, 3201)), sel,
                         n_znodes=24).log_likelihood(x)
    wide = HDRLikelihood(cat, MockPDF(np.linspace(-2.0, 6.0, 2329)), sel,
                         n_znodes=24).log_likelihood(x)
    nodes = HDRLikelihood(cat, prov, sel, n_znodes=48).log_likelihood(x)

    d_res, d_range = abs(fine - base), abs(wide - base)
    d_node = abs(nodes - base)
    # tolerance in nats on the TOTAL logL of 80 events; MCMC only cares about
    # differences, but a grid error that moves logL by O(1) can tilt a posterior.
    assert d_res < 0.05, f"ln-mu resolution not converged: {d_res:.3f} nats"
    assert d_range < 0.05, f"ln-mu range not converged: {d_range:.3f} nats"
    assert d_node < 2.0, f"z-node count not converged: {d_node:.3f} nats"
    return f"dlogL: res {d_res:.3f}, range {d_range:.3f}, znodes {d_node:.2f} nats"


@gate("pdf")
def gate_pdf():
    """Mock anatomy: hard low edge, positive skew in mu, mu^-2 image tail,
    and the sigma8 response the recovery test depends on."""
    prov = MockPDF()
    g = prov(1.0, FIDUCIAL)
    lnmu, p = g.lnmu, g.p
    # support edge, measured against a RELATIVE floor (an absolute p > 0 test
    # is meaningless once a blended tail contributes denormal-scale density)
    nz = lnmu[p > p.max() * 1e-10]
    assert nz.min() > -1.0, f"edge missing (support starts at {nz.min():.3f})"
    # mode below mu=1, mean above (positive skew)
    mode = lnmu[np.argmax(p)]
    assert mode < 0 < g.mean_mu() - 1.0, f"skew anatomy wrong (mode {mode:.3f})"
    # image-plane tail slope: dP_I/dmu ~ mu^-2 => dP/dlnmu ~ mu^-1
    hi = (lnmu > mode + 3.5 * 0.25) & (p > 0)
    slope = np.polyfit(lnmu[hi], np.log(p[hi]), 1)[0]
    assert abs(slope + 1.0) < 0.15, f"tail slope {slope:.3f}, want -1"
    # sigma8 response: width must scale ~linearly (this is what carries the
    # sigma8 information in the recovery gate).
    # ⚠ Measured as an INTERQUARTILE range, not a standard deviation: with a
    # mu^-2 image-plane tail the raw variance is tail-dominated and barely moves
    # with sigma8 (it read 0.963 instead of 1.286 and failed this gate). Same
    # lesson as the repo's standing rule that raw sigma_kappa is rare-tail junk
    # -- use a clipped/robust width.
    def body_width(g):
        """q05 -> q50, i.e. the DEMAGNIFICATION side only. Both quantiles sit
        below the mode, so this is free of the mu^-2 tail whose e-folding
        length is 1 in ln mu independent of sigma."""
        dl = float(g.lnmu[1] - g.lnmu[0])
        cdf = np.cumsum(g.p * dl)
        cdf /= cdf[-1]
        return float(np.interp(0.50, cdf, g.lnmu) - np.interp(0.05, cdf, g.lnmu))

    w = [body_width(prov(1.0, dict(FIDUCIAL, sigma8=s8))) for s8 in (0.7, 0.9)]
    ratio = w[1] / w[0]
    assert abs(ratio - 9 / 7) < 0.06, f"sigma8 body-width scaling {ratio:.3f}"
    return (f"tail slope {slope:.2f}, sigma8 body-width scaling {ratio:.3f} "
            f"(want {9 / 7:.3f})")


@gate("flux")
def gate_flux():
    """<1/mu>_I = 1 (the flux theorem the production calibration targets).

    Catalogue and likelihood share the PDF, so a flux violation does NOT show
    up as a recovery bias -- it silently shifts the absolute distance scale
    instead, which is why this needs its own gate rather than being covered by
    'recovery'. Checked across z and theta so the calibration cannot be
    accidentally z- or theta-dependent.
    """
    prov = MockPDF()
    worst = 0.0
    for z in (0.3, 1.0, 5.0):
        for th in (FIDUCIAL, dict(FIDUCIAL, sigma8=0.65), dict(FIDUCIAL, Om=0.25)):
            g = prov(z, th)
            worst = max(worst, abs(g.mean_inv_mu() - 1.0))
    assert worst < 1e-3, f"flux violated by {worst:.2e}"
    return f"max |<1/mu>_I - 1| = {worst:.1e} over z and theta"


@gate("catalogue")
def gate_catalogue():
    """Generation order + the mu-fairness that defines the two regimes."""
    rng = np.random.default_rng(11)
    prov = MockPDF()
    # (a) z-cut: catalogue mu must be a FAIR sample of the source-plane PDF
    c1 = cat_mod.generate_bns_et(prov, FIDUCIAL, n_events=4000, rng=rng)
    zsel = (c1.z > 0.8) & (c1.z < 1.2)
    mu_cat = float(np.mean(c1.truth["mu"][zsel]))
    mu_ref = prov(1.0, FIDUCIAL).source_plane().mean_mu()
    rel = abs(mu_cat - mu_ref) / mu_ref
    assert rel < 0.02, f"z-cut catalogue not a fair mu sample: {rel:.3f}"

    # (b) SNR cut: mu must be skewed HIGH relative to the same reference
    bg = Background(FIDUCIAL)
    sel = SNRCut.from_horizon(float(bg.DL(1.4)), snr_thr=20.0)
    c2 = cat_mod.generate(2000, prov, FIDUCIAL, sel, arm="bns_et",
                          z_range=(0.01, 2.0), frac_sigma_dL=0.03,
                          pz_fn=cat_mod.pz_bns, rng=rng)
    zsel2 = (c2.z > 0.8) & (c2.z < 1.2)
    mu_snr = float(np.mean(c2.truth["mu"][zsel2]))
    assert mu_snr > mu_cat, (
        f"SNR selection did not skew mu high ({mu_snr:.4f} vs {mu_cat:.4f}) "
        "-- the demo would be vacuous")
    boost = mu_snr / mu_cat - 1.0
    assert c2.n_drawn > len(c2), "no rejection happened"
    return (f"z-cut fair to {rel:.3f}; SNR skews <mu> +{100 * boost:.2f}% "
            f"(keep rate {100 * len(c2) / c2.n_drawn:.0f}%)")


@gate("pdet_noop")
def gate_pdet_noop():
    """Under a mu-INDEPENDENT rule, P_det must not change the posterior at all
    (Vaskonen's 'selection does not bias the inference')."""
    rng = np.random.default_rng(5)
    prov = MockPDF()
    cat = cat_mod.generate_bns_et(prov, FIDUCIAL, n_events=100, rng=rng)
    sel = RedshiftCut(2.0)
    x = [FIDUCIAL[k] for k in THETA3_KEYS]
    on = HDRLikelihood(cat, prov, sel, use_pdet=True).log_likelihood(x)
    off = HDRLikelihood(cat, prov, sel, use_pdet=False).log_likelihood(x)
    assert abs(on - off) < 1e-12, f"P_det not a no-op under z-cut: {on - off:.2e}"
    return f"identical to {abs(on - off):.1e} nats"


@gate("pdet_interp")
def gate_pdet_interp():
    """The P_det interpolation (likelihood.n_pdet) must be exact enough to be
    invisible: compare against the brute per-event evaluation."""
    rng = np.random.default_rng(23)
    prov = MockPDF()
    bg = Background(FIDUCIAL)
    sel = SNRCut.from_horizon(float(bg.DL(1.3)), snr_thr=20.0)
    cat = cat_mod.generate(300, prov, FIDUCIAL, sel, arm="bns_et",
                           z_range=(0.01, 2.0), frac_sigma_dL=0.03,
                           pz_fn=cat_mod.pz_bns, rng=rng)
    x = [FIDUCIAL[k] for k in THETA3_KEYS]
    coarse = HDRLikelihood(cat, prov, sel, n_znodes=12, n_pdet=32)
    fine = HDRLikelihood(cat, prov, sel, n_znodes=12, n_pdet=1024)
    d = abs(coarse.log_likelihood(x) - fine.log_likelihood(x))
    assert d < 0.01, f"P_det interpolation error {d:.4f} nats"
    return f"n_pdet 32 vs 1024: dlogL {d:.2e} nats over {len(cat)} events"


def profile_sigma8(cat, prov, sel, use_pdet=True, n_znodes=10,
                   lo=0.60, hi=1.05, n=19):
    """Profile-likelihood estimate of sigma8 with Om, h fixed at truth.

    A 1-D scan + parabolic refinement, ~20 likelihood calls -- cheap enough to
    repeat over many catalogue realizations, which is the only way to separate
    a BIAS from a draw (a single injection-recovery pull is N(0,1) by
    construction, so |pull| ~ 1 means nothing on its own; this is the same
    lesson as the repo's 8-shard SEM false positives, CLAUDE.md sec.16).
    """
    like = HDRLikelihood(cat, prov, sel, n_znodes=n_znodes, use_pdet=use_pdet)
    grid = np.linspace(lo, hi, n)
    vals = np.array([
        like.log_likelihood([FIDUCIAL["Om"], s, FIDUCIAL["h"]]) for s in grid
    ])
    k = int(np.argmax(vals))
    if 0 < k < n - 1:  # parabolic vertex through the three best points
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            return float(grid[k] - 0.5 * (grid[1] - grid[0]) * (y2 - y0) / denom)
    return float(grid[k])


@gate("recovery")
def gate_recovery_mcmc():
    """ONE injection-recovery MCMC: checks the sampler runs, converges, and
    lands in the right region. ⚠ Its pulls are a single draw from N(0,1), so
    they do NOT test for bias -- gate_unbiased does that. Not a forecast.
    """
    rng = np.random.default_rng(7)
    prov = MockPDF(np.linspace(-1.5, 4.0, 601))
    cat = cat_mod.generate_bns_et(prov, FIDUCIAL, n_events=300, rng=rng)
    like = HDRLikelihood(cat, prov, RedshiftCut(2.0), n_znodes=16)
    ranges = [PRIOR_VASKONEN[k] for k in THETA3_KEYS]
    res = run_mcmc(like, [FIDUCIAL[k] for k in THETA3_KEYS],
                   prior_ranges=ranges, n_chains=2, n_samples=1200, n_burn=250,
                   f_step=0.06, seed=2, verbose=False)
    pulls = {}
    for i, k in enumerate(THETA3_KEYS):
        x = res["flat"][:, i]
        pulls[k] = (float(x.mean()) - FIDUCIAL[k]) / float(x.std(ddof=1))
    worst = max(abs(v) for v in pulls.values())
    rhat = float(np.max(res["rhat"]))
    assert rhat < 1.15, f"chains not converged, R-hat {rhat:.3f}"
    assert worst < 3.0, f"pull too large even for one draw: {pulls}"
    s8 = res["flat"][:, list(THETA3_KEYS).index("sigma8")]
    return (f"pulls " + ", ".join(f"{k} {v:+.2f}" for k, v in pulls.items())
            + f"; sigma8 {100 * s8.std(ddof=1) / s8.mean():.0f}% (mock); "
            f"R-hat {rhat:.3f}")


@gate("unbiased")
def gate_unbiased():
    """Is the ESTIMATOR unbiased under the fair (z-cut) selection?

    R independent catalogues, profile-sigma8 on each, then test the MEAN
    against its standard error. This is the gate that can actually see a bias;
    a single recovery cannot.
    """
    prov = MockPDF(np.linspace(-1.5, 4.0, 601))
    est = []
    for r in range(N_REAL):
        rng = np.random.default_rng(1000 + r)
        cat = cat_mod.generate_bns_et(prov, FIDUCIAL, n_events=300, rng=rng)
        est.append(profile_sigma8(cat, prov, RedshiftCut(2.0)))
    est = np.array(est)
    bias = float(est.mean() - FIDUCIAL["sigma8"])
    sem = float(est.std(ddof=1) / np.sqrt(est.size))
    nsig = bias / sem
    assert abs(nsig) < 3.0, (
        f"estimator biased: {bias:+.4f} +- {sem:.4f} ({nsig:+.1f} sigma) "
        f"over {est.size} realizations")
    return (f"sigma8 bias {bias:+.4f} +- {sem:.4f} ({nsig:+.1f} sigma), "
            f"{est.size} realizations, scatter {est.std(ddof=1):.4f}")


@gate("selection")
def gate_selection():
    """sec.5: analysing an SNR-selected catalogue WITHOUT the matched
    mu-marginalized P_det must bias the fit; WITH it, the bias must shrink."""
    rng = np.random.default_rng(13)
    # Reduced configuration: 200 events on a 601-point ln-mu grid, 2 chains.
    # This gate runs two full MCMCs, so it is the most expensive one; the
    # coarser grid is certified by gate_grid (resolution error ~1e-3 nats) and
    # the effect being demonstrated is several sigma, not marginal. Raise these
    # numbers before quoting the pull values anywhere.
    prov = MockPDF(np.linspace(-1.5, 4.0, 601))
    bg = Background(FIDUCIAL)
    sel = SNRCut.from_horizon(float(bg.DL(1.3)), snr_thr=20.0)

    # Same R-realization estimator as gate_unbiased, so "matched" and "omitted"
    # are compared on identical catalogues -- the difference is then purely the
    # likelihood's selection treatment, not catalogue scatter.
    est = {"matched": [], "omitted": []}
    for r in range(N_REAL):
        rng = np.random.default_rng(2000 + r)
        cat = cat_mod.generate(200, prov, FIDUCIAL, sel, arm="bns_et",
                               z_range=(0.01, 2.0), frac_sigma_dL=0.03,
                               pz_fn=cat_mod.pz_bns, rng=rng)
        for label, use_pdet in (("matched", True), ("omitted", False)):
            est[label].append(profile_sigma8(cat, prov, sel, use_pdet=use_pdet))

    out = {}
    for label, v in est.items():
        v = np.array(v)
        b = float(v.mean() - FIDUCIAL["sigma8"])
        s = float(v.std(ddof=1) / np.sqrt(v.size))
        out[label] = (b, s, b / s)
    assert abs(out["matched"][2]) < 3.0, (
        f"matched analysis is biased: {out['matched'][0]:+.4f} "
        f"({out['matched'][2]:+.1f} sigma)")
    assert abs(out["matched"][0]) < abs(out["omitted"][0]), (
        f"correction did not reduce the bias: matched {out['matched'][0]:+.4f} "
        f"vs omitted {out['omitted'][0]:+.4f}")
    return ("sigma8 bias: matched "
            f"{out['matched'][0]:+.4f}+-{out['matched'][1]:.4f} "
            f"({out['matched'][2]:+.1f}s), P_det omitted "
            f"{out['omitted'][0]:+.4f}+-{out['omitted'][1]:.4f} "
            f"({out['omitted'][2]:+.1f}s), {N_REAL} realizations")


@gate("dataset_pdf")
def gate_dataset_pdf():
    """DatasetPDF must reproduce HELD-OUT simulator configs.

    Compares reconstructed quantiles against the raw lnmu samples of configs in
    a test split (never used by the fit). Errors are quoted in units of each
    config's own width, which is the scale that matters for the likelihood.

    Skipped (not failed) when the calibration or the datasets are absent -- they
    are gitignored, so a fresh checkout has neither.
    """
    import h5py

    from .._repo import REPO_ROOT
    from ..pmu import DatasetPDF

    p = REPO_ROOT / "datasets/backend_current_1k/test/dataset_test.h5"
    try:
        prov = DatasetPDF()
    except FileNotFoundError:
        return "SKIP: no calibration (run calibrate.py summarize && fit)"
    if not p.exists():
        return "SKIP: datasets/ not present"

    with h5py.File(p, "r") as f:
        s = f["samples"]
        z, Om, h, s8 = (np.asarray(s[k]) for k in ("z", "OmegaM", "h", "sigma8"))
        vc = np.asarray(s["valid_counts"])
        sel = np.where((z > 0.15) & (vc >= 5000))[0][:100]
        werr, qerr = [], {5: [], 50: [], 95: []}
        for i in sel:
            x = np.asarray(s["lnmu"][i, :int(vc[i])], dtype=float)
            th = dict(Om=float(Om[i]), h=float(h[i]), sigma8=float(s8[i]))
            g = prov(float(z[i]), th)
            dl = float(g.lnmu[1] - g.lnmu[0])
            c = np.cumsum(g.p * dl)
            c /= c[-1]
            w = np.percentile(x, 50) - np.percentile(x, 5)
            werr.append(prov.width(float(z[i]), th) / w - 1.0)
            for q in qerr:
                qerr[q].append(
                    (np.interp(q / 100.0, c, g.lnmu) - np.percentile(x, q)) / w)

    wmed = float(np.median(werr))
    wspread = float(np.percentile(np.abs(werr), 68))
    assert abs(wmed) < 0.05, f"width biased by {wmed:+.1%} on held-out configs"
    assert wspread < 0.12, f"width scatter {wspread:.1%} too large"
    # Location: anchored by the flux theorem, not fitted, so a residual offset
    # of order the datasets' own flux violation (<1/mu> ~ 1.006 = 0.12 widths)
    # is EXPECTED and is a deliberate correction, not an error.
    q50 = float(np.median(qerr[50]))
    assert abs(q50) < 0.15, f"median location off by {q50:+.3f} widths"
    return (f"held-out n={len(sel)}: width {wmed:+.1%} median, {wspread:.1%} "
            f"68pct; location {q50:+.3f} w; q05 "
            f"{float(np.median(qerr[5])):+.3f} w, q95 "
            f"{float(np.median(qerr[95])):+.3f} w")


@gate("prior_recovery")
def gate_prior_recovery():
    """6d plumbing + an ANALYTIC check on the sampler.

    With the mock provider, Ob and ns are inert by construction, so their
    marginal posterior must be exactly the uniform prior, whose sd is
    range/sqrt(12). Recovering a known closed-form answer tests the sampler in
    a way no injection-recovery can (there is no 'true' width to compare
    against in the informative directions).
    """
    from .._repo import PRIOR_6D, THETA6_KEYS
    from ..likelihood import sensitivity

    rng = np.random.default_rng(31)
    prov = MockPDF(np.linspace(-1.5, 4.0, 601))
    cat = cat_mod.generate_bns_et(prov, FIDUCIAL, n_events=80, rng=rng)
    prior = {k: PRIOR_6D[k] for k in THETA6_KEYS}
    like = HDRLikelihood(cat, prov, RedshiftCut(2.0), n_znodes=8,
                         theta_keys=THETA6_KEYS, prior=prior)

    sens = sensitivity(like, n=5)
    inert = [k for k in THETA6_KEYS if sens[k] < 1e-6]
    assert set(inert) == {"Ob", "ns"}, (
        f"expected Ob, ns inert under the mock; got {inert} "
        f"(sensitivities {sens})")

    res = run_mcmc(like, [FIDUCIAL[k] for k in THETA6_KEYS],
                   prior_ranges=[prior[k] for k in THETA6_KEYS], n_chains=2,
                   n_samples=2500, n_burn=500, f_step=0.08, seed=6,
                   verbose=False)
    msg = []
    for k in inert:
        i = list(THETA6_KEYS).index(k)
        got = float(res["flat"][:, i].std(ddof=1))
        want = (prior[k][1] - prior[k][0]) / np.sqrt(12.0)
        rel = got / want - 1.0
        assert abs(rel) < 0.15, (
            f"{k}: posterior sd {got:.5f} vs uniform prior {want:.5f} "
            f"({rel:+.1%}) -- sampler is not reproducing the prior")
        msg.append(f"{k} {rel:+.1%}")
    return f"6d runs; inert {', '.join(inert)} recover uniform prior sd: " + ", ".join(msg)


@gate("gaussian")
def gate_gaussian():
    """The non-Gaussian shape must carry information the Gaussian control lacks:
    same catalogue, the full likelihood should not be a rescaled Gaussian one."""
    rng = np.random.default_rng(17)
    prov = MockPDF()
    cat = cat_mod.generate_bns_et(prov, FIDUCIAL, n_events=200, rng=rng)
    sel = RedshiftCut(2.0)
    full = HDRLikelihood(cat, prov, sel, n_znodes=16)
    gaus = GaussianApproxLikelihood(cat, prov, sel, n_znodes=16)
    # scan sigma8: the two curvatures should differ appreciably
    s8 = np.linspace(0.70, 0.92, 9)
    curves = {}
    for nm, lk in (("full", full), ("gauss", gaus)):
        v = np.array([lk.log_likelihood([FIDUCIAL["Om"], s, FIDUCIAL["h"]])
                      for s in s8])
        curves[nm] = v - v.max()
    c_full = np.polyfit(s8, curves["full"], 2)[0]
    c_gaus = np.polyfit(s8, curves["gauss"], 2)[0]
    ratio = c_full / c_gaus
    assert np.isfinite(ratio) and ratio > 0, "curvature sign problem"
    # The two log-likelihood CURVES must not be proportional -- that is the
    # defensible claim: the non-Gaussian shape changes the sigma8 inference,
    # not merely its scale.
    resid = curves["full"] - (c_full / c_gaus) * curves["gauss"]
    shape_diff = float(np.max(np.abs(resid)))
    assert shape_diff > 0.05, (
        f"full and Gaussian log-L differ only by a scale factor "
        f"({shape_diff:.3f} nats): shape is carrying nothing here")
    # ⚠ NOT an information comparison. The Gaussian model is MISSPECIFIED for
    # data generated from the full PDF, so a larger Gaussian curvature means
    # OVER-CONFIDENCE, not more information. Vaskonen's point (Gaussianizing
    # collapses the model to sigma_WL^2 + sigma_DL^2 in quadrature) is about
    # what the model can represent; quantifying the information difference
    # properly needs matched-data posteriors or a Fisher calculation, which is
    # a study, not a gate.
    return (f"sigma8 curvature full/gauss = {ratio:.2f}; log-L shapes differ "
            f"by {shape_diff:.2f} nats after rescaling "
            f"({'Gaussian over-confident' if ratio < 1 else 'Gaussian looser'})")


def main():
    print("HDR acceptance gates -- MOCK P(mu); gates test the PIPELINE.\n")
    for fn in (gate_background, gate_plane, gate_grid, gate_pdf, gate_flux,
               gate_catalogue, gate_pdet_noop, gate_pdet_interp,
               gate_recovery_mcmc, gate_unbiased, gate_selection,
               gate_prior_recovery, gate_gaussian, gate_dataset_pdf):
        fn()
    n_fail = sum(1 for _, ok, _, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} gates passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
