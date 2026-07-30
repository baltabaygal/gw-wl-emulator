"""P(mu) providers + THE plane conversion.

Every magnification PDF in this package flows through here, and this module is
the ONLY place that converts image plane -> source plane. That is deliberate:
CLAUDE.md / the HDR memo record the plane trap as the one-line, silently-wrong
failure mode of this pipeline, so it gets exactly one implementation site.

Convention (paper_writer.md sec.5, verified three independent ways):

    dP_S/dmu = mu^{-1} dP_I/dmu,   then RENORMALIZE.

The renormalization is required, not automatic: after mu^{-1} weighting the
integral is <1/mu>_I, which is only approximately 1 (the emulator's flux
calibration targets the trimmed flux). The residual is theta-dependent, so
skipping it injects a spurious theta-dependent term into log L.

Providers all return an ImagePlanePDF on a shared ln-mu grid, and the consumer
asks for `.source_plane()`. Provider plane is declared explicitly by each class
so a future ML emulator (image plane) and ACE (verify!) cannot be mixed up.

Three providers:
  * MockPDF        -- analytic, dependency-free, 6d-responsive. DEFAULT.
  * ACEPDF         -- ace_lensing.predict_pdf. Mac test env only. 4 params.
  * EmulatorPDF    -- STUB for the production flow. Raises until wired.
"""
from __future__ import annotations

import numpy as np

# Shared ln-mu grid. Wide enough for z_s = 10 (the low-mu edge reaches
# ln mu ~ -0.6 in the sim at z_s=10) and for the high-mu tail to fall below
# any weight the likelihood gives it. Resolution gated in tests/run_gates.py.
#
# ⚠ LNMU_MAX was raised 2.5 -> 4.0 (2026-07-29) because the grid gate showed the
# likelihood moving 0.063 nats when the range was widened: the mu^-2 image-plane
# tail still carries normalization out there, and truncating it renormalizes the
# body. Range convergence is re-checked by tests/run_gates.py::gate_grid.
LNMU_MIN, LNMU_MAX, LNMU_N = -1.5, 4.0, 1601


def default_grid() -> np.ndarray:
    return np.linspace(LNMU_MIN, LNMU_MAX, LNMU_N)


class PDFGrid:
    """A normalized 1-D PDF in ln mu, tagged with its plane.

    p is dP/dln mu (NOT dP/dmu) -- the sampler and likelihood both work in
    ln mu, matching the C++ (`Plnmuf`). Convert with dP/dmu = (dP/dlnmu)/mu.
    """

    def __init__(self, lnmu: np.ndarray, p: np.ndarray, plane: str):
        if plane not in ("image", "source"):
            raise ValueError(f"plane must be 'image' or 'source', got {plane!r}")
        self.lnmu = np.asarray(lnmu, dtype=float)
        p = np.asarray(p, dtype=float)
        if np.any(p < 0):
            p = np.clip(p, 0.0, None)
        norm = np.trapezoid(p, self.lnmu)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("PDF has non-positive or non-finite normalization")
        self.p = p / norm
        self.plane = plane

    # -- the one conversion site ------------------------------------------------
    def source_plane(self) -> "PDFGrid":
        """dP_S/dlnmu = mu^{-1} dP_I/dlnmu, renormalized. Idempotent."""
        if self.plane == "source":
            return self
        return PDFGrid(self.lnmu, self.p * np.exp(-self.lnmu), "source")

    def image_plane(self) -> "PDFGrid":
        """Inverse conversion. Provided for figure work; the likelihood never
        needs it. Renormalizes, so it is NOT bitwise round-trip exact."""
        if self.plane == "image":
            return self
        return PDFGrid(self.lnmu, self.p * np.exp(self.lnmu), "image")

    # -- diagnostics used by the gates -----------------------------------------
    def mean_inv_mu(self) -> float:
        """<1/mu> under THIS plane's measure."""
        return float(np.trapezoid(self.p * np.exp(-self.lnmu), self.lnmu))

    def mean_mu(self) -> float:
        return float(np.trapezoid(self.p * np.exp(self.lnmu), self.lnmu))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ln mu by inverse-CDF on the grid."""
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (self.p[1:] + self.p[:-1]) * np.diff(self.lnmu))]
        )
        cdf /= cdf[-1]
        return np.interp(rng.random(n), cdf, self.lnmu)


# ---------------------------------------------------------------------------
# Provider 1: analytic mock (default while no production ML checkpoint exists)
# ---------------------------------------------------------------------------
class MockPDF:
    """Analytic stand-in for the production emulator. NOT PHYSICS.

    ⚠⚠ This is a caricature with the right *qualitative* anatomy -- a sharp
    low-mu empty-beam edge, a mode just below mu = 1, positive skew, and a
    power-law high-mu tail -- and a sigma(z, theta) scaling lifted from the
    repo's own measurements. It exists so the pipeline can be built, gated and
    debugged end-to-end before the retrain lands. **No number produced through
    this provider is a forecast.** Any sigma_8 figure of merit computed on the
    mock says something about the pipeline's information content under a toy
    scatter model, and nothing about the real one.

    Anatomy, in ln mu:
      body   -- skew-normal-ish: Gaussian in a shifted variable with the
                empty-beam edge imposed as a hard floor at lnmu_edge.
      tail   -- image-plane dP_I/dmu ~ mu^-2 (paper_writer.md sec.5; source
                plane is mu^-3), blended in above lnmu_c.

    theta response (the part that matters for a recovery test):
      sigma  ~ sigma8^1.0 * (Om/0.315)^0.5 * f(z)     [width carries sigma8]
      edge   ~ tied to sigma (empty beam moves with the mean convergence)
    Weak, deliberate h-dependence: none. h enters the forecast through the
    background D_L(z), not through P(mu) -- which is also true of the real
    simulator to first order, and is why the Om-h degeneracy is a background
    effect while sigma_8 comes from the scatter.
    """

    plane = "image"

    def __init__(self, grid: np.ndarray | None = None):
        self.lnmu = default_grid() if grid is None else np.asarray(grid, float)
        self._cache: dict = {}

    # -- width law -------------------------------------------------------------
    @staticmethod
    def sigma_lnmu(z, theta: dict) -> float:
        """sigma(ln mu) at redshift z.

        Anchored to the repo's measured z_s = 1 value (sigma ~ 0.25 at the
        fiducial with the production config; see CLAUDE.md sec.13) and the
        z^{3/2}-ish growth of docs/sigma_full_analytic_note.md, saturating at
        high z. Scaled by sigma8 (linearly: sigma_kappa ~ sigma8) and weakly by
        Om.
        """
        z = np.asarray(z, dtype=float)
        s8 = float(theta.get("sigma8", 0.811))
        Om = float(theta.get("Om", 0.315))
        shape = (z / 1.0) ** 1.5 / (1.0 + 0.55 * z) ** 1.0
        return 0.25 * shape * (s8 / 0.811) * (Om / 0.315) ** 0.5

    def __call__(self, z: float, theta: dict) -> PDFGrid:
        """Build, then flux-calibrate by a shift in ln mu so <1/mu>_I = 1.

        The shift is exactly what the production pipeline's flux calibration
        does (CLAUDE.md: FLUX_TARGET = "unit", a lnmu shift delta), and it is
        cheap here: <1/mu> -> e^{-delta} <1/mu>, so delta = ln<1/mu> zeroes it
        in one step. Without it the mock violates the flux theorem by ~11%,
        which would silently bias any absolute-distance statement made through
        it (the recovery gate would still pass, since catalogue and likelihood
        share the PDF -- which is exactly why this needs its own gate).
        """
        # Iterated: shifting the profile also moves it relative to the grid
        # edges, so one step leaves ~6e-3. Three fixed-point steps converge to
        # <1e-6, which the flux gate checks.
        # The fixed point converges quadratically once the grid truncation is
        # small: step 1 lands at ~1e-5, step 2 at ~1e-9. Cap at 3 and break at
        # 1e-7 -- the flux gate's tolerance is 1e-3, so this is 4 decades of
        # headroom, and each extra iteration is a full profile rebuild inside
        # the MCMC inner loop (it doubled the recovery gate's runtime).
        key = (round(float(z), 9), tuple(sorted((k, round(float(v), 12))
                                                for k, v in theta.items())))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        delta = 0.0
        g = None
        for _ in range(3):
            g = PDFGrid(self.lnmu, self._build(self.lnmu - delta, z, theta),
                        self.plane)
            step = np.log(g.mean_inv_mu())
            if abs(step) < 1e-7:
                break
            delta += step
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = g
        return g

    def _build(self, lnmu, z: float, theta: dict) -> np.ndarray:
        lnmu = np.asarray(lnmu, float)
        sig = float(np.atleast_1d(self.sigma_lnmu(z, theta))[0])
        sig = max(sig, 1e-3)

        # Empty-beam edge: the most demagnified ray. Scales with the width.
        lnmu_edge = -2.6 * sig
        # Mode sits just below mu = 1 (positive skew pulls the mean up).
        mode = -0.35 * sig
        # Skewed body: half-Gaussian widths differ either side of the mode.
        left, right = 0.62 * sig, 1.0 * sig
        w = np.where(lnmu < mode, left, right)
        body = np.exp(-0.5 * ((lnmu - mode) / w) ** 2)
        body[lnmu < lnmu_edge] = 0.0
        # Soften the edge slightly so the likelihood has no hard zero-derivative
        # cliff (the real edge is steep but finite).
        soft = 0.12 * sig
        ramp = np.clip((lnmu - lnmu_edge) / soft, 0.0, 1.0)
        body *= ramp

        # Image-plane power-law tail: dP_I/dmu ~ mu^-2  =>  dP_I/dlnmu ~ mu^-1.
        # ⚠ Blend point matters more than it looks. dP/dlnmu ~ e^{-lnmu} has an
        # e-folding length of 1 in ln mu REGARDLESS of sigma (that is the mu^-2
        # physics), so if the tail carries much mass it dominates the quantiles
        # and the normalization: at 2.2 sigma it held ~18% of the mass, which
        # (a) made the interquartile width sigma-INDEPENDENT and (b) left 0.2%
        # of the mass beyond the grid, breaking range convergence. 3.3 sigma
        # puts the blend near mu_c ~ 1.7 at z_s = 1, matching the production
        # model's blend point (CLAUDE.md, NSF emulator), and drops the tail
        # mass to ~0.5%.
        lnmu_c = mode + 3.3 * sig
        tail = np.exp(-(lnmu - lnmu_c))
        # clip the sigmoid argument: at the low-mu end it overflows to inf,
        # which is harmless (blend -> 0) but noisy.
        arg = np.clip((lnmu - lnmu_c) / (0.25 * sig), -500.0, 500.0)
        blend = 1.0 / (1.0 + np.exp(-arg))
        amp = np.interp(lnmu_c, lnmu, body) / max(np.interp(lnmu_c, lnmu, tail), 1e-300)
        p = (1.0 - blend) * body + blend * amp * tail
        # Re-impose the empty-beam edge as a HARD zero. The blended tail leaves
        # an exponentially small but nonzero density below the edge, which would
        # make the support unbounded below -- physically wrong (no ray is more
        # demagnified than the empty beam) and it defeats the edge gate.
        p[lnmu < lnmu_edge] = 0.0
        return p


# ---------------------------------------------------------------------------
# Provider 2: ACE
# ---------------------------------------------------------------------------
class ACEPDF:
    """ace_lensing.predict_pdf adapter.

    ⚠ THREE CAVEATS, all load-bearing:
    1. **Plane is ASSUMED image and is NOT verified.** ACE's docs do not state a
       plane convention. `plane` is a constructor arg so it can be corrected in
       one place; run tests/run_gates.py --ace, which reports <1/mu> and <mu> in
       both readings, and set it from the evidence before trusting any output.
       (A source-plane PDF has <mu>_S ~ 1 + var-ish and <1/mu>_I = 1 exactly for
       a flux-conserving image-plane one; neither is decisive alone, which is
       why this stays an open item rather than an auto-detect.)
    2. **Parameter space is (Om, h, w, sigma8) only.** Ob, ns, zeq are IGNORED,
       so a 6d run through ACE has three parameters that move nothing in P(mu)
       and are constrained by the background alone (zeq) or not at all (Ob, ns).
    3. **ACE disagrees with this repo's simulator by 36-42% in <kappa^2>**
       (CLAUDE.md sec.12). It is a different model, not a proxy for ours.

    Mac `test` env only (needs the ace_lensing install).
    """

    def __init__(self, grid: np.ndarray | None = None, plane: str = "image",
                 w: float = -1.0):
        self.lnmu = default_grid() if grid is None else np.asarray(grid, float)
        self.plane = plane
        self.w = float(w)
        self._fn = None

    def _load(self):
        if self._fn is None:
            import sys
            from ._repo import REPO_ROOT

            p = str(REPO_ROOT / "ace_lensing")
            if p not in sys.path:
                sys.path.insert(0, p)
            from ace_lensing import predict_pdf  # type: ignore

            self._fn = predict_pdf
        return self._fn

    def __call__(self, z: float, theta: dict) -> PDFGrid:
        fn = self._load()
        mu, pdf = fn(
            float(theta.get("Om", 0.315)),
            float(theta.get("h", 0.674)),
            self.w,
            float(theta.get("sigma8", 0.811)),
            float(z),
            verbose=False,
        )
        mu = np.asarray(mu, float)
        pdf = np.asarray(pdf, float)
        good = np.isfinite(mu) & np.isfinite(pdf) & (mu > 0)
        mu, pdf = mu[good], pdf[good]
        # dP/dmu on an arbitrary mu grid -> dP/dlnmu on ours.
        lnmu_src = np.log(mu)
        dpdlnmu = pdf * mu
        p = np.interp(self.lnmu, lnmu_src, dpdlnmu, left=0.0, right=0.0)
        return PDFGrid(self.lnmu, p, self.plane)


# ---------------------------------------------------------------------------
# Provider 3: the production emulator (stub)
# ---------------------------------------------------------------------------
class EmulatorPDF:
    """Adapter for ml/autoresearch/smooth_model.py::load_smooth_fn().

    Deliberately a stub: as of 2026-07-29 the 1+6d retrain has not been run
    (CLAUDE.md -- caches are on the old amplitude/physics, the 6d edge and tail
    fit caches do not exist). Wiring this now would produce a call that either
    raises FileNotFoundError deep in smooth_model or, worse, silently loads the
    OLD 1+3d body and answers in the wrong parameterization.

    When the retrain lands: load the smooth fn, evaluate on `self.lnmu` at
    context (z, h, Om, sigma8, Ob, ns, zeq/1000) in ml.params.CONTEXT_KEYS
    order, wrap as PDFGrid(..., plane="image") -- the shipped body is IMAGE
    plane (paper_writer.md sec.5) -- and add a gate comparing <1/mu> against
    the flux target.
    """

    plane = "image"

    def __init__(self, grid: np.ndarray | None = None):
        self.lnmu = default_grid() if grid is None else np.asarray(grid, float)

    def __call__(self, z: float, theta: dict) -> PDFGrid:
        raise NotImplementedError(
            "EmulatorPDF is a stub: the 1+6d retrain has not been run "
            "(see CLAUDE.md 'PIPELINE REWIRED FOR THE RETRAIN'). Use "
            "MockPDF for pipeline work, or ACEPDF on the Mac test env. "
            "Wiring instructions are in this class's docstring."
        )


# ---------------------------------------------------------------------------
# Provider 4: calibrated against the stored simulator datasets
# ---------------------------------------------------------------------------
class DatasetPDF:
    """P(mu) reconstructed from quantiles fitted to the C++ engine's own output.

    Built by `calibrate.py` from ~3100 configs of real `sample_lnmu` output
    (datasets/{tailrich,logz_1k,backend_current_1k}). Two pieces:

      width(z, Om, h, sigma8)  -- a log-linear law, 5.8% rms over the configs
      shape template           -- standardized quantiles (q - q50)/width,
                                  interpolated in ln z (the shape is NOT
                                  z-universal: the tail grows with z)

    The PDF is then the derivative of the reconstructed quantile function:
    monotone cubic (PCHIP) through the (quantile level, ln mu) knots, so
    p(lnmu) = dF/dlnmu >= 0 by construction.

    WHY NOT interpolate the datasets directly: 3308 scattered points in 4-D is
    ~7 per dimension, so a nearest-neighbour or local-regression lookup gives a
    discontinuous likelihood whose sigma8 response is dominated by the LHS
    design rather than by physics. A smooth parametric fit is both better
    behaved and honest about what it is.

    ⚠⚠ JUNE-2026 PHYSICS, NOT THE PAPER'S. subhalo_model=3, bias_model=0,
    kappa_anchor=0 (see calibrate.py's banner). Also 1+3d: **Ob, ns and zeq do
    nothing here**, exactly as for the mock -- the datasets do not vary them.
    Better than MockPDF because the (z, Om, h, sigma8) response is MEASURED;
    still not the production model.

    ⚠ The engine's own <1/mu>_I is not 1 in these datasets (median 1.006, 37% of
    configs above 1.01, worst 1.39 -- the kappa_anchor=0 monster-ray artifact).
    `flux_calibrate=True` (default) applies the production convention
    (FLUX_TARGET="unit") by shifting ln mu, so the provider satisfies the flux
    theorem even though the training data did not. Set False to reproduce the
    datasets' own violation.
    """

    plane = "image"

    def __init__(self, grid: np.ndarray | None = None, calib_path=None,
                 flux_calibrate: bool = True):
        import json

        from ._repo import PKG_ROOT

        self.lnmu = default_grid() if grid is None else np.asarray(grid, float)
        p = (PKG_ROOT / "cache" / "calibration_1p3d.json"
             if calib_path is None else calib_path)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- build it with:\n"
                "  python -m hubble_reconstruct.calibrate summarize\n"
                "  python -m hubble_reconstruct.calibrate fit")
        c = json.loads(p.read_text())
        self.calib = c
        self.beta = np.array(c["width_beta"], float)
        self.qlev = np.array(c["qlevels"], float) / 100.0
        self.zc = np.array(c["z_centers"], float)
        self.tmpl = np.array(c["shape_template"], float)
        self.flux_calibrate = bool(flux_calibrate)
        self._cache: dict = {}

    def width(self, z, theta: dict) -> float:
        z = float(z)
        x = np.array([1.0, np.log(z), np.log1p(z),
                      np.log(float(theta["sigma8"])),
                      np.log(float(theta["Om"])),
                      np.log(float(theta["h"]))])
        return float(np.exp(x @ self.beta))

    def _shape(self, z: float) -> np.ndarray:
        """Standardized quantile offsets at redshift z, linear in ln z between
        template bins and clamped outside the calibrated range."""
        lz = np.log(np.clip(z, self.zc[0], self.zc[-1]))
        lzc = np.log(self.zc)
        j = int(np.clip(np.searchsorted(lzc, lz) - 1, 0, len(lzc) - 2))
        t = (lz - lzc[j]) / (lzc[j + 1] - lzc[j])
        return (1.0 - t) * self.tmpl[j] + t * self.tmpl[j + 1]

    def __call__(self, z: float, theta: dict) -> PDFGrid:
        key = (round(float(z), 9),
               tuple(sorted((k, round(float(theta[k]), 12))
                            for k in ("Om", "h", "sigma8"))))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        w = self.width(z, theta)
        knots = self._shape(float(z)) * w          # ln mu offsets from q50
        # Reconstruct the CDF: monotone interpolation of quantile level vs lnmu,
        # differentiated to give the density. PCHIP keeps F monotone => p >= 0.
        from scipy.interpolate import PchipInterpolator

        # enforce strict monotonicity of the knots (median-of-configs templates
        # can tie at the extremes for narrow z bins)
        k = np.maximum.accumulate(knots + np.arange(knots.size) * 1e-12)
        F = PchipInterpolator(k, self.qlev, extrapolate=False)
        x = self.lnmu - 0.0
        p = F.derivative()(x)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p[x < k[0]] = 0.0
        # exponential closure beyond the last knot: dP_I/dmu ~ mu^-2
        tail = x > k[-1]
        if np.any(tail):
            amp = max(float(F.derivative()(k[-1])), 1e-300)
            p[tail] = amp * np.exp(-(x[tail] - k[-1]))
        g = PDFGrid(self.lnmu, p, self.plane)

        if self.flux_calibrate:
            delta = 0.0
            for _ in range(3):
                step = np.log(g.mean_inv_mu())
                if abs(step) < 1e-7:
                    break
                delta += step
                kk = k + delta
                F = PchipInterpolator(kk, self.qlev, extrapolate=False)
                p = np.nan_to_num(F.derivative()(x), nan=0.0, posinf=0.0,
                                  neginf=0.0)
                p[x < kk[0]] = 0.0
                tl = x > kk[-1]
                if np.any(tl):
                    amp = max(float(F.derivative()(kk[-1])), 1e-300)
                    p[tl] = amp * np.exp(-(x[tl] - kk[-1]))
                g = PDFGrid(self.lnmu, p, self.plane)

        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = g
        return g


PROVIDERS = {"mock": MockPDF, "ace": ACEPDF, "emulator": EmulatorPDF,
             "dataset": DatasetPDF}


def get_provider(name: str, **kw):
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; have {sorted(PROVIDERS)}")
    return PROVIDERS[name](**kw)
