"""Stage 3 -- the likelihood. Python reimplementation of Vaskonen's Eqs. (10)-(12).

Reimplemented rather than bound (HDR memo sec.6): the C++ `loglikelihood`
computes `Plnmuf` internally, so binding it buys the slow stochastic MC, not an
emulator-driven run.

The model, per event j:

    p_mod(dL | z, theta) = INT dlnmu p_S(lnmu | z, theta) delta(dL - Dtilde/sqrt(mu))
    L_j = INT ddL p_obs(dL_j | dL) p_mod(dL | z_j, theta) / P_det(z_j, theta)

Discretized on the ln-mu grid, the delta collapses the dL integral and the
numerator is a weighted sum of Gaussians -- the same structure as the C++
(:1469-1480), one Gaussian per magnification bin, evaluated at the LENSED model
distance Dtilde(z, theta)/sqrt(mu):

    L_j = SUM_i dlnmu p_S[i] N(dL_j ; Dtilde/sqrt(mu_i), sigma_j)  /  P_det

TWO CONVENTIONS THAT MUST NOT DRIFT:

1. **Plane.** p_S is SOURCE plane (pmu.PDFGrid.source_plane()). The emulator is
   image plane. The conversion happens in pmu.py and nowhere else.
2. **Selection.** P_det is computed from the SAME SelectionRule object that
   generated the catalogue, marginalized over mu at the TRIAL theta:

       P_det(z, theta) = SUM_i dlnmu p_S[i] p_det(Dtilde(z,theta)/sqrt(mu_i))

   For a mu-independent rule (RedshiftCut) this is a constant and cancels; for a
   mu-coupled rule it does not, and omitting it is the lensing-Malmquist bias.

⚠ Conditioning (HDR memo sec.5): this is the likelihood CONDITIONAL on the
observed z_j. Under an SNR-type rule the detected-z distribution is also
theta-dependent; conditioning it away is legitimate but means P_det here is the
per-event, per-z normalization -- NOT the population-integrated P(det|theta).
Do not "upgrade" one without the other (Mandel+19).

⚠ Measurement error is Gaussian in dL with a sigma_j FIXED PER EVENT from the
catalogue (as Vaskonen does: sigma_j = f * dL_true,j, a number attached to the
event). It is therefore theta-independent, which keeps the Gaussian
normalization out of the theta-dependence. If a future GWFish-style sigma
depends on theta, that normalization must be restored.
"""
from __future__ import annotations

import numpy as np

from .background import Background
from .catalogue import Catalogue
from .selection import SelectionRule

SQRT2PI = float(np.sqrt(2.0 * np.pi))


class HDRLikelihood:
    """Callable log-likelihood over theta for one catalogue.

    Parameters
    ----------
    catalogue : Catalogue
    provider  : P(mu) provider (pmu.MockPDF etc.)
    selection : the SAME rule used to generate the catalogue. Passing a
                different one is the mismatch demo -- allowed, but you have to
                do it on purpose.
    n_znodes  : number of z nodes at which P(mu) is evaluated per theta.
                Vaskonen uses 6 (10 combined); the emulator makes more nodes
                nearly free, so the default is higher. Set 6 to mimic him.
    use_pdet  : master switch, for the demo that shows what omitting it does.
    """

    def __init__(
        self,
        catalogue: Catalogue,
        provider,
        selection: SelectionRule,
        *,
        n_znodes: int = 24,
        use_pdet: bool = True,
        n_pdet: int = 32,
        theta_keys=("Om", "sigma8", "h"),
        fiducial: dict | None = None,
        prior: dict | None = None,
    ):
        from ._repo import FIDUCIAL, PRIOR_VASKONEN

        self.cat = catalogue
        self.provider = provider
        self.selection = selection
        self.use_pdet = bool(use_pdet)
        self.n_pdet = int(n_pdet)
        self.theta_keys = tuple(theta_keys)
        self.fiducial = dict(FIDUCIAL if fiducial is None else fiducial)
        self.prior = dict(PRIOR_VASKONEN if prior is None else prior)
        self.n_calls = 0

        z = np.asarray(catalogue.z, float)
        self.z_nodes = np.linspace(z.min(), z.max(), int(n_znodes))
        # nearest-node assignment (the C++ does the same, with 6 nodes)
        self._node_of = np.abs(z[:, None] - self.z_nodes[None, :]).argmin(axis=1)

    # -- theta handling --------------------------------------------------------
    def theta_dict(self, vec) -> dict:
        th = dict(self.fiducial)
        th.update({k: float(v) for k, v in zip(self.theta_keys, vec)})
        return th

    def log_prior(self, vec) -> float:
        for k, v in zip(self.theta_keys, vec):
            lo, hi = self.prior.get(k, (-np.inf, np.inf))
            if not (lo <= v <= hi):
                return -np.inf
        return 0.0

    # -- the likelihood --------------------------------------------------------
    def log_likelihood(self, vec) -> float:
        self.n_calls += 1
        theta = self.theta_dict(vec)
        bg = Background(theta)

        z = np.asarray(self.cat.z, float)
        dL_obs = np.asarray(self.cat.dL_obs, float)
        sig = np.asarray(self.cat.sigma_dL, float)
        Dt = bg.DL(z)                                    # unlensed model distance

        # P(mu) at each z node, SOURCE plane (single conversion site: pmu.py)
        grids = [self.provider(zn, theta).source_plane() for zn in self.z_nodes]

        logL = 0.0
        for jn, g in enumerate(grids):
            m = self._node_of == jn
            if not np.any(m):
                continue
            lnmu, p = g.lnmu, g.p
            dlnmu = float(lnmu[1] - lnmu[0])
            inv_sqrt_mu = np.exp(-0.5 * lnmu)            # 1/sqrt(mu)

            # model lensed distance: (events, mu-bins)
            DL_model = Dt[m][:, None] * inv_sqrt_mu[None, :]
            s = sig[m][:, None]
            resid = (dL_obs[m][:, None] - DL_model) / s
            gauss = np.exp(-0.5 * resid**2) / (s * SQRT2PI)
            if self.use_pdet and self.selection.mu_coupled:
                # SELECTION ON THE TRUE LENSED DISTANCE (what catalogue.py
                # does, and what an SNR trigger physically is: the signal
                # amplitude is set by the true dL, and the measurement error is
                # applied afterwards). The detected-event density is then
                #
                #   p(dL_obs | z, detected) =
                #       INT dlnmu p_S(lnmu) p_det(D~/sqrt(mu)) N(dL_obs; D~/sqrt(mu), s)
                #     / INT dlnmu p_S(lnmu) p_det(D~/sqrt(mu))
                #
                # i.e. p_det appears in BOTH the numerator and the
                # normalization -- the numerator integrand is restricted to
                # magnifications that could have been detected. Omitting it
                # from the numerator (an easy and quiet mistake -- the author
                # made it, and gate_selection caught it by reporting a LARGER
                # bias for the "corrected" fit than the uncorrected one) fits a
                # model in which selection reweights the normalization but not
                # the scatter.
                #
                # ⚠ This differs from Vaskonen's C++, which is self-consistent
                # under the OTHER convention: his `DLthr` cuts on the OBSERVED
                # distance (his Pdet integrates N(Y; DL0, sigma) over Y < DLthr),
                # and under observed-distance selection the numerator correctly
                # carries no extra factor. Two valid schemes; mixing them is
                # not. If a future selection cuts on the observed dL, move the
                # factor accordingly and say so here.
                pdet_bins = self.selection.p_det_of_dL(DL_model)
                num = (gauss * pdet_bins * p[None, :]).sum(axis=1) * dlnmu
                # Denominator depends on the event only through Dtilde(z_j), so
                # it is a smooth 1-D function F(Dtilde): evaluating it on a
                # small grid and interpolating replaces an (n_events x n_bins)
                # reduction with (n_pdet x n_bins). Accuracy: gate_pdet_interp.
                Dt_m = Dt[m]
                lo, hi = float(Dt_m.min()), float(Dt_m.max())
                if hi - lo < 1e-9 * max(hi, 1.0):
                    F = (self.selection.p_det_of_dL(lo * inv_sqrt_mu)
                         * p).sum() * dlnmu
                    Pdet = np.full_like(Dt_m, F)
                else:
                    grid_Dt = np.linspace(lo, hi, self.n_pdet)
                    F = (self.selection.p_det_of_dL(
                            grid_Dt[:, None] * inv_sqrt_mu[None, :])
                         * p[None, :]).sum(axis=1) * dlnmu
                    Pdet = np.interp(Dt_m, grid_Dt, F)
                L = num / np.clip(Pdet, 1e-300, None)
            else:
                L = (gauss * p[None, :]).sum(axis=1) * dlnmu

            if np.any(~np.isfinite(L)) or np.any(L <= 0):
                return -np.inf
            logL += float(np.log(L).sum())

        return logL if np.isfinite(logL) else -np.inf

    def __call__(self, vec) -> float:
        lp = self.log_prior(vec)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(vec)


def sensitivity(like, keys=None, n=9) -> "dict[str, float]":
    """How much does log L move across each parameter's prior range?

    Returns {key: max|dlogL| relative to the fiducial}. A value of ~0 means the
    parameter is INERT: the posterior is then exactly the prior, MH random-walks
    it, and Gelman-Rubin will look terrible for a reason that is not a sampling
    problem. Run this before blaming the sampler.

    ⚠ With the mock P(mu), Ob and ns are inert BY CONSTRUCTION (MockPDF ignores
    them and the background does not use them) and zeq is nearly so (it enters
    only through Omega_R = Om/(1+zeq), a ~1e-4 effect on D_L). With the real
    emulator they act on P(k) and should show small but nonzero sensitivity --
    so this function doubles as a check that a new provider is actually wired to
    all six parameters.
    """
    keys = tuple(like.theta_keys) if keys is None else tuple(keys)
    x0 = [like.fiducial[k] for k in keys]
    base = like.log_likelihood(x0)
    out = {}
    for i, k in enumerate(keys):
        lo, hi = like.prior[k]
        vals = []
        for v in np.linspace(lo, hi, n):
            x = list(x0)
            x[i] = float(v)
            ll = like.log_likelihood(x)
            vals.append(ll - base if np.isfinite(ll) else np.nan)
        out[k] = float(np.nanmax(np.abs(vals)))
    return out


class GaussianApproxLikelihood(HDRLikelihood):
    """Control: replace the non-Gaussian p_mu by a Gaussian of the same width.

    Vaskonen notes explicitly that Gaussianizing collapses the model to adding
    sigma_WL^2 + sigma_DL^2 in quadrature -- i.e. the shape information is
    discarded and only the width survives. Running both is how you demonstrate
    that the non-Gaussian shape carries information, which is a claim the paper
    makes and this package should be able to back.
    """

    def log_likelihood(self, vec) -> float:
        self.n_calls += 1
        theta = self.theta_dict(vec)
        bg = Background(theta)
        z = np.asarray(self.cat.z, float)
        dL_obs = np.asarray(self.cat.dL_obs, float)
        sig = np.asarray(self.cat.sigma_dL, float)
        Dt = bg.DL(z)

        logL = 0.0
        for jn, zn in enumerate(self.z_nodes):
            m = self._node_of == jn
            if not np.any(m):
                continue
            g = self.provider(zn, theta).source_plane()
            # width of dL/Dtilde = 1/sqrt(mu) about its mean
            r = np.exp(-0.5 * g.lnmu)
            dl = float(g.lnmu[1] - g.lnmu[0])
            mean_r = float((r * g.p).sum() * dl)
            var_r = float(((r - mean_r) ** 2 * g.p).sum() * dl)
            mu_model = Dt[m] * mean_r
            s_tot = np.sqrt(sig[m] ** 2 + (Dt[m] ** 2) * var_r)
            logL += float(
                (-0.5 * ((dL_obs[m] - mu_model) / s_tot) ** 2
                 - np.log(s_tot * SQRT2PI)).sum()
            )
        return logL if np.isfinite(logL) else -np.inf


class JointLikelihood:
    """Sum of per-arm likelihoods (ET + LISA), sharing one theta."""

    def __init__(self, likelihoods):
        self.parts = list(likelihoods)
        self.theta_keys = self.parts[0].theta_keys
        self.prior = self.parts[0].prior
        self.fiducial = self.parts[0].fiducial

    def log_prior(self, vec) -> float:
        return self.parts[0].log_prior(vec)

    def log_likelihood(self, vec) -> float:
        tot = 0.0
        for p in self.parts:
            v = p.log_likelihood(vec)
            if not np.isfinite(v):
                return -np.inf
            tot += v
        return tot

    def __call__(self, vec) -> float:
        lp = self.log_prior(vec)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(vec)
