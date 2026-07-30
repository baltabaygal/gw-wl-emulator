"""Detection rules -- ONE object, used by both the catalogue and the likelihood.

This module exists because of HDR memo sec.5: the likelihood must encode the same
detection rule that produced the catalogue. The structural guarantee here is that
stage 2 and stage 3 import the SAME object; a mismatch has to be requested
explicitly (which the bias demo in run_forecast.py --demo-mismatch does, on
purpose).

A rule provides two things:

    accept(dL_lensed, rng, **props) -> bool mask   (stage 2: who gets in)
    p_det(dL_lensed)                -> probability (stage 3: normalization)

and declares `mu_coupled`: whether detection depends on the magnification. That
flag is what separates the two published regimes:

  * RedshiftCut (Vaskonen)  -- mu_coupled = False. Lensing does not change z, so
    at fixed z the catalogue's mu are a FAIR sample of p_mu and P_det is a
    mu-independent constant that cancels in the likelihood ratio.
  * DistanceThreshold / SNRCut (De Leo-like) -- mu_coupled = True. Magnified
    events look closer, are louder, and are detected when their unmagnified
    twins are not => catalogue mu are skewed high (lensing-induced Malmquist
    bias). P_det must then be marginalized over mu, at the trial theta.

Literature for the P_det-normalized likelihood and the magnification-selection
coupling: HDR memo sec.5 "Literature" (Mandel+19, Loredo 04, Mortlock+19;
Dai+17, Oguri 18, Cusin & Tamanini 21, Canevarolo & Chisari 24, Mpetha+24).

⚠ SCOPE. The SNR rule here is a *distance-threshold caricature* with an optional
inclination factor. It is NOT a GWFish Fisher forecast and it does not model
masses or sky position. It is enough to create -- and therefore to test the
correction of -- the mu-coupling, which is the structural point. Swapping in a
real P_det(dL, iota, ...) means replacing ONE method, `p_det_of_dL`.
"""
from __future__ import annotations

import numpy as np


class SelectionRule:
    """Base class. Subclasses set `mu_coupled` and implement p_det_of_dL."""

    name = "base"
    mu_coupled = False

    def p_det_of_dL(self, dL: np.ndarray) -> np.ndarray:
        """P(detect | lensed luminosity distance dL). Vectorized, in [0, 1]."""
        raise NotImplementedError

    def accept(self, dL: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Bernoulli draw against p_det (stage 2)."""
        p = self.p_det_of_dL(np.asarray(dL, float))
        return rng.random(np.shape(p)) < p

    def describe(self) -> dict:
        return dict(name=self.name, mu_coupled=self.mu_coupled)


class RedshiftCut(SelectionRule):
    """Vaskonen's rule: keep everything below z_thr, regardless of mu.

    P_det is mu-INDEPENDENT (identically 1 for events in the catalogue), so it
    is a theta-independent constant per event and cancels. This is exactly why
    his paper can state that selection effects do not bias the inference -- and
    reproducing that no-op is a gate, not a triviality: the pipeline must give
    the same posterior with p_det on and off under this rule.
    """

    name = "redshift_cut"
    mu_coupled = False

    def __init__(self, z_thr: float):
        self.z_thr = float(z_thr)

    def p_det_of_dL(self, dL: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(dL, float))

    def accept_z(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, float) < self.z_thr

    def describe(self) -> dict:
        return dict(super().describe(), z_thr=self.z_thr)


class DistanceThreshold(SelectionRule):
    """Sharp cut in the LENSED luminosity distance: detect iff dL < dL_thr.

    This is the rule Vaskonen's own C++ supports via the `DLthr` argument (HDR
    memo sec.5) and never exercises (the driver hardcodes 300 Gpc). It is
    mu-coupled: dL = Dtilde/sqrt(mu), so a magnified event has a smaller dL and
    can pass a cut its unmagnified twin fails.
    """

    name = "distance_threshold"
    mu_coupled = True

    def __init__(self, dL_thr: float):
        self.dL_thr = float(dL_thr)

    def p_det_of_dL(self, dL: np.ndarray) -> np.ndarray:
        return (np.asarray(dL, float) < self.dL_thr).astype(float)

    def describe(self) -> dict:
        return dict(super().describe(), dL_thr_Mpc=self.dL_thr)


class SNRCut(SelectionRule):
    """SNR > SNR_thr with SNR = A * w(iota) / dL, marginalized over inclination.

    The caricature (see module ⚠): amplitude A absorbs masses/detector/sky, and
    the antenna/inclination factor uses the standard quadrature form for a
    circular binary,

        w(iota)^2 = [ (1+cos^2 iota)/2 ]^2 + cos^2 iota   (normalized to face-on)

    with cos iota ~ U(-1, 1) (De Leo+ draw it this way). Marginalizing over
    iota turns a sharp cut into a smooth, monotone P_det(dL) -- which is the
    qualitative feature that matters: real detection probabilities are graded,
    not step functions, and a graded P_det is what stops the likelihood from
    seeing a hard support boundary.

    P_det(dL) = P( w(iota) > SNR_thr * dL / A ) computed on a cos-iota grid.

    ⚠ Not a Fisher forecast. When GWFish is wired, replace p_det_of_dL.
    """

    name = "snr_cut"
    mu_coupled = True

    def __init__(self, amplitude: float, snr_thr: float = 20.0, n_iota: int = 2048):
        self.amplitude = float(amplitude)      # SNR of a face-on source at 1 Mpc
        self.snr_thr = float(snr_thr)
        ci = np.linspace(-1.0, 1.0, int(n_iota))
        w2 = ((1.0 + ci**2) / 2.0) ** 2 + ci**2
        self._w = np.sqrt(w2 / w2.max())       # normalized so face-on = 1
        self._w_sorted = np.sort(self._w)

    @classmethod
    def from_horizon(cls, dL_horizon: float, snr_thr: float = 20.0, **kw):
        """Calibrate so a FACE-ON source at dL_horizon sits exactly at threshold."""
        return cls(amplitude=snr_thr * float(dL_horizon), snr_thr=snr_thr, **kw)

    def p_det_of_dL(self, dL: np.ndarray) -> np.ndarray:
        dL = np.asarray(dL, float)
        # need w > w_min := snr_thr * dL / A
        w_min = self.snr_thr * dL / self.amplitude
        # fraction of the cos-iota grid with w above w_min
        idx = np.searchsorted(self._w_sorted, w_min)
        return 1.0 - idx / self._w_sorted.size

    def sample_w(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw the per-event inclination factor (stage 2 uses this so the
        catalogue's own iota is realized, not marginalized)."""
        ci = rng.uniform(-1.0, 1.0, n)
        w2 = ((1.0 + ci**2) / 2.0) ** 2 + ci**2
        w2max = ((1.0 + 1.0) / 2.0) ** 2 + 1.0
        return np.sqrt(w2 / w2max)

    def accept_with_iota(self, dL: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Deterministic acceptance given a realized inclination factor."""
        return self.amplitude * np.asarray(w, float) / np.asarray(dL, float) > self.snr_thr

    def describe(self) -> dict:
        return dict(
            super().describe(),
            snr_thr=self.snr_thr,
            amplitude=self.amplitude,
            dL_horizon_faceon_Mpc=self.amplitude / self.snr_thr,
        )


RULES = {
    "redshift_cut": RedshiftCut,
    "distance_threshold": DistanceThreshold,
    "snr_cut": SNRCut,
}
