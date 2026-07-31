"""Stage 2 -- mock catalogues.

Two arms, following Vaskonen 2026 sec.3 (and `cpp/main_lensing.cpp` :99-183):

  BNS/ET   : N = 300, z < 2,  sigma_DL = 0.03 DL, p_z ~ (1+z)^2.7 dVc/dz
  BMBH/LISA: N = 12,  z < 10, sigma_DL = 0.003 DL, p_z from an EPS+co-evolution
             list -- which we DO NOT HAVE (see SMBH_PLACEHOLDER below).

⚠ GENERATION ORDER IS PART OF THE PHYSICS (HDR memo sec.5):

    draw z  ->  draw mu  ->  form lensed dL  ->  THEN test detection.

Cutting before lensing is wrong: it is precisely the mu-dependence of detection
that creates the lensing Malmquist bias the likelihood then has to undo.

⚠ PLANE. mu is drawn from the SOURCE-plane PDF. A GW event is a random source,
not a random sky direction -- the same reason the C++ draws from `Plnmuf`, which
is source plane. This module never touches the conversion itself; it asks
pmu.PDFGrid.source_plane() and lets that single site own it.

⚠ The C++ SMBH arm has an unbounded out-of-bounds read (HDR memo sec.2). This
module does not reproduce it. It is not a bug-for-bug port.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from .background import Background
from .selection import RedshiftCut, SelectionRule, SNRCut


@dataclass
class Catalogue:
    """Detected events. `arm` and `selection` travel WITH the data, because
    analysing a catalogue under the wrong selection rule is the failure mode
    this whole package is organized around."""

    z: np.ndarray
    dL_obs: np.ndarray          # Mpc, lensed + measurement noise
    sigma_dL: np.ndarray        # Mpc
    arm: str
    selection: dict
    theta_fid: dict
    provider: str
    n_drawn: int = 0            # events proposed before selection
    n_accepted: int = 0         # events that PASSED selection (>= len(self),
                                # since generation stops on a batch boundary)
    truth: dict = field(default_factory=dict)   # unobservable diagnostics

    def __len__(self) -> int:
        return int(np.size(self.z))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            z=self.z,
            dL_obs=self.dL_obs,
            sigma_dL=self.sigma_dL,
            meta=json.dumps(
                dict(
                    arm=self.arm,
                    selection=self.selection,
                    theta_fid=self.theta_fid,
                    provider=self.provider,
                    n_drawn=self.n_drawn,
                    n_accepted=self.n_accepted,
                )
            ),
            **{f"truth_{k}": v for k, v in self.truth.items()},
        )

    @staticmethod
    def load(path) -> "Catalogue":
        d = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        truth = {k[6:]: d[k] for k in d.files if k.startswith("truth_")}
        return Catalogue(
            z=d["z"], dL_obs=d["dL_obs"], sigma_dL=d["sigma_dL"],
            truth=truth, **meta,
        )


# ---------------------------------------------------------------------------
# redshift distributions
# ---------------------------------------------------------------------------
def pz_bns(z, bg: Background, zeta: float = 2.7):
    """(1+z)^zeta dV_c/dz -- the C++ form (`main_lensing.cpp` :9-11).

    ⚠ This is the pure RISING branch, with no Madau-Dickinson turnover, exactly
    as the C++ has it. Harmless below z = 2 (MD peaks at z ~ 1.9); do not reuse
    at higher z without adding the turnover.
    """
    return (1.0 + np.asarray(z, float)) ** zeta * bg.dVc_dz(z)


SMBH_PLACEHOLDER = """\
⚠⚠ PLACEHOLDER, NOT VASKONEN'S CATALOGUE. His LISA redshifts come from
`zSMBHlist.dat`, generated from extended Press-Schechter + the Urrutia+25
MBH-galaxy co-evolution model. That file is NOT in this repo. What follows is a
smooth stand-in with a plausible shape (broad, peaked near z ~ 3, reaching 10),
chosen so the 12-event arm is TESTABLE, not so it is right. Any LISA-arm number
from this is a pipeline check, never a forecast. Replace by asking Ville for
zSMBHlist.dat (or regenerating it from his EPS code) and using EmpiricalZList.
"""


def pz_smbh_placeholder(z, bg: Background):
    """Broad lognormal-in-(1+z) stand-in peaked near z ~ 3. See SMBH_PLACEHOLDER."""
    z = np.asarray(z, float)
    x = np.log(1.0 + z)
    x0 = np.log(1.0 + 3.0)
    return np.exp(-0.5 * ((x - x0) / 0.55) ** 2) * bg.dVc_dz(z)


def _sample_z(pz_fn, zmin, zmax, n, rng, bg, ngrid=4000):
    zg = np.linspace(max(zmin, 1e-3), zmax, ngrid)
    w = np.asarray(pz_fn(zg, bg), float)
    w = np.clip(w, 0.0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, zg)


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------
def generate(
    n_events: int,
    provider,
    theta_fid: dict,
    selection: SelectionRule,
    *,
    arm: str = "bns_et",
    z_range=(0.01, 2.0),
    frac_sigma_dL=0.03,
    pz_fn=None,
    rng: np.random.Generator | None = None,
    max_batches: int = 10000,
) -> Catalogue:
    """Draw `n_events` DETECTED events.

    Rejection loop over batches; `n_drawn` records how many were proposed, so
    the selection's effect on the sample is measurable after the fact.
    """
    rng = np.random.default_rng() if rng is None else rng
    bg = Background(theta_fid)
    pz_fn = pz_bns if pz_fn is None else pz_fn
    zmin, zmax = z_range

    # ⚠ mu is drawn at each event's EXACT z, not at a node.
    #
    # An earlier version cached P(mu) on a 24-node z grid while the likelihood
    # used its own (coarser) node set. sigma(ln mu) grows steeply with z, so a
    # node offset of dz ~ 0.1 shifts the width by ~15% -- a model
    # misspecification between stage 2 and stage 3 large enough to bias the
    # recovered parameters by ~1 sigma, which is exactly the class of silent
    # error this package exists to avoid. The likelihood still node-discretizes
    # (as Vaskonen's does, with 6 nodes); that approximation is then the
    # LIKELIHOOD's alone and is measured by gate_znodes, instead of being
    # entangled with the truth.

    keep = {k: [] for k in ("z", "dL_obs", "sigma_dL", "mu", "dL_true", "w")}
    n_drawn = n_accepted = 0
    for _ in range(max_batches):
        # Adaptive batch size. A fixed 4x over-draw wastes a per-event P(mu)
        # build on every discarded candidate, which dominates the cost at large
        # N under a permissive rule (a z-cut passes ~100%, so 3/4 of the work
        # was thrown away). Size the next batch from the acceptance measured so
        # far, with a 30% margin and a floor.
        need = n_events - sum(len(a) for a in keep["z"])
        rate = (n_accepted / n_drawn) if n_drawn > 0 else 1.0
        rate = min(max(rate, 1e-3), 1.0)
        batch = int(min(max(1.3 * need / rate, 256), 200_000))
        # 1. redshift
        z = _sample_z(pz_fn, zmin, zmax, batch, rng, bg)
        # 2. magnification, from the SOURCE-plane PDF at that event's own z
        lnmu = np.array([
            provider(float(zz), theta_fid).source_plane().sample(1, rng)[0]
            for zz in z
        ])
        mu = np.exp(lnmu)
        # 3. lensed luminosity distance
        dL_true = bg.DL(z) / np.sqrt(mu)
        # 4. NOW test detection
        if isinstance(selection, RedshiftCut):
            acc = selection.accept_z(z)
            w = np.ones(batch)
        elif isinstance(selection, SNRCut):
            w = selection.sample_w(batch, rng)
            acc = selection.accept_with_iota(dL_true, w)
        else:
            w = np.ones(batch)
            acc = selection.accept(dL_true, rng)
        n_drawn += batch
        n_accepted += int(np.count_nonzero(acc))

        # 5. measurement noise (applied to detected events)
        zk, muk, dk, wk = z[acc], mu[acc], dL_true[acc], w[acc]
        # frac_sigma_dL may be a scalar (Vaskonen's flat 3% / 0.3%) or a
        # CALLABLE (z, dL, w) -> sigma_dL, which is the hook for a per-event
        # Fisher forecast a la GWFish (Dupletsa+23) -- the upgrade Ville flagged
        # in De Leo+ 2026. Nothing downstream changes: Catalogue already stores
        # sigma_dL per event and the likelihood already reads it per event.
        # ⚠ The likelihood treats sigma_j as theta-INDEPENDENT (it is a number
        # attached to the observed event). That holds for a Fisher forecast made
        # from the observed signal; it would NOT hold for a sigma computed from
        # the trial cosmology, which would resurrect the Gaussian normalization
        # in the theta-dependence.
        sig = (np.asarray(frac_sigma_dL(zk, dk, wk), float)
               if callable(frac_sigma_dL) else frac_sigma_dL * dk)
        obs = rng.normal(dk, sig)
        good = obs > 0
        keep["z"].append(zk[good])
        keep["mu"].append(muk[good])
        keep["dL_true"].append(dk[good])
        keep["w"].append(wk[good])
        keep["dL_obs"].append(obs[good])
        keep["sigma_dL"].append(sig[good])
        if sum(len(a) for a in keep["z"]) >= n_events:
            break
    else:
        raise RuntimeError(
            f"selection {selection.name} too aggressive: only "
            f"{sum(len(a) for a in keep['z'])}/{n_events} events in "
            f"{max_batches} batches"
        )

    cat = {k: np.concatenate(v)[:n_events] for k, v in keep.items()}
    return Catalogue(
        z=cat["z"],
        dL_obs=cat["dL_obs"],
        sigma_dL=cat["sigma_dL"],
        arm=arm,
        selection=selection.describe(),
        theta_fid=dict(theta_fid),
        provider=type(provider).__name__,
        n_drawn=n_drawn,
        n_accepted=n_accepted,
        truth=dict(mu=cat["mu"], dL_true=cat["dL_true"], w=cat["w"]),
    )


def generate_bns_et(provider, theta_fid, selection=None, n_events=300, rng=None):
    """Vaskonen's ET arm: 300 BNS, z < 2, 3% distance errors."""
    sel = RedshiftCut(2.0) if selection is None else selection
    return generate(
        n_events, provider, theta_fid, sel, arm="bns_et",
        z_range=(0.01, 2.0), frac_sigma_dL=0.03, pz_fn=pz_bns, rng=rng,
    )


def generate_bmbh_lisa(provider, theta_fid, selection=None, n_events=12, rng=None):
    """Vaskonen's LISA arm: 12 BMBH, z < 10, 0.3% distance errors.

    ⚠ Uses the placeholder p_z -- see SMBH_PLACEHOLDER.
    """
    sel = RedshiftCut(10.0) if selection is None else selection
    return generate(
        n_events, provider, theta_fid, sel, arm="bmbh_lisa",
        z_range=(0.05, 10.0), frac_sigma_dL=0.003,
        pz_fn=pz_smbh_placeholder, rng=rng,
    )
