"""Validate the ANALYTIC Campbell quadrature against the PRODUCTION C++ subhalo sampler.

The population-weighted kappa_thr,sub sizing (data/results/subkappathr_population/) is
pure analytic: it never runs the engine. That is only trustworthy if the analytic radial
profile, bias scale and psi convention agree with cpp/subhalo.cpp. Historically they did
not -- the analytic carried bias x0 = 0.54 in r_200 units and extent x <= 1, while the
engine uses x0 = BIAS_X0_RVIR(0.86) * eta(c200,z) and, with subhalo_virial (ON in
PRODUCTION_CONFIG), extent x <= eta with psi = m/M_vir.

Oracle: playground/subhalo_single_host_probe.cpp calls the production
Subhalo::precompute + addClumps (model 4, brute to the floor, mass-conserving carve) on
one grid host over a grid of ray impact parameters, so its <N_c>, <Sum m_i>, <kappa_sub>
and sigma_sub ARE the engine's.

Run:
  ./playground/subhalo_single_host_probe 2000 1e14 0.5 1.0 1e7 20260728 tmp/probe_v0.csv 0
  ./playground/subhalo_single_host_probe 2000 1e14 0.5 1.0 1e7 20260728 tmp/probe_v1.csv 1
  /Users/baltabay/miniforge3/envs/test/bin/python \
      scripts/convergence/check_analytic_vs_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from playground.analytic.sigma_vs_subkappathr import run_kthr  # noqa: E402


def read_probe(path: Path):
    head = {}
    for tok in path.read_text().splitlines()[0].lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            head[k] = float(v)
    d = np.genfromtxt(path, delimiter=",", names=True, skip_header=1)
    return head, d


def compare(tag: str, probe_path: Path, virial: bool):
    head, d = read_probe(probe_path)
    M, zl, zs = head["M"], head["zl"], head["zs"]
    print(f"\n{'='*78}\n{tag}: M={M:.4e} z_l={zl:.4f} z_s={zs} virial={int(virial)}")
    print(f"  engine: r200={head['r200']:.2f} c={head['c_host']:.4f} "
          f"fs={head['fs_jvdb14']:.5f} xmax={head.get('xmax', 1.0):.5f} "
          f"Mpsi/M={head.get('Mpsi', M)/M:.5f}")

    x = d["x_r200"]
    # kappa_thr_host: the probe passes 1e-2 to precompute, but model 4 never consults
    # r_thr, so any value gives the same population. Use the production-scale host
    # threshold so the analytic aperture rmax_host matches the figure convention.
    meta, rows = run_kthr(M, zl, zs, 1.276e-4, [0.0], m_floor=1.0e7,
                          y_probe_r200=x, virial=virial)
    r = rows[0]["probe"]
    print(f"  analytic: psi_min={meta['psi_min']:.4e}  fd_norm_err={meta['fd_norm_err']:.2e}")

    def row(name, a, b, fmt="{:11.5g}"):
        ratio = np.array(b) / np.where(np.array(a) == 0, np.nan, np.array(a))
        print(f"  {name:<12} engine[0]={fmt.format(a[0])} analytic[0]={fmt.format(b[0])} "
              f"| ratio med={np.nanmedian(ratio):.4f} "
              f"range=[{np.nanmin(ratio):.4f},{np.nanmax(ratio):.4f}]")
        return ratio

    print(f"\n  {'quantity':<12} {'engine(x=0.03)':>16} {'analytic':>12}   ratio analytic/engine")
    rN = row("<N_c>", d["mean_Nc"], r["n_retained"])
    rk = row("<kappa_sub>", d["mean_sub"], r["kappa_sub"])
    rs = row("sigma_sub", d["sig_sub"], r["sigma_sub"])
    rt = row("kappa_total", d["mean_tot"], r["kappa_total"])

    print(f"\n  {'x/r200':>7} {'Nc_eng':>10} {'Nc_ana':>10} {'ksub_eng':>11} "
          f"{'ksub_ana':>11} {'ratio':>7}")
    for i in range(0, len(x), 4):
        print(f"  {x[i]:7.3f} {d['mean_Nc'][i]:10.1f} {r['n_retained'][i]:10.1f} "
              f"{d['mean_sub'][i]:11.5f} {r['kappa_sub'][i]:11.5f} {rk[i]:7.4f}")
    return {"N": rN, "kappa_sub": rk, "sigma_sub": rs, "total": rt}


if __name__ == "__main__":
    out = {}
    for tag, p, v in (("virial OFF", ROOT / "tmp/probe_v0.csv", False),
                      ("virial ON (PRODUCTION)", ROOT / "tmp/probe_v1.csv", True)):
        if p.exists():
            out[tag] = compare(tag, p, v)
        else:
            print(f"missing {p} -- run the probe first")

    print(f"\n{'='*78}\nSUMMARY (median ratio analytic/engine; 1.000 = conventions agree)")
    for tag, r in out.items():
        print(f"  {tag:<24} "
              + "  ".join(f"{k}={np.nanmedian(v):.4f}" for k, v in r.items()))
