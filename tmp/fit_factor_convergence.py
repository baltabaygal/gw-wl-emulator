#!/usr/bin/env python3
"""Fit zs=1 subhalo dynamic-split convergence against brute-force reference."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
DATA = ROOT / "data"
OUT = ROOT / "tmp" / "fit_factor_convergence_report.txt"

NPZ = DATA / "subhalo_factor_convergence_z1.npz"
REDSHIFT_CSV = DATA / "subhalo_factor_redshift_check.csv"
TAIL_CSV = DATA / "subhalo_factor_z1_tail.csv"

ZS_TARGET = 1.0
FACTOR_MAX = 1e-2


@dataclass(frozen=True)
class Point:
    factor: float
    excess: float
    err: float
    source: str
    seed: str | int | None = None
    n: int | None = None


@dataclass(frozen=True)
class Brute:
    excess: float
    err: float
    source: str
    seed: str | int | None = None
    n: int | None = None


def finite_positive(*xs: float) -> bool:
    return all(np.isfinite(x) and x > 0 for x in xs)


def fmt(x: float, sig: int = 6) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{sig}g}"


def fmt_pm(x: float, e: float, sig: int = 6) -> str:
    return f"{fmt(x, sig)} +- {fmt(e, sig)}"


def read_npz() -> tuple[list[Point], list[Brute]]:
    points: list[Point] = []
    brutes: list[Brute] = []
    with np.load(NPZ) as d:
        factors = np.asarray(d["factors"], dtype=float)
        excess = np.asarray(d["excess"], dtype=float)
        err = np.asarray(d["err"], dtype=float)
        zs = float(d["zs"]) if "zs" in d else ZS_TARGET
        n = int(d["N"]) if "N" in d else None
        if math.isclose(zs, ZS_TARGET, rel_tol=0.0, abs_tol=1e-12):
            for f, y, s in zip(factors, excess, err):
                if finite_positive(f, s) and np.isfinite(y):
                    points.append(Point(f, y, s, "npz", seed=42, n=n))
        brutes.append(
            Brute(
                float(d["excess_brute"]),
                float(d["err_brute"]),
                "npz",
                seed=42,
                n=n,
            )
        )
    return points, brutes


def read_redshift_csv() -> list[Point]:
    points: list[Point] = []
    if not REDSHIFT_CSV.exists():
        return points
    with REDSHIFT_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if not math.isclose(float(row["zs"]), ZS_TARGET, rel_tol=0.0, abs_tol=1e-12):
                continue
            f = float(row["factor"])
            y = float(row["excess"])
            s = float(row["err"])
            if finite_positive(f, s) and np.isfinite(y):
                points.append(
                    Point(
                        f,
                        y,
                        s,
                        "redshift_csv",
                        seed=row.get("seed"),
                        n=int(row["N"]) if row.get("N") else None,
                    )
                )
    return points


def read_tail_csv() -> tuple[list[Point], list[Brute], str]:
    points: list[Point] = []
    brutes: list[Brute] = []
    if not TAIL_CSV.exists():
        return points, brutes, f"tail CSV absent: {TAIL_CSV}"

    rows_total = 0
    rows_z1 = 0
    with TAIL_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows_total += 1
            if not math.isclose(float(row["zs"]), ZS_TARGET, rel_tol=0.0, abs_tol=1e-12):
                continue
            rows_z1 += 1
            label = row["label"].strip()
            y = float(row["excess"])
            s = float(row["err"])
            seed = row.get("seed")
            n = int(row["N"]) if row.get("N") else None
            if not finite_positive(s) or not np.isfinite(y):
                continue
            if label.lower() == "brute":
                brutes.append(Brute(y, s, "tail_csv", seed=seed, n=n))
            else:
                f = float(label)
                if finite_positive(f):
                    points.append(Point(f, y, s, "tail_csv", seed=seed, n=n))

    status = (
        f"tail CSV present: {TAIL_CSV}; rows={rows_total}, zs=1 rows={rows_z1}, "
        f"dynamic rows={len(points)}, brute rows={len(brutes)}"
    )
    return points, brutes, status


def dedupe_points(points_by_priority: list[Point]) -> tuple[list[Point], list[str]]:
    selected: list[Point] = []
    notes: list[str] = []
    for p in points_by_priority:
        if p.factor > FACTOR_MAX:
            continue
        duplicate = None
        for i, existing in enumerate(selected):
            if np.isclose(p.factor, existing.factor, rtol=5e-4, atol=0.0):
                duplicate = i
                break
        if duplicate is None:
            selected.append(p)
        else:
            notes.append(
                f"deduped factor {p.factor:.12g} from {p.source} as duplicate of "
                f"{selected[duplicate].factor:.12g} from {selected[duplicate].source}"
            )
    return sorted(selected, key=lambda p: p.factor), notes


def invvar_mean(values: np.ndarray, errs: np.ndarray) -> tuple[float, float]:
    w = 1.0 / np.square(errs)
    mean = float(np.sum(w * values) / np.sum(w))
    err = float(np.sqrt(1.0 / np.sum(w)))
    return mean, err


def model_free(f: np.ndarray, y: np.ndarray, s: np.ndarray, limit: float) -> tuple[float, float, int]:
    mask = f <= limit
    if not np.any(mask):
        return np.nan, np.nan, 0
    mean, err = invvar_mean(y[mask], s[mask])
    return mean, err, int(np.sum(mask))


def recruitment_model(f: np.ndarray, e_inf: float, f0: float, p: float) -> np.ndarray:
    return e_inf * (1.0 - np.power(f / f0, p))


def recruitment_fixed_p(p_fixed: float):
    def fn(f: np.ndarray, e_inf: float, f0: float) -> np.ndarray:
        return recruitment_model(f, e_inf, f0, p_fixed)

    return fn


def fit_free(f: np.ndarray, y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    e0 = max(float(np.max(y)), 1e-12)
    p0 = [e0, max(float(np.max(f)) * 10.0, 1e-2), 0.5]
    bounds = ([1e-15, 1e-15, 1e-6], [np.inf, np.inf, 10.0])
    popt, pcov = curve_fit(
        recruitment_model,
        f,
        y,
        p0=p0,
        sigma=s,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=200000,
    )
    resid = (y - recruitment_model(f, *popt)) / s
    chi2 = float(np.sum(np.square(resid)))
    dof = len(f) - len(popt)
    return popt, pcov, chi2, dof


def fit_fixed(f: np.ndarray, y: np.ndarray, s: np.ndarray, p_fixed: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    e0 = max(float(np.max(y)), 1e-12)
    p0 = [e0, max(float(np.max(f)) * 10.0, 1e-2)]
    bounds = ([1e-15, 1e-15], [np.inf, np.inf])
    fn = recruitment_fixed_p(p_fixed)
    popt, pcov = curve_fit(
        fn,
        f,
        y,
        p0=p0,
        sigma=s,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=200000,
    )
    resid = (y - fn(f, *popt)) / s
    chi2 = float(np.sum(np.square(resid)))
    dof = len(f) - len(popt)
    return popt, pcov, chi2, dof


def sample_params(popt: np.ndarray, pcov: np.ndarray, nsamp: int = 200000) -> np.ndarray:
    rng = np.random.default_rng(12345)
    pcov = np.asarray(pcov, dtype=float)
    if not np.all(np.isfinite(pcov)):
        return np.empty((0, len(popt)))
    samples = rng.multivariate_normal(popt, pcov, size=nsamp, check_valid="ignore")
    mask = np.all(samples > 0.0, axis=1) & np.all(np.isfinite(samples), axis=1)
    return samples[mask]


def summarize_samples(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    q16, q50, q84 = np.percentile(values, [15.865525, 50.0, 84.134475])
    return float(q50), float(q50 - q16), float(q84 - q50)


def main() -> None:
    npz_points, npz_brutes = read_npz()
    redshift_points = read_redshift_csv()
    tail_points, tail_brutes, tail_status = read_tail_csv()

    all_priority_points = tail_points + redshift_points + npz_points
    points, dedupe_notes = dedupe_points(all_priority_points)
    brutes = npz_brutes + tail_brutes

    f = np.array([p.factor for p in points], dtype=float)
    y = np.array([p.excess for p in points], dtype=float)
    s = np.array([p.err for p in points], dtype=float)

    brute_y = np.array([b.excess for b in brutes], dtype=float)
    brute_s = np.array([b.err for b in brutes], dtype=float)
    e_brute, err_brute = invvar_mean(brute_y, brute_s)

    popt, pcov, chi2, dof = fit_free(f, y, s)
    perr = np.sqrt(np.diag(pcov))
    fixed_half, fixed_half_cov, chi2_half, dof_half = fit_fixed(f, y, s, 0.5)
    fixed_third, fixed_third_cov, chi2_third, dof_third = fit_fixed(f, y, s, 1.0 / 3.0)

    e_inf, f0, p_best = popt
    e_inf_err, f0_err, p_err = perr
    diff_sigma = (e_inf - e_brute) / math.sqrt(e_inf_err**2 + err_brute**2)

    samples = sample_params(popt, pcov)
    missing_stats = {}
    threshold_stats = {}
    if len(samples):
        e_s, f0_s, p_s = samples[:, 0], samples[:, 1], samples[:, 2]
        for factor in (1e-5, 1e-6):
            missing = np.power(factor / f0_s, p_s)
            missing_stats[factor] = summarize_samples(missing)
        for frac in (0.05, 0.02):
            threshold = f0_s * np.power(frac, 1.0 / p_s)
            threshold_stats[frac] = summarize_samples(threshold)
    else:
        for factor in (1e-5, 1e-6):
            missing_stats[factor] = (np.nan, np.nan, np.nan)
        for frac in (0.05, 0.02):
            threshold_stats[frac] = (np.nan, np.nan, np.nan)

    mf_mean, mf_err, mf_n = model_free(f, y, s, 1e-5)
    mf_sigma = (mf_mean - e_brute) / math.sqrt(mf_err**2 + err_brute**2) if mf_n else np.nan

    lines: list[str] = []
    lines.append("Fit-factor convergence report")
    lines.append("=============================")
    lines.append("")
    lines.append(f"Inputs:")
    lines.append(f"- NPZ: {NPZ}")
    lines.append(f"- redshift CSV: {REDSHIFT_CSV} ({'present' if REDSHIFT_CSV.exists() else 'absent'})")
    lines.append(f"- {tail_status}")
    lines.append("")
    lines.append("Selection:")
    lines.append(f"- zs = {ZS_TARGET:g}")
    lines.append(f"- dynamic-split fit gate: factor <= {FACTOR_MAX:g}")
    lines.append("- duplicate factors resolved by priority: tail CSV, then redshift CSV, then NPZ")
    lines.append("- duplicate tolerance: relative 5e-4, to merge CSV-truncated sqrt-decade factors")
    lines.append(f"- retained dynamic points: {len(points)}")
    lines.append(f"- pooled brute measurements: {len(brutes)}")
    if dedupe_notes:
        lines.append("- duplicate notes:")
        for note in dedupe_notes:
            lines.append(f"  {note}")
    lines.append("")
    lines.append("Retained dynamic-split points:")
    lines.append("factor, excess, err, source, seed, N")
    for pnt in points:
        lines.append(
            f"{pnt.factor:.12g}, {pnt.excess:.12g}, {pnt.err:.12g}, "
            f"{pnt.source}, {pnt.seed}, {pnt.n}"
        )
    lines.append("")
    lines.append("Brute-force pool:")
    for brute in brutes:
        lines.append(
            f"- {brute.source}, seed={brute.seed}, N={brute.n}: "
            f"{fmt_pm(brute.excess, brute.err)}"
        )
    lines.append(f"- inverse-variance pooled E_brute = {fmt_pm(e_brute, err_brute)}")
    lines.append("")
    lines.append("Weighted recruitment-model fits:")
    lines.append("model: excess(f) = E_inf * (1 - (f/f0)^p)")
    lines.append(
        f"- free p: E_inf = {fmt_pm(e_inf, e_inf_err)}, "
        f"f0 = {fmt_pm(f0, f0_err)}, p = {fmt_pm(p_best, p_err)}, "
        f"chi2/dof = {fmt(chi2)}/{dof} = {fmt(chi2 / dof if dof > 0 else np.nan)}"
    )
    lines.append(
        f"- fixed p=1/2: E_inf = {fmt_pm(fixed_half[0], math.sqrt(fixed_half_cov[0, 0]))}, "
        f"f0 = {fmt_pm(fixed_half[1], math.sqrt(fixed_half_cov[1, 1]))}, "
        f"chi2/dof = {fmt(chi2_half)}/{dof_half} = {fmt(chi2_half / dof_half if dof_half > 0 else np.nan)}"
    )
    lines.append(
        f"- fixed p=1/3: E_inf = {fmt_pm(fixed_third[0], math.sqrt(fixed_third_cov[0, 0]))}, "
        f"f0 = {fmt_pm(fixed_third[1], math.sqrt(fixed_third_cov[1, 1]))}, "
        f"chi2/dof = {fmt(chi2_third)}/{dof_third} = {fmt(chi2_third / dof_third if dof_third > 0 else np.nan)}"
    )
    lines.append("")
    lines.append("Convergence tests:")
    lines.append(
        f"- E_inf - E_brute_pooled = {fmt(e_inf - e_brute)}; "
        f"combined sigma = {fmt(math.sqrt(e_inf_err**2 + err_brute**2))}; "
        f"offset = {fmt(diff_sigma)} sigma"
    )
    for factor in (1e-5, 1e-6):
        med, lo, hi = missing_stats[factor]
        lines.append(
            f"- predicted missing fraction at f={factor:g}: "
            f"{fmt(med)} -{fmt(lo)} +{fmt(hi)}"
        )
    for frac in (0.05, 0.02):
        med, lo, hi = threshold_stats[frac]
        lines.append(
            f"- missing fraction drops below {100 * frac:g}% for factor <= "
            f"{fmt(med)} -{fmt(lo)} +{fmt(hi)}"
        )
    lines.append("")
    lines.append("Model-free low-factor check:")
    lines.append(
        f"- inverse-variance mean of points with f <= 1e-5 (n={mf_n}): "
        f"{fmt_pm(mf_mean, mf_err)}"
    )
    lines.append(
        f"- low-factor mean - E_brute_pooled = {fmt(mf_mean - e_brute)}; "
        f"combined sigma = {fmt(math.sqrt(mf_err**2 + err_brute**2))}; "
        f"offset = {fmt(mf_sigma)} sigma"
    )
    lines.append("")

    report = "\n".join(lines)
    OUT.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
