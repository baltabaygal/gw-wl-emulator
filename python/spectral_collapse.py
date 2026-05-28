import json
import numpy as np

from validate_transport import (
    make_x_grid,
    sample_lnmu,
    lnmu_to_px,
    fft_phi,
    lambda_eff,
)


RELIABILITY_FLOOR = 2e-2


def extract_normalized_spectrum(z, nreal=50000):
    x, dx = make_x_grid()

    lnmu = sample_lnmu(
        z=z,
        Nreal=nreal,
    )

    p_x = lnmu_to_px(lnmu, x)
    phi = fft_phi(p_x, dx)
    lam = lambda_eff(phi, z)

    reliable = np.abs(phi) > RELIABILITY_FLOOR

    # use saturation / large-k amplitude as normalization scale
    amp = np.max(np.abs(np.real(lam[reliable]))) if np.any(reliable) else np.inf

    if np.isfinite(amp) and amp > 0:
        lam_norm = lam / amp
    else:
        lam_norm = lam

    return {
        "z": z,
        "lambda": lam,
        "lambda_norm": lam_norm,
        "phi": phi,
        "reliable": reliable,
        "amplitude": float(amp),
    }



def collapse_statistics(results):
    masks = [r["reliable"] for r in results]
    joint = np.logical_and.reduce(masks)

    if not np.any(joint):
        return {
            "joint_modes": 0,
            "median_shape_scatter": np.inf,
            "max_shape_scatter": np.inf,
        }

    spectra = np.array([
        np.real(r["lambda_norm"])[joint]
        for r in results
    ])

    mean_spec = np.mean(spectra, axis=0)
    scatter = np.std(spectra, axis=0)

    denom = np.maximum(np.abs(mean_spec), 1e-12)
    rel_scatter = scatter / denom

    return {
        "joint_modes": int(np.count_nonzero(joint)),
        "median_shape_scatter": float(np.median(rel_scatter)),
        "max_shape_scatter": float(np.max(rel_scatter)),
    }



def run_collapse_test(z_values=(0.5, 1.0, 2.0, 5.0), nreal=50000):
    results = [extract_normalized_spectrum(z, nreal=nreal) for z in z_values]
    stats = collapse_statistics(results)

    summary = {
        "redshifts": list(z_values),
        "collapse_statistics": stats,
        "amplitudes": {
            str(r["z"]): r["amplitude"]
            for r in results
        },
    }

    return summary


if __name__ == "__main__":
    out = run_collapse_test()
    print(json.dumps(out, indent=2, sort_keys=True))
