import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'build')
import numpy as np
import gwlensing

KAPPATHR = 1e-4          # user decision: flat threshold
OM, S8, H = 0.315, 0.811, 0.674

def bench(z, subhalo, nreal, seed=123):
    t0 = time.perf_counter()
    out = gwlensing.sample_lnmu(
        z=z, OmegaM=OM, sigma8=S8, h=H,
        Nreal=nreal, seed=seed,
        subhalo=subhalo, subhalo_model=3,
        subhalo_factor=1.0e-5,
        kappathr_flat=KAPPATHR,
    )
    dt = time.perf_counter() - t0
    arr = np.asarray(out)
    return dt, arr.size

# small warm-up (build tables / caches) so first real timing isn't polluted
for z in (1.0,):
    bench(z, False, 200)
    bench(z, True, 100)

Z = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
# pick nreal per config so each timed run is ~a few seconds but bounded
NREAL_OFF = 20000
NREAL_ON  = 3000

print(f"{'z_s':>5} | {'off: s/1e4':>11} {'thr/s':>9} | {'m3: s/1e4':>11} {'thr/s':>9} | {'m3/off':>7}")
print("-"*70)
rows = []
for z in Z:
    dt_off, n_off = bench(z, False, NREAL_OFF)
    dt_on,  n_on  = bench(z, True,  NREAL_ON)
    per_off = dt_off / n_off * 1e4
    per_on  = dt_on  / n_on  * 1e4
    thr_off = n_off / dt_off
    thr_on  = n_on  / dt_on
    ratio   = per_on / per_off
    rows.append((z, per_off, thr_off, per_on, thr_on, ratio))
    print(f"{z:>5} | {per_off:>11.3f} {thr_off:>9.0f} | {per_on:>11.3f} {thr_on:>9.0f} | {ratio:>7.1f}x")

np.save('/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/34537cb7-1aef-4e68-81b8-af88b0b1a978/scratchpad/bench_rows.npy', np.array(rows))
print("\n# per_off, per_on are seconds per 10,000 realizations")
