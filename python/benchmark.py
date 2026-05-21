import sys
import os
import argparse
import time
import numpy as np
import h5py
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import gwlensing as gw

def run_benchmark():
    num_points = 5
    nsamples_per_point = 1000
    seed = 42

    z = 1.0
    h = 0.674
    OmegaM = 0.315
    sigma8 = 0.811

    print("Running Throughput Benchmarking...")

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1e6

    start_time = time.time()

    total_samples = 0
    lnmu_chunks = []

    for i in range(num_points):
        lnmu = gw.sample_lnmu_ml(z, h, OmegaM, sigma8, nsamples_per_point, seed + i)
        lnmu_chunks.append(lnmu)
        total_samples += len(lnmu)

    sim_time = time.time() - start_time
    mem_after = process.memory_info().rss / 1e6

    # Test HDF5 write throughput
    write_start = time.time()
    vlen_type = h5py.vlen_dtype(np.float64)
    with h5py.File("benchmark_tmp.h5", "w") as f:
        ds = f.create_dataset("lnmu", (num_points,), dtype=vlen_type)
        for i in range(num_points):
            ds[i] = lnmu_chunks[i]

    write_time = time.time() - write_start
    file_size = os.path.getsize("benchmark_tmp.h5") / 1e6
    os.remove("benchmark_tmp.h5")

    report = f"""# Performance Report

## Simulation Throughput
- **Points Simulated:** {num_points}
- **Samples Per Point:** {nsamples_per_point}
- **Total Valid Samples:** {total_samples}
- **Simulation Time:** {sim_time:.2f} s
- **Throughput (Points/sec):** {num_points / sim_time:.2f}
- **Throughput (Samples/sec):** {total_samples / sim_time:.0f}

## Memory Usage
- **Memory Before:** {mem_before:.2f} MB
- **Memory After (Holding {num_points} chunks):** {mem_after:.2f} MB
- **Memory Used by Simulation:** {mem_after - mem_before:.2f} MB

## HDF5 I/O Throughput
- **Write Time:** {write_time:.4f} s
- **File Size:** {file_size:.2f} MB
- **Write Throughput:** {file_size / write_time:.2f} MB/s
"""

    os.makedirs("docs", exist_ok=True)
    with open("docs/performance_report.md", "w") as f:
        f.write(report)

    print(report)
    print("Report saved to docs/performance_report.md")

if __name__ == "__main__":
    run_benchmark()
