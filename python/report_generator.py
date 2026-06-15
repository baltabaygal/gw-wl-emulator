import os
import sys
import time
import argparse
import h5py
import numpy as np
import shutil
import subprocess
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import DEFAULT_MIN_VALID_FRACTION

def get_split_stats(path):
    if not os.path.exists(path):
        return None
    with h5py.File(path, 'r') as f:
        z = f['samples/z'][:]
        h = f['samples/h'][:]
        om = f['samples/OmegaM'][:]
        s8 = f['samples/sigma8'][:]
        counts = f['samples/valid_counts'][:]
        lnmu = f['samples/lnmu'][:]
        split_types = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['samples/split_type'][:]]

        num_points = len(z)
        nsamples_per_point = lnmu.shape[1]
        total_slots = num_points * nsamples_per_point
        total_valid = int(np.sum(counts))
        total_nan = total_slots - total_valid
        valid_fraction = total_valid / total_slots if total_slots > 0 else 0.0

        # Parameter coverage
        coverage = {
            'z': (float(np.min(z)), float(np.max(z))),
            'h': (float(np.min(h)), float(np.max(h))),
            'OmegaM': (float(np.min(om)), float(np.max(om))),
            'sigma8': (float(np.min(s8)), float(np.max(s8))),
        }

        # Metadata attributes
        meta = f['metadata']
        metadata_dict = {k: (meta.attrs[k].decode('utf-8') if isinstance(meta.attrs[k], bytes) else meta.attrs[k]) for k in meta.attrs}
        # Parameter ranges group
        ranges_grp = f['metadata/parameter_ranges']
        ranges_dict = {k: ranges_grp.attrs[k].tolist() for k in ranges_grp.attrs}

        # Split type counts
        from collections import Counter
        type_counts = dict(Counter(split_types))

        return {
            'num_points': num_points,
            'nsamples_per_point': nsamples_per_point,
            'total_valid': total_valid,
            'total_nan': total_nan,
            'valid_fraction': valid_fraction,
            'coverage': coverage,
            'metadata': metadata_dict,
            'parameter_ranges': ranges_dict,
            'type_counts': type_counts,
        }

def run_reproducibility_check():
    # Helper to run generation and compare
    dir1 = "tmp_report_repro_1"
    dir2 = "tmp_report_repro_2"
    if os.path.exists(dir1): shutil.rmtree(dir1)
    if os.path.exists(dir2): shutil.rmtree(dir2)

    try:
        # Run generation 1
        subprocess.run([
            sys.executable, "python/generate_dataset.py",
            "--num_points", "3",
            "--nsamples", "50",
            "--seed", "99",
            "--output_dir", dir1
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Run generation 2
        subprocess.run([
            sys.executable, "python/generate_dataset.py",
            "--num_points", "3",
            "--nsamples", "50",
            "--seed", "99",
            "--output_dir", dir2
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Compare files
        identical = True
        for split in ["train", "validation", "test"]:
            f1_path = os.path.join(dir1, split, f"dataset_{split}.h5")
            f2_path = os.path.join(dir2, split, f"dataset_{split}.h5")
            exists1 = os.path.exists(f1_path)
            exists2 = os.path.exists(f2_path)
            if exists1 != exists2:
                identical = False
                break
            if not exists1:
                continue

            with h5py.File(f1_path, 'r') as f1, h5py.File(f2_path, 'r') as f2:
                for key in ["lnmu", "valid_counts", "z", "h", "OmegaM", "sigma8", "split_type"]:
                    p = f"samples/{key}"
                    if p not in f1 or p not in f2:
                        identical = False
                        break
                    d1 = f1[p][:]
                    d2 = f2[p][:]
                    if d1.dtype.kind == 'S' or d1.dtype.kind == 'O':
                        d1_str = [x.decode('utf-8') if isinstance(x, bytes) else x for x in d1]
                        d2_str = [x.decode('utf-8') if isinstance(x, bytes) else x for x in d2]
                        if d1_str != d2_str:
                            identical = False
                            break
                    else:
                        if not np.array_equal(d1, d2, equal_nan=True):
                            identical = False
                            break
                if not identical:
                    break
        return "PASSED" if identical else "FAILED"
    except Exception as e:
        return f"FAILED (Error: {e})"
    finally:
        if os.path.exists(dir1): shutil.rmtree(dir1)
        if os.path.exists(dir2): shutil.rmtree(dir2)

def generate_report(dataset_dir, output_report_path="docs/dataset_validation_report.md"):
    print(f"Generating report from dataset directory: {dataset_dir}...")
    
    splits = ["train", "validation", "test"]
    stats = {}
    
    for split in splits:
        path = os.path.join(dataset_dir, split, f"dataset_{split}.h5")
        split_stat = get_split_stats(path)
        if split_stat is not None:
            stats[split] = split_stat

    repro_status = run_reproducibility_check()

    lines = []
    lines.append("# Statistical Stability and Dataset Validation Report")
    lines.append("")
    lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"Dataset directory: `{dataset_dir}`")
    lines.append(f"Validation threshold (`DEFAULT_MIN_VALID_FRACTION`): **{DEFAULT_MIN_VALID_FRACTION:.2f}**")
    lines.append(f"Reproducibility status: **{repro_status}**")
    lines.append("")

    if not stats:
        lines.append("No dataset splits found in the directory.")
    else:
        lines.append("## 1. Split Summary")
        lines.append("")
        lines.append("| Split | Configurations | Samples/Config | Total Valid | Total NaN | Valid Fraction | Pass/Fail |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for split in splits:
            if split not in stats:
                continue
            s = stats[split]
            status = "PASS" if s['valid_fraction'] >= DEFAULT_MIN_VALID_FRACTION else "FAIL"
            lines.append(
                f"| {split} | {s['num_points']} | {s['nsamples_per_point']} | {s['total_valid']} | {s['total_nan']} | {s['valid_fraction']:.4f} | {status} |"
            )
        lines.append("")

        lines.append("## 2. Parameter Space Coverage (Measured Ranges)")
        lines.append("")
        lines.append("| Split | z Range | h Range | OmegaM Range | sigma8 Range |")
        lines.append("|---|---|---|---|---|")
        for split in splits:
            if split not in stats:
                continue
            s = stats[split]
            cov = s['coverage']
            lines.append(
                f"| {split} | [{cov['z'][0]:.4f}, {cov['z'][1]:.4f}] | [{cov['h'][0]:.4f}, {cov['h'][1]:.4f}] | [{cov['OmegaM'][0]:.4f}, {cov['OmegaM'][1]:.4f}] | [{cov['sigma8'][0]:.4f}, {cov['sigma8'][1]:.4f}] |"
            )
        lines.append("")

        lines.append("## 3. Split Configuration Types")
        lines.append("")
        lines.append("| Split | Train Configs | Interpolation Configs | OoD Configs |")
        lines.append("|---|---|---|---|")
        for split in splits:
            if split not in stats:
                continue
            s = stats[split]
            tc = s['type_counts']
            lines.append(
                f"| {split} | {tc.get('train', 0)} | {tc.get('interpolation', 0)} | {tc.get('ood', 0)} |"
            )
        lines.append("")

        lines.append("## 4. Metadata Provenance")
        lines.append("")
        for split in splits:
            if split not in stats:
                continue
            s = stats[split]
            meta = s['metadata']
            lines.extend([
                f"### {split.capitalize()} Split",
                f"- **Git Commit**: `{meta.get('git_commit', 'unknown')}`",
                f"- **Git Branch**: `{meta.get('git_branch', 'unknown')}`",
                f"- **Dataset Schema Version**: `{meta.get('dataset_schema_version', 'unknown')}`",
                f"- **Generation Seed**: `{meta.get('seed', 'unknown')}`",
                f"- **Generation Timestamp**: `{meta.get('generation_timestamp', 'unknown')}`",
                ""
            ])

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w', encoding='utf-8') as out_f:
        out_f.write("\n".join(lines) + "\n")

    print(f"Report written to {output_report_path}")
    return stats

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--output", type=str, default="docs/dataset_validation_report.md")
    parser.add_argument("--json_output", type=str, default=None)
    args = parser.parse_args()
    
    stats = generate_report(args.dataset_dir, args.output)
    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
        with open(args.json_output, 'w', encoding='utf-8') as jf:
            json.dump(stats, jf, cls=NumpyEncoder, indent=2)
        print(f"Metadata JSON written to {args.json_output}")
