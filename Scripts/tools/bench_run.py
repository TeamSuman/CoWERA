"""Summarize a CoWERA profiling run into a bottleneck breakdown (Roadmap A0).

Two uses:

  # 1. Summarize an existing run's profile.csv (+ optional gpu_util.csv):
  python Scripts/tools/bench_run.py --profile-csv <output>/profile.csv [--gpu-csv gpu_util.csv]

  # 2. Launch a short profiling run then summarize it (host with the MD stack):
  python Scripts/tools/bench_run.py --config ./Systems/chignolin/config.yml --run

The summary ranks where per-cycle wall-clock goes -- GPU propagation vs the CPU
resampling analysis (projection / distance-matrix / changepoint+intensity) vs
warp vs work-mapper overhead -- and reports mean GPU utilization. That ranking
is what orders the optimization work (A1-A5). Reading/summarizing is pure Python
(no MD deps) so it is unit-testable; --run just shells out to run_cowera.py.
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys

# Buckets to rank as a fraction of cycle wall time.
BUCKETS = [
    ("runner_time", "GPU propagation (segment phase)"),
    ("bc_time", "boundary/warp"),
    ("resampling_time", "resampling (total)"),
    ("mapper_overhead", "work-mapper overhead (fork/serialize)"),
]
SUB_BUCKETS = [
    ("images_time", "  - projection of current state"),
    ("distmat_time", "  - pairwise distance matrix O(N^2)"),
    ("intensity_time", "  - CV history + changepoint + intensity"),
]


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if v not in (None, ""):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def summarize_profile(rows):
    """Return a dict of {label: (mean_seconds, pct_of_cycle_wall)} plus meta."""
    if not rows:
        return {"cycles": 0}
    walls = _floats(rows, "cycle_wall")
    mean_wall = statistics.mean(walls) if walls else 0.0
    result = {
        "cycles": len(rows),
        "mean_cycle_wall_s": mean_wall,
        "cycles_per_s": (1.0 / mean_wall) if mean_wall > 0 else float("inf"),
        "buckets": [],
        "sub_buckets": [],
    }
    for key, label in BUCKETS:
        vals = _floats(rows, key)
        m = statistics.mean(vals) if vals else 0.0
        pct = (100.0 * m / mean_wall) if mean_wall > 0 else 0.0
        result["buckets"].append((label, m, pct))
    for key, label in SUB_BUCKETS:
        vals = _floats(rows, key)
        m = statistics.mean(vals) if vals else 0.0
        pct = (100.0 * m / mean_wall) if mean_wall > 0 else 0.0
        result["sub_buckets"].append((label, m, pct))
    # rank the top-level buckets by time spent
    result["buckets"].sort(key=lambda t: t[1], reverse=True)
    return result


def summarize_gpu(rows):
    utils = _floats(rows, "utilization_pct")
    if not utils:
        return None
    return {
        "samples": len(utils),
        "mean_util_pct": statistics.mean(utils),
        "median_util_pct": statistics.median(utils),
        "idle_frac": sum(1 for u in utils if u < 5) / len(utils),
    }


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _print_report(prof, gpu):
    print("=" * 64)
    print("CoWERA profiling summary")
    print("=" * 64)
    if prof.get("cycles", 0) == 0:
        print("no profile rows found.")
        return
    print(f"cycles                : {prof['cycles']}")
    print(f"mean cycle wall       : {prof['mean_cycle_wall_s']*1e3:.2f} ms "
          f"({prof['cycles_per_s']:.2f} cycles/s)")
    print("-" * 64)
    print("where the wall-clock goes (mean per cycle, ranked):")
    for label, m, pct in prof["buckets"]:
        print(f"  {label:<40} {m*1e3:8.2f} ms  {pct:5.1f}%")
    print("  resampling breakdown:")
    for label, m, pct in prof["sub_buckets"]:
        print(f"  {label:<40} {m*1e3:8.2f} ms  {pct:5.1f}%")
    if gpu:
        print("-" * 64)
        print(f"GPU utilization       : mean {gpu['mean_util_pct']:.1f}%  "
              f"median {gpu['median_util_pct']:.1f}%  "
              f"idle(<5%) {gpu['idle_frac']*100:.1f}% of samples")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Summarize a CoWERA profiling run.")
    ap.add_argument("--profile-csv", default="profile.csv")
    ap.add_argument("--gpu-csv", default=None)
    ap.add_argument("--config", default=None, help="run this config first (with --run)")
    ap.add_argument("--run", action="store_true", help="launch run_cowera.py before summarizing")
    args = ap.parse_args()

    if args.run:
        if not args.config:
            ap.error("--run requires --config")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        script = os.path.join(repo_root, "Scripts", "run_cowera.py")
        print(f"launching: {sys.executable} {script} --config {args.config}")
        subprocess.run([sys.executable, script, "--config", args.config],
                       cwd=repo_root, check=True)

    if not os.path.isfile(args.profile_csv):
        ap.error(f"profile CSV not found: {args.profile_csv}")
    prof = summarize_profile(_read_csv(args.profile_csv))
    gpu = summarize_gpu(_read_csv(args.gpu_csv)) if args.gpu_csv and os.path.isfile(args.gpu_csv) else None
    _print_report(prof, gpu)


if __name__ == "__main__":
    main()
