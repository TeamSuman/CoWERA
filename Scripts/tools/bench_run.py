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
    ("reporting_time", "reporting I/O (HDF5/pkl/dashboard/DCD)"),
    ("mapper_overhead", "work-mapper overhead (fork/serialize)"),
]
SUB_BUCKETS = [
    ("images_time", "  - projection of current state"),
    ("distmat_time", "  - pairwise distance matrix O(N^2)"),
    ("cv_update_time", "  - CV-history update: FULL DCD reload (A2 target)"),
    ("intensity_time", "  - changepoint (P3b) + intensity/binning"),
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


def _mean(rows, key):
    vals = _floats(rows, key)
    return statistics.mean(vals) if vals else 0.0


def summarize_profile(rows):
    """Return a dict of {label: (mean_seconds, pct_of_wall)} plus meta.

    Percentages are relative to the TRUE end-to-end cycle wall when the profile has
    it (`true_cycle_wall`, added by the P0' profiler fix); older CSVs without it
    fall back to the accounted `cycle_wall` (= runner+bc+resampling) so this stays
    backward compatible.
    """
    if not rows:
        return {"cycles": 0}
    mean_accounted = _mean(rows, "cycle_wall")          # runner+bc+resampling (legacy)
    mean_true = _mean(rows, "true_cycle_wall")           # real wall incl. reporting
    mean_reporting = _mean(rows, "reporting_time")
    mean_unacc = _mean(rows, "unaccounted")
    denom = mean_true if mean_true > 0 else mean_accounted   # % denominator
    nwalkers = [int(v) for v in _floats(rows, "n_walkers")]
    # Aggregate MD throughput = walker-steps sampled per wall-second. This -- not
    # GPU-util% -- is the efficiency metric that matters (for small systems/toy
    # potentials GPU util is meaningless: tiny kernels can't fill the device). It
    # answers "does adding walkers raise total sampling rate, or does overhead eat
    # it?" GPU-hours-to-converged-kinetics is the ultimate metric (computed at
    # convergence). Per cycle: n_walkers * n_segment_steps / true_cycle_wall.
    throughput = None
    ws_wall = []
    for r in rows:
        try:
            nw = float(r.get("n_walkers")); ns = float(r.get("n_segment_steps"))
            tw = float(r.get("true_cycle_wall") or r.get("cycle_wall") or 0)
            if tw > 0 and nw > 0 and ns > 0:
                ws_wall.append((nw * ns, tw))
        except (TypeError, ValueError):
            continue
    if ws_wall:
        throughput = sum(x for x, _ in ws_wall) / sum(t for _, t in ws_wall)
    result_throughput = throughput
    result = {
        "cycles": len(rows),
        "throughput_walker_steps_per_s": result_throughput,
        "mean_cycle_wall_s": mean_accounted,             # kept for back-compat callers
        "mean_true_wall_s": mean_true,
        "mean_reporting_s": mean_reporting,
        "mean_unaccounted_s": mean_unacc,
        "has_true_wall": mean_true > 0,
        # walker population: min/max/last confirm steady state (CoWERA fixes N, so
        # GPU-util-vs-N is a cross-run num_walkers scan, not within-run growth).
        "n_walkers_min": min(nwalkers) if nwalkers else None,
        "n_walkers_max": max(nwalkers) if nwalkers else None,
        "n_walkers_last": nwalkers[-1] if nwalkers else None,
        # fraction of true wall NOT in the accounted cycle_wall (the old blind spot)
        "blind_spot_frac": ((denom - mean_accounted) / denom) if denom > 0 else 0.0,
        "cycles_per_s": (1.0 / denom) if denom > 0 else float("inf"),
        "buckets": [],
        "sub_buckets": [],
    }
    for key, label in BUCKETS:
        m = _mean(rows, key)
        pct = (100.0 * m / denom) if denom > 0 else 0.0
        result["buckets"].append((label, m, pct))
    for key, label in SUB_BUCKETS:
        m = _mean(rows, key)
        pct = (100.0 * m / denom) if denom > 0 else 0.0
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
    if prof.get("n_walkers_max") is not None:
        nmin, nmax, nlast = prof["n_walkers_min"], prof["n_walkers_max"], prof["n_walkers_last"]
        steady = "steady" if nmin == nmax else f"varies {nmin}-{nmax}"
        print(f"walkers (min/max/last): {nmin} / {nmax} / {nlast}  ({steady})")
    thr = prof.get("throughput_walker_steps_per_s")
    if thr:
        print(f"THROUGHPUT            : {thr:,.0f} walker-steps/s  "
              f"<- the efficiency metric (GPU-util% only matters for large systems)")
    if prof.get("has_true_wall"):
        print(f"TRUE cycle wall       : {prof['mean_true_wall_s']*1e3:.2f} ms "
              f"({prof['cycles_per_s']:.2f} cycles/s)  <- trust this")
        print(f"  accounted (r+bc+rs) : {prof['mean_cycle_wall_s']*1e3:.2f} ms")
        print(f"  reporting I/O       : {prof['mean_reporting_s']*1e3:.2f} ms")
        print(f"  unaccounted residual: {prof['mean_unaccounted_s']*1e3:.2f} ms")
        print(f"  >> profiler blind spot (true not in accounted): "
              f"{prof['blind_spot_frac']*100:.1f}%")
    else:
        print(f"mean cycle wall (acct): {prof['mean_cycle_wall_s']*1e3:.2f} ms "
              f"({prof['cycles_per_s']:.2f} cycles/s)  [no true_cycle_wall in CSV]")
    print("-" * 64)
    print("where the wall-clock goes (mean per cycle, ranked, % of true wall):")
    for label, m, pct in prof["buckets"]:
        print(f"  {label:<40} {m*1e3:8.2f} ms  {pct:5.1f}%")
    print("  resampling breakdown:")
    for label, m, pct in prof["sub_buckets"]:
        print(f"  {label:<40} {m*1e3:8.2f} ms  {pct:5.1f}%")
    if gpu:
        print("-" * 64)
        print(f"GPU util (DIAGNOSTIC)  : mean {gpu['mean_util_pct']:.1f}%  "
              f"median {gpu['median_util_pct']:.1f}%  "
              f"idle(<5%) {gpu['idle_frac']*100:.1f}% of samples")
        print("  (only a meaningful lever for LARGE, GPU-bound systems; low util on a "
              "small\n   system/toy potential is expected, not a defect -- optimize "
              "throughput instead)")
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
