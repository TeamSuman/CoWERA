"""Aggregate independent-replica CoWERA runs into a rate + confidence interval.

Each replica's rate comes from the Hill relation (k = Sum(w_target)/(cycle*dT*N),
paper Eq. 2/7); across replicas we report the mean, standard deviation, standard
error, and a 95% CI (mean +/- t*SE) -- the paper's block-averaged-over-replicas
estimate. Pure h5py + numpy.

    python Scripts/tools/replica_stats.py --n-walkers 16 --n-steps 1000 \
        --h5 run0/wepy.results.h5 run1/wepy.results.h5 ...
"""
import argparse
import os

import numpy as np


# 95% two-sided t critical values for small samples (df = n-1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086}


def _t95(df):
    if df <= 0:
        return float("nan")
    keys = sorted(_T95)
    for k in keys:
        if df <= k:
            return _T95[k]
    return 1.96


def replica_rate(h5_path, n_walkers, n_steps, dt_ps=0.002):
    """Final Hill-relation rate (x1e7 s^-1) and warp count for one replica."""
    import h5py
    dT = n_steps * dt_ps
    cyc, w = [], []
    with h5py.File(h5_path, "r") as f:
        for rn in sorted(f["runs"].keys(), key=lambda s: int(s)):
            run = f["runs"][rn]
            if "warping" in run and run["warping"]["weight"].shape[0] > 0:
                cyc.append(np.asarray(run["warping"]["_cycle_idxs"][()]).ravel().astype(float))
                w.append(np.array([x[0] if np.ndim(x) else x
                                   for x in run["warping"]["weight"][()]], float))
    if not cyc:
        return None, 0
    cyc = np.concatenate(cyc); w = np.concatenate(w)
    o = np.argsort(cyc); cyc, w = cyc[o], w[o]
    rate = (np.cumsum(w)[-1] / (cyc[-1] * dT * n_walkers)) * 1e12   # s^-1
    return rate / 1e7, int(len(cyc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", nargs="+", required=True, help="replica wepy.results.h5 files")
    ap.add_argument("--n-walkers", type=int, required=True)
    ap.add_argument("--n-steps", type=int, required=True)
    a = ap.parse_args()

    rates, labels = [], []
    print("per-replica folding rate (Hill relation):")
    for p in a.h5:
        try:
            r, nw = replica_rate(p, a.n_walkers, a.n_steps)
        except Exception as exc:
            print(f"  {p}: read error ({exc})"); continue
        tag = os.path.basename(os.path.dirname(p))
        if r is None:
            print(f"  {tag:28s}  no warps yet"); continue
        rates.append(r); labels.append(tag)
        print(f"  {tag:28s}  rate = {r:.4f} x1e7 s^-1   (warps={nw})")

    if len(rates) < 1:
        print("no replicas with warps."); return
    rates = np.array(rates)
    n = len(rates); mean = rates.mean()
    sd = rates.std(ddof=1) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    ci = _t95(n - 1) * se if n > 1 else float("nan")
    print("-" * 56)
    print(f"replicas          : {n}")
    print(f"mean K_fold       : {mean:.4f} x1e7 s^-1")
    if n > 1:
        print(f"std / SE          : {sd:.4f} / {se:.4f}")
        print(f"95% CI (t)        : {mean:.4f} ({mean-ci:.4f}, {mean+ci:.4f}) x1e7 s^-1")
    print(f"MFPT (from mean)  : {1e-7/(mean*1e7)*1e6:.4f} us")
    print("paper (CoWERA)    : 0.96 (0.74, 1.18) x1e7 s^-1")


if __name__ == "__main__":
    main()
