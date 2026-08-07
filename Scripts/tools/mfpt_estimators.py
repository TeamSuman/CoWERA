"""Systematic comparison of MFPT/rate estimators for CoWERA weighted-ensemble runs.

Motivation
----------
The rate is estimated from the Hill relation, k = Sum(w_warp) / T_agg. How T_agg is
defined, and which portion of the run is used, materially changes both the BIAS and
the VARIANCE of the estimate. This tool implements several estimators and scores them
on a set of independent replicas, so the choice can be made on evidence rather than
convention.

Estimators
----------
E0  "last-warp"   k = Sum(w) / T_agg(cycle of LAST warp)
      The convention in the paper's Analysis/WE_analysis.ipynb (T_list is built from
      the warp cycle indices and the final value is taken at the last warp). It throws
      away all simulated time AFTER the last warp -- time that was paid for and
      produced zero flux -- so it is biased HIGH, and the bias depends on where the
      last warp happened to land (adding replica-to-replica variance).
E1  "full-time"   k = Sum(w) / T_agg(TOTAL cycles run)
      The unbiased use of all sampling: every unit of simulated time is in the
      denominator whether or not it produced a warp.
E2  "steady-state" k = Sum(w in [t0,T]) / (T - t0)
      The Hill relation assumes STEADY-STATE flux. The pre-steady-state transient
      (walkers still relaxing from the initial state) is not steady-state flux, so
      including it biases the estimate. Discard a burn-in fraction and use only the
      steady portion.
E3  "block"       mean of per-block fluxes over the steady region, with a within-run
      CI from the block spread. Gives an INTERNAL uncertainty (does a single run know
      how uncertain it is?), which we can check against the true inter-replica spread.

Diagnostics
-----------
ESS   Kish effective sample size of the warp weights, (Sum w)^2 / Sum(w^2). The flux is
      a WEIGHTED sum of rare events; if a few events carry most of the weight, the
      effective number of independent contributions is far below the raw warp count,
      and the estimator's variance is correspondingly huge. This is the key statistic
      for explaining replica scatter.
dead  fraction of the run occurring AFTER the last warp (the time E0 discards).
top1  largest single warp weight / total weight.

    python mfpt_estimators.py --h5 r0.h5 r1.h5 ... --n-walkers 16 --n-steps 1000 --n-cycles 5000
"""
import argparse
import os

import numpy as np

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086}


def t95(df):
    if df <= 0:
        return float("nan")
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def load_warps(path):
    import h5py
    with h5py.File(path, "r") as f:
        g = f["runs/0/warping"]
        if g["weight"].shape[0] == 0:
            return np.array([]), np.array([])
        c = np.asarray(g["_cycle_idxs"][()]).ravel().astype(float)
        w = np.array([x[0] if np.ndim(x) else x for x in g["weight"][()]], float)
    o = np.argsort(c, kind="stable")
    return c[o], w[o]


def n_cycles_run(path):
    """Cycles ACTUALLY completed by this run.

    Must be read per replica, never assumed: a replica that is cut short (walltime,
    node failure) has a smaller denominator, and using a nominal n_cycles for it
    silently deflates E1/E2 and inflates the reported 'dead' fraction. Returns None
    if it cannot be determined, so the caller can fall back to --n-cycles.
    """
    import h5py
    try:
        with h5py.File(path, "r") as f:
            if "runs/0/progress/distances_fromtarget" in f:
                return int(len(f["runs/0/progress/distances_fromtarget"]))
            if "runs/0/resampling/_cycle_idxs" in f:
                return int(np.max(f["runs/0/resampling/_cycle_idxs"][()])) + 1
    except Exception:
        pass
    return None


def estimators(cyc, w, n_cycles, n_walkers, dT_s, burn=0.5, nblocks=5):
    """Return dict of rate estimates (s^-1) + diagnostics. dT_s = per-cycle time (s)."""
    out = {}
    # aggregate time per cycle across all walkers
    def T(c):
        return c * dT_s * n_walkers

    T_tot = T(n_cycles)
    sw = w.sum()

    # E0: denominator = time of the LAST warp (paper/notebook convention)
    out["E0_lastwarp"] = sw / T(cyc[-1]) if len(cyc) else np.nan
    # E1: denominator = the FULL simulated time
    out["E1_fulltime"] = sw / T_tot
    # E2: steady-state -- drop a burn-in fraction of the RUN, use flux in [t0, T]
    c0 = burn * n_cycles
    m = cyc >= c0
    out["E2_steady"] = w[m].sum() / (T_tot - T(c0)) if (T_tot - T(c0)) > 0 else np.nan
    # E3: per-block flux over the steady region (also yields a within-run CI)
    edges = np.linspace(c0, n_cycles, nblocks + 1)
    bf = []
    for a, b in zip(edges[:-1], edges[1:]):
        mm = (cyc >= a) & (cyc < b)
        dt = T(b) - T(a)
        if dt > 0:
            bf.append(w[mm].sum() / dt)
    bf = np.array(bf)
    out["E3_block"] = bf.mean() if len(bf) else np.nan
    out["_E3_ci"] = (t95(len(bf) - 1) * bf.std(ddof=1) / np.sqrt(len(bf))) if len(bf) > 1 else np.nan

    # diagnostics
    out["_warps"] = len(w)
    out["_ESS"] = (sw ** 2 / np.sum(w ** 2)) if sw > 0 else 0.0
    out["_top1"] = (w.max() / sw) if sw > 0 else np.nan
    out["_dead"] = 1.0 - (cyc[-1] / n_cycles) if len(cyc) else 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", nargs="+", required=True)
    ap.add_argument("--n-walkers", type=int, required=True)
    ap.add_argument("--n-steps", type=int, required=True)
    ap.add_argument("--n-cycles", type=int, default=None,
                    help="fallback only; the completed cycle count is read PER REPLICA "
                         "from each file (a short replica must not use a nominal value)")
    ap.add_argument("--dt-ps", type=float, default=0.002)
    ap.add_argument("--burn", type=float, default=0.5, help="burn-in fraction for E2/E3")
    ap.add_argument("--nblocks", type=int, default=5)
    a = ap.parse_args()

    dT_s = a.n_steps * a.dt_ps * 1e-12          # per-cycle time, seconds
    keys = ["E0_lastwarp", "E1_fulltime", "E2_steady", "E3_block"]

    rows, labels = [], []
    print(f"{'replica':12s} {'cycles':>7s} {'warps':>5s} {'ESS':>6s} {'top1':>6s} {'dead':>6s}  "
          + "  ".join(f"{k.split('_')[0]:>10s}" for k in keys))
    for p in a.h5:
        cyc, w = load_warps(p)
        tag = os.path.basename(p).replace(".h5", "")
        nc = n_cycles_run(p) or a.n_cycles
        if nc is None:
            print(f"{tag:12s}   cannot determine cycles run; pass --n-cycles"); continue
        if len(cyc) == 0:
            print(f"{tag:12s} {nc:7d}   no warps"); continue
        e = estimators(cyc, w, nc, a.n_walkers, dT_s, a.burn, a.nblocks)
        rows.append(e); labels.append(tag)
        print(f"{tag:12s} {nc:7d} {e['_warps']:5d} {e['_ESS']:6.1f} {e['_top1']:6.2f} {e['_dead']:6.2f}  "
              + "  ".join(f"{e[k]/1e7:10.3f}" for k in keys))

    if not rows:
        return
    print("\n" + "=" * 78)
    print(f"ENSEMBLE over {len(rows)} replicas   (rate in 1e7 s^-1)")
    print(f"{'estimator':14s} {'mean':>8s} {'std':>8s} {'CV%':>7s} {'95% CI (t)':>22s}")
    for k in keys:
        v = np.array([r[k] for r in rows]) / 1e7
        v = v[np.isfinite(v)]
        n = len(v)
        m, sd = v.mean(), (v.std(ddof=1) if n > 1 else np.nan)
        se = sd / np.sqrt(n) if n > 1 else np.nan
        ci = t95(n - 1) * se if n > 1 else np.nan
        cv = 100 * sd / m if n > 1 and m else np.nan
        print(f"{k:14s} {m:8.3f} {sd:8.3f} {cv:7.1f} {'(%.3f, %.3f)' % (m-ci, m+ci):>22s}")

    # calibration: does a single run's INTERNAL CI (E3 blocks) match the true
    # inter-replica spread? If it is much narrower, single-run error bars lie.
    e3 = np.array([r["E3_block"] for r in rows]) / 1e7
    ci3 = np.array([r["_E3_ci"] for r in rows]) / 1e7
    ok = np.isfinite(e3) & np.isfinite(ci3)
    if ok.sum() > 1:
        wr, ir = float(np.mean(ci3[ok])), float(e3[ok].std(ddof=1))
        print(f"\nCALIBRATION: mean within-run 95% CI half-width = {wr:.3f}")
        print(f"             actual inter-replica std           = {ir:.3f}")
        if wr < 0.5 * ir:
            print("             within-run CI << replica spread => single-run error bars are OVERCONFIDENT")
        elif wr > 2 * ir:
            print("             within-run CI >> replica spread => single-run error bars are inflated "
                  "(too few events per block to estimate flux)")
        else:
            print("             within-run CI ~ replica spread => single-run error bars are reasonable")
    print("\npaper: CoWERA 0.96 (0.74,1.18) ; reference unbiased MD 0.71 (0.44,1.24)  x1e7 s^-1")


if __name__ == "__main__":
    main()
