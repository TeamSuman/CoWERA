"""Validate P3b (changepoint_window): does bounding the changepoint history to the
most recent W frames REPRODUCE the unbounded (paper-faithful) result, and how much
does it speed up?

Pure numpy + ruptures (no MD / GPU) -- run on any host with ruptures. Only the
`phase` output of `phase_from_projection` depends on W (weight and the last-n_d
bins keep the trajectory tail, so they are windowing-invariant). We check, across
several synthetic CV series, the smallest W whose phase matches the unbounded phase
exactly, and the per-call time. Conclusion drives the safe default window.

    python Scripts/tools/validate_changepoint_window.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from cowera.metric import Calculate_Distances


def synth_cv(T, seed):
    """A piecewise-coherent CV series (drifting segments + noise) in [0, 1] --
    mimics a folding progress coordinate with a few relevant-history changepoints."""
    rng = np.random.default_rng(seed)
    out = []
    val = 0.3
    while sum(len(s) for s in out) < T:
        L = int(rng.integers(30, 130))
        drift = rng.uniform(-0.003, 0.003)
        seg = val + np.cumsum(rng.normal(drift, 0.012, L))
        out.append(seg)
        val = float(seg[-1])
    x = np.concatenate(out)[:T]
    return np.clip(x, 0.0, 1.0)


def main():
    cd = Calculate_Distances(feat="best_hummer_q", increment=1)
    n_d, n_bins = 5, 122
    print(f"{'T':>6} {'W':>7} {'phase==unbounded':>18} {'phaseΔ':>10} {'ms':>8}")
    print("-" * 56)
    worst_ok_W = {}   # smallest W that still matched, per T (across seeds)
    for T in (200, 500, 1000, 2000):
        for seed in (1, 2, 3):
            proj = synth_cv(T, seed)
            drange = np.array([proj.min(), proj.max()])
            cd.changepoint_window = None
            t0 = time.perf_counter()
            ref = cd.phase_from_projection(proj, drange, n_d, n_bins)
            t_ref = (time.perf_counter() - t0) * 1e3
            ref_phase = np.asarray(ref[1], dtype=float)
            if seed == 1:
                print(f"{T:6d} {'None':>7} {'(reference)':>18} {'--':>10} {t_ref:8.1f}")
            for W in (T, T // 2, 300, 200, 100, 50):
                cd.changepoint_window = W
                t0 = time.perf_counter()
                out = cd.phase_from_projection(proj, drange, n_d, n_bins)
                ms = (time.perf_counter() - t0) * 1e3
                d = float(np.max(np.abs(np.asarray(out[1], dtype=float) - ref_phase)))
                ok = d < 1e-9
                if ok:
                    worst_ok_W[T] = min(worst_ok_W.get(T, W), W)
                if seed == 1:
                    print(f"{T:6d} {W:7d} {str(ok):>18} {d:10.2e} {ms:8.1f}")
    print("-" * 56)
    print("smallest W that reproduced the unbounded phase (all seeds), per T:")
    for T, W in sorted(worst_ok_W.items()):
        print(f"  T={T}: W>={W}")
    print("\n=> pick a default changepoint_window >= max relevant-history length seen;")
    print("   phase must match EXACTLY (Δ=0) to preserve accuracy/Tables II-III.")


if __name__ == "__main__":
    main()
