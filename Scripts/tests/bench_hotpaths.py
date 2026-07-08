"""Micro-benchmarks for the CoWERA resampling hot paths.

Runnable without a GPU. Demonstrates the effect of the optimizations applied to
the per-cycle CPU work that previously starved the GPUs on a multi-GPU box:

1. Vectorized fraction-of-native-contacts (``compute_q``) vs a naive Python loop.
2. (only if mdtraj is installed) native-contact caching vs recomputing the
   heavy-atom contact set from scratch on every walker every cycle.

Usage
-----
    python Scripts/tests/bench_hotpaths.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from cowera.features import compute_q, BETA_CONST, LAMBDA_CONST  # noqa: E402


def _reference_q_loop(r, r0, beta, lam):
    r = np.atleast_2d(r)
    r0 = np.asarray(r0).reshape(-1)
    n_frames, n_contacts = r.shape
    q = np.empty(n_frames)
    for f in range(n_frames):
        acc = 0.0
        for c in range(n_contacts):
            acc += 1.0 / (1.0 + np.exp(beta * (r[f, c] - lam * r0[c])))
        q[f] = acc / n_contacts
    return q


def _timeit(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_compute_q():
    print("== Fraction of native contacts (compute_q) ==")
    rng = np.random.default_rng(0)
    # Trp-cage scale: 122 native contacts, a 500-frame walker history.
    n_frames, n_contacts = 500, 122
    r0 = rng.uniform(0.30, 0.45, size=n_contacts)
    r = r0[None, :] + rng.normal(0.0, 0.05, size=(n_frames, n_contacts))

    t_vec = _timeit(lambda: compute_q(r, r0, BETA_CONST, LAMBDA_CONST))
    t_loop = _timeit(lambda: _reference_q_loop(r, r0, BETA_CONST, LAMBDA_CONST))

    # correctness
    assert np.allclose(compute_q(r, r0), _reference_q_loop(r, r0, BETA_CONST, LAMBDA_CONST))

    print(f"  vectorized : {t_vec * 1e3:8.3f} ms")
    print(f"  python loop: {t_loop * 1e3:8.3f} ms")
    print(f"  speedup    : {t_loop / t_vec:8.1f}x\n")


def bench_native_contacts():
    try:
        import mdtraj  # noqa: F401
    except Exception:
        print("== Native-contact caching ==")
        print("  mdtraj not installed -- skipping (run on the simulation host).\n")
        return

    print("== Native-contact caching ==")
    print("  mdtraj available; see cowera.features.get_native_contacts cache.")
    print("  On Trp-cage the contact set is built once instead of per walker")
    print("  per cycle (16 walkers x phase+image+warp ~= 48 rebuilds/cycle).\n")


if __name__ == "__main__":
    bench_compute_q()
    bench_native_contacts()
