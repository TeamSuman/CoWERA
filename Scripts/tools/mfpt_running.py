"""Running MFPT / rate from a CoWERA wepy.results.h5 (Hill relation, paper Eq. 2/7).

Reads only the warp records -- `/runs/*/warping/_cycle_idxs` and `.../weight` --
so it is cheap and does not need the MD stack (pure h5py + numpy). Prints the
running rate/MFPT vs aggregated simulation time and a simple plateau metric, so a
production run can be watched for convergence.

    python Scripts/tools/mfpt_running.py --h5 <results.h5> --n-walkers 64 --n-steps 1000

MFPT = T_agg / Sum(w_target),   T_agg = cycle * dT * N   (all N walkers advance dT/cycle)
dT = n_steps * dt,  dt = 0.002 ps.   rate k = 1 / MFPT.

Run it on a COPY of the h5 while a job is live (HDF5 has no reader/writer lock
without SWMR); `cp results.h5 /tmp/x.h5 && mfpt_running.py --h5 /tmp/x.h5 ...`.
Handles restart continuation runs (/runs/0, /runs/1, ...) by concatenating with a
cumulative cycle offset.
"""
import argparse
import sys

import numpy as np


def _flat(dset):
    a = np.asarray(dset[()])
    if a.size == 0:
        return np.array([])
    return a.reshape(a.shape[0], -1)[:, 0] if a.ndim > 1 else a.ravel()


def _run_cycle_count(run_grp):
    """Cycles in a run, for offsetting a continuation run's warp cycle indices."""
    for path in ("resampling/_cycle_idxs", "progress/_cycle_idxs", "warping/_cycle_idxs"):
        try:
            c = _flat(run_grp[path])
            if len(c):
                return int(c.max()) + 1
        except Exception:
            pass
    return 0


def load_warps(h5_path):
    import h5py
    cycles, weights = [], []
    offset = 0
    with h5py.File(h5_path, "r") as f:
        run_names = sorted(f["runs"].keys(), key=lambda s: int(s))
        for rn in run_names:
            run = f["runs"][rn]
            if "warping" not in run:
                offset += _run_cycle_count(run)
                continue
            c = _flat(run["warping"]["_cycle_idxs"]).astype(float)
            w = _flat(run["warping"]["weight"]).astype(float)
            n = min(len(c), len(w))
            cycles.append(c[:n] + offset)
            weights.append(w[:n])
            offset += _run_cycle_count(run)
    if not cycles:
        return np.array([]), np.array([]), run_names
    return np.concatenate(cycles), np.concatenate(weights), run_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--n-walkers", type=int, required=True)
    ap.add_argument("--n-steps", type=int, required=True, help="MD steps per cycle")
    ap.add_argument("--dt-ps", type=float, default=0.002, help="integration timestep (ps)")
    ap.add_argument("--checkpoints", type=int, default=10, help="rows in the running table")
    args = ap.parse_args()

    dT_ps = args.n_steps * args.dt_ps
    N = args.n_walkers

    try:
        cyc, w, runs = load_warps(args.h5)
    except Exception as exc:
        print(f"[mfpt] could not read {args.h5}: {exc}", file=sys.stderr)
        sys.exit(1)

    if len(cyc) == 0:
        print(f"[mfpt] runs={runs}: no warp events yet -- no rate estimate.")
        return

    order = np.argsort(cyc, kind="stable")
    cyc, w = cyc[order], w[order]
    cumw = np.cumsum(w)
    Tagg_ps = cyc * dT_ps * N
    mfpt_us = (Tagg_ps / cumw) * 1e-6
    rate_1e7 = (1e12 / (Tagg_ps / cumw)) / 1e7

    mol_ns_last = cyc[-1] * dT_ps * 1e-3          # per-walker molecular time
    print(f"[mfpt] runs={runs}  warp_events={len(cyc)}  N={N}  dT={dT_ps:g} ps")
    print(f"[mfpt] last warp @ cycle {int(cyc[-1])}  "
          f"molecular time {mol_ns_last:.2f} ns/walker  "
          f"aggregate {mol_ns_last*N*1e-3:.3f} us")
    print(f"[mfpt] FINAL  rate = {rate_1e7[-1]:.4f} x1e7 s^-1   MFPT = {mfpt_us[-1]:.4f} us")

    # running table at evenly spaced event checkpoints
    idx = np.unique(np.linspace(0, len(cyc) - 1, args.checkpoints).astype(int))
    print("   agg_time(us)   n_warps   rate(1e7/s)   MFPT(us)")
    for i in idx:
        agg_us = cyc[i] * dT_ps * N * 1e-6
        print(f"   {agg_us:11.4f}   {i+1:7d}   {rate_1e7[i]:11.4f}   {mfpt_us[i]:9.4f}")

    # plateau metric: relative spread of the rate over the last third of events
    tail = rate_1e7[max(0, (2 * len(rate_1e7)) // 3):]
    if len(tail) >= 3:
        spread = (tail.max() - tail.min()) / np.mean(tail)
        drift = abs(tail[-1] - tail[0]) / abs(tail[0]) if tail[0] else float("nan")
        verdict = "PLATEAU (looks converged)" if spread < 0.10 and drift < 0.10 else "still drifting"
        print(f"[mfpt] last-third rate spread {spread*100:.1f}% , end-to-end drift "
              f"{drift*100:.1f}%  ->  {verdict}")


if __name__ == "__main__":
    main()
