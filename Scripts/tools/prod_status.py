"""One-shot production-run status: profiling flatness + folding progress + rate.

Reads profile.csv (per-cycle timing) and a COPY of wepy.results.h5 (warp records +
progress distances) and prints a compact status plus a machine-readable VERDICT
line for a monitor loop. Pure h5py+numpy (no MD stack).

    python Scripts/tools/prod_status.py --dir <run_output_dir> --n-walkers 16 --n-steps 1000 \
        --h5 /tmp/copy.h5   # optional: read warps/progress from this copy instead of dir/wepy.results.h5
"""
import argparse
import csv
import os
import sys

import numpy as np


def read_profile(path, tail=25):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): v for k, v in r.items()})
    if not rows:
        return None
    def col(rows, k):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(k, "") or "nan"))
            except ValueError:
                out.append(float("nan"))
        return np.array(out)
    n = len(rows)
    last = rows[-tail:]
    def m(k):
        return float(np.nanmean(col(last, k)))
    return dict(n_cycles=n, true_wall=m("true_cycle_wall"), runner=m("runner_time"),
               intensity=m("intensity_time"), resamp=m("resampling_time"),
               report=m("reporting_time"), cv_update=m("cv_update_time"))


def read_h5(path, n_walkers, n_steps, dt_ps=0.002):
    import h5py
    out = dict(warps=0, rate_1e7=None, mfpt_us=None, plateau=False,
               min_dist_global=None, min_dist_recent=None)
    dT = n_steps * dt_ps
    with h5py.File(path, "r") as f:
        # warps
        cyc, w = [], []
        for rn in sorted(f["runs"].keys(), key=lambda s: int(s)):
            run = f["runs"][rn]
            if "warping" in run and run["warping"]["weight"].shape[0] > 0:
                c = np.asarray(run["warping"]["_cycle_idxs"][()]).ravel().astype(float)
                wt = np.array([x[0] if np.ndim(x) else x for x in run["warping"]["weight"][()]], float)
                cyc.append(c); w.append(wt)
        if cyc:
            cyc = np.concatenate(cyc); w = np.concatenate(w)
            o = np.argsort(cyc); cyc, w = cyc[o], w[o]
            cumw = np.cumsum(w)
            rate = (1e12 / (cyc * dT * n_walkers / cumw)) / 1e7
            out["warps"] = int(len(cyc))
            out["rate_1e7"] = float(rate[-1]); out["mfpt_us"] = float((cyc[-1]*dT*n_walkers/cumw[-1])*1e-6)
            tail = rate[max(0, 2*len(rate)//3):]
            if len(tail) >= 4:
                spread = (tail.max()-tail.min())/np.mean(tail)
                out["plateau"] = bool(spread < 0.10)
        # progress distances (object array: per-cycle array of per-walker distances)
        try:
            d = f["runs/0/progress/distances_fromtarget"][()]
            per_cyc_min = np.array([np.min(np.asarray(x, float)) for x in d])
            out["min_dist_global"] = float(per_cyc_min.min())
            out["min_dist_recent"] = float(per_cyc_min[-200:].min())
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n-walkers", type=int, required=True)
    ap.add_argument("--n-steps", type=int, required=True)
    ap.add_argument("--h5", default=None)
    args = ap.parse_args()

    prof = read_profile(os.path.join(args.dir, "profile.csv"))
    h5path = args.h5 or os.path.join(args.dir, "wepy.results.h5")
    try:
        h = read_h5(h5path, args.n_walkers, args.n_steps)
    except Exception as exc:
        h = None
        print(f"[status] h5 read failed: {exc}")

    cyc = prof["n_cycles"] if prof else 0
    if prof:
        print(f"[status] cycle={cyc}  true_wall={prof['true_wall']:.2f}s "
              f"(runner={prof['runner']:.2f} intensity={prof['intensity']:.3f} "
              f"resamp={prof['resamp']:.3f} report={prof['report']:.3f})")
    if h:
        md = f"{h['min_dist_recent']:.4f}" if h['min_dist_recent'] is not None else "n/a"
        if h["warps"]:
            print(f"[status] warps={h['warps']}  rate={h['rate_1e7']:.4f} x1e7/s  "
                  f"MFPT={h['mfpt_us']:.4f} us  plateau={h['plateau']}  min_dist_recent={md}")
        else:
            print(f"[status] warps=0  min_dist_recent={md} (warp threshold varies; folding not yet crossing)")

    # machine-readable verdict for the monitor loop
    if h and h["warps"] and h["plateau"] and cyc > 3500:
        print(f"VERDICT converged cycle={cyc} rate={h['rate_1e7']:.4f}")
    elif cyc > 3200 and (not h or h["warps"] == 0):
        print(f"VERDICT no_warps cycle={cyc}")
    else:
        print(f"VERDICT running cycle={cyc} warps={h['warps'] if h else '?'}")


if __name__ == "__main__":
    main()
