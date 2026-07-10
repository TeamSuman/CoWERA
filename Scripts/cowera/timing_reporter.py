"""Per-cycle timing reporter for CoWERA performance profiling (Roadmap A0).

A ``wepy`` reporter that consumes the timing fields the simulation manager already
computes each cycle (``cowera/sim_manager.py::_run_cycle`` puts them in the report
dict) plus the fine-grained resampling sub-timings stashed on the resampler
(``CoWERAResampler._last_subtimings``), and appends one row per cycle to a CSV.

This localizes where wall-clock goes -- GPU propagation vs the per-cycle CPU
resampling analysis (projection / distance matrix / changepoint+intensity) vs
warp vs work-mapper overhead (process fork + state serialization) -- which is the
data that orders the optimization work. It is pure bookkeeping (no MD deps) and is
only attached when ``profile: true``, so production runs are unaffected.
"""
import csv
import os.path as osp
import time

from wepy.reporter.reporter import Reporter

FIELDNAMES = [
    "cycle_idx",
    "wall_time",         # epoch time at cycle end -- join with gpu_util.csv to get GPU util vs N
    "n_walkers",         # walker population this cycle (track growth to steady state / max)
    "n_segment_steps",
    "true_cycle_wall",   # TRUE end-to-end cycle wall (incl. reporting/DCD) -- trust this
    "cycle_wall",        # accounted = runner + bc + resampling (undercount; legacy)
    "reporting_time",    # reporter I/O phase (HDF5/pkl/dashboard/DCD)
    "unaccounted",       # true_cycle_wall - (cycle_wall + reporting_time); should be ~0
    "runner_time",       # segment/propagation phase wall time
    "bc_time",           # boundary conditions / warp
    "resampling_time",   # resampler.resample total
    "worker_seg_sum",    # sum of per-walker segment times (aggregate GPU-seconds)
    "worker_seg_max",    # busiest single walker segment (critical path lower bound)
    "worker_busy_max",   # busiest worker's total (sum over its walkers)
    "mapper_overhead",   # runner_time - worker_busy_max (fork/serialize/queue)
    "images_time",       # resampler: per-walker projection of current state
    "distmat_time",      # resampler: O(N^2) pairwise distance matrix
    "cv_update_time",    # resampler: CV-history update incl. FULL DCD reload (A2 target)
    "intensity_time",    # resampler: changepoint (P3b) + intensity/binning
]


def _worker_seg_stats(worker_segment_times):
    """Return (sum_all, max_single, max_worker_total) from the manager's dict of
    {worker_idx: [segment_times...]}. Robust to empty/missing values."""
    if not worker_segment_times:
        return 0.0, 0.0, 0.0
    all_times = []
    worker_totals = []
    for times in worker_segment_times.values():
        times = list(times) if times else []
        all_times.extend(times)
        worker_totals.append(sum(times))
    if not all_times:
        return 0.0, 0.0, 0.0
    return sum(all_times), max(all_times), max(worker_totals)


class TimingReporter(Reporter):
    """Writes a per-cycle wall-clock breakdown to ``<save_path>``.

    Marked ``consumes_cycle_timing`` so the sim manager runs it LAST and feeds it
    the reporting-phase time + the true end-to-end cycle wall (which are only known
    after every other reporter has run).
    """

    consumes_cycle_timing = True

    def __init__(self, save_path="profile.csv"):
        self.save_path = save_path
        self.resampler = None
        self._fh = None
        self._writer = None

    def init(self, resampler=None, **kwargs):
        self.resampler = resampler
        # append if the file already exists (e.g. on --restart) but only write the
        # header for a fresh file.
        write_header = not osp.exists(self.save_path)
        self._fh = open(self.save_path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDNAMES)
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def report(self, cycle_idx=None, **kwargs):
        runner_time = float(kwargs.get("cycle_runner_time", 0.0) or 0.0)
        bc_time = float(kwargs.get("cycle_bc_time", 0.0) or 0.0)
        resampling_time = float(kwargs.get("cycle_resampling_time", 0.0) or 0.0)
        reporting_time = float(kwargs.get("cycle_reporting_time", 0.0) or 0.0)
        true_wall = float(kwargs.get("cycle_true_wall", 0.0) or 0.0)

        seg_sum, seg_max, worker_busy_max = _worker_seg_stats(
            kwargs.get("worker_segment_times"))

        sub = getattr(self.resampler, "_last_subtimings", {}) or {}

        # walker population this cycle (CoWERA conserves N via balanced clone/merge,
        # but track it to confirm steady state and catch warp/recycle effects).
        walkers = (kwargs.get("resampled_walkers") or kwargs.get("new_walkers") or [])
        n_walkers = len(walkers)

        accounted = runner_time + bc_time + resampling_time
        row = {
            "cycle_idx": cycle_idx,
            "wall_time": time.time(),          # cycle-end epoch; aligns with gpu_util.csv
            "n_walkers": n_walkers,
            "n_segment_steps": kwargs.get("n_segment_steps"),
            "true_cycle_wall": true_wall,
            "cycle_wall": accounted,
            "reporting_time": reporting_time,
            # residual the two timers don't explain (loop overhead, monitor, etc.);
            # should be small once reporting is captured.
            "unaccounted": max(0.0, true_wall - accounted - reporting_time),
            "runner_time": runner_time,
            "bc_time": bc_time,
            "resampling_time": resampling_time,
            "worker_seg_sum": seg_sum,
            "worker_seg_max": seg_max,
            "worker_busy_max": worker_busy_max,
            "mapper_overhead": max(0.0, runner_time - worker_busy_max),
            "images_time": sub.get("images_time"),
            "distmat_time": sub.get("distmat_time"),
            "cv_update_time": sub.get("cv_update_time"),
            "intensity_time": sub.get("intensity_time"),
        }
        self._writer.writerow(row)
        if self._fh is not None:
            self._fh.flush()

    def cleanup(self, **kwargs):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
