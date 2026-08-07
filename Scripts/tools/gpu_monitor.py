"""Sample GPU utilization/memory to a CSV alongside a CoWERA run (Roadmap A0).

Run it in the background next to `run_cowera.py` on the GPU host:

    python Scripts/tools/gpu_monitor.py --out gpu_util.csv --interval 1 &
    MON=$!
    python Scripts/run_cowera.py --config ...
    kill $MON

It prefers NVML (`pynvml`) and falls back to parsing `nvidia-smi`. Each row is
(timestamp, gpu_index, utilization_pct, memory_used_mb). Correlate the
utilization trace with `<output>/profile.csv` to see GPU idling during the
per-cycle CPU resampling (the gap the persistent-worker and CPU/GPU-overlap work
is meant to close).
"""
import argparse
import csv
import subprocess
import sys
import time


def _sample_nvml(handles, pynvml):
    rows = []
    now = time.time()
    for idx, h in enumerate(handles):
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
        rows.append((now, idx, util, round(mem, 1)))
    return rows


def _sample_smi():
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        text=True)
    now = time.time()
    rows = []
    for line in out.strip().splitlines():
        idx, util, mem = (x.strip() for x in line.split(","))
        rows.append((now, int(idx), int(util), float(mem)))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Sample GPU utilization to CSV.")
    ap.add_argument("--out", default="gpu_util.csv")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds (default: until killed)")
    args = ap.parse_args()

    pynvml = None
    handles = None
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
    except Exception:
        pynvml = None  # fall back to nvidia-smi

    start = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "gpu_index", "utilization_pct", "memory_used_mb"])
        try:
            while True:
                try:
                    rows = _sample_nvml(handles, pynvml) if pynvml else _sample_smi()
                    w.writerows(rows)
                    f.flush()
                except Exception as exc:
                    print(f"gpu_monitor: sample failed ({exc})", file=sys.stderr)
                if args.duration is not None and (time.time() - start) >= args.duration:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
