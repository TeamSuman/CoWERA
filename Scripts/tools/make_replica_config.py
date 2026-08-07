"""Generate a per-replica CoWERA config from a base config.

Independent replicas differ only by `run` (=> separate output dir) and `seed`
(=> independent resampler RNG + OpenMM dynamics), so a multi-GPU launcher can run
one per GPU and the results block-average into a rate + confidence interval.
Forces profiling on and the live dashboard off (O(T) reporting) for production.

    python Scripts/tools/make_replica_config.py --base <cfg.yml> --run rep_0 --seed 1000 --out <out.yml>
"""
import argparse

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.base) as f:
        cfg = yaml.safe_load(f)

    cfg["run"] = a.run
    cfg["seed"] = a.seed
    cfg["profile"] = True
    cfg["dashboard"] = False          # avoid O(T) dashboard reporting on long runs
    # each replica writes its own trajectories under a run-labelled scratch subdir,
    # so a shared scratch_dir base does not collide across replicas.

    with open(a.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {a.out}  (run={a.run} seed={a.seed})")


if __name__ == "__main__":
    main()
