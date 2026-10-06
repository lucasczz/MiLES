import json
from pathlib import Path

import numpy as np
from torch.optim import Adam

from run import get_config_grid, run_configs, setup_device_config
from src.models import BiTULER


REPORTS = Path(__file__).parent.parent / "reports"
TRAJ = REPORTS / "level_weights_800.jsonl"
SCHED_OUT = REPORTS / "level_lr_schedule.jsonl"
FIXED_OUT = REPORTS / "level_fixed_gain.jsonl"
SEEDS = [0, 1, 2, 3, 4]

SHARED = dict(
    model_cls=BiTULER,
    dataset="foursquare_NYC",
    n_users=800,
    loc_levels=4,
    time_levels=1,
    optimizer_cls=Adam,
    discretization_rows=200,
    discretization_shape="hex",
    aggregation_mode="grow",
    grow_factor=2,
    lr=2e-4,
    n_hidden=1024,
    n_layers=1,
    loc_embedding_factor=1,
    time_embedding_factor=1 / 16,
    subsample=None,
)


def _seeds_in(path, require_weights=False):
    if not path.exists():
        return set()
    seen = set()
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        if require_weights and row.get("embedding_weight_0") is None:
            continue
        seen.add(row["seed"])
    return seen


def main():
    devices, num_workers, _ = setup_device_config()

    # Phase 1: per-seed 800-user full-MiLES weight trajectories.
    missing = [s for s in SEEDS if s not in _seeds_in(TRAJ, require_weights=True)]
    if missing:
        print(f"[phase1] generating trajectories for seeds {missing}", flush=True)
        configs = get_config_grid(
            embedding_type="lookup_weighted_concat",
            embedding_weight_factor=2,
            seed=missing,
            log_path=TRAJ,
            track_embeddings=True,
            **SHARED,
        )
        run_configs(configs, devices=devices, num_workers=num_workers, path=TRAJ)
    else:
        print("[phase1] trajectories present, skipping", flush=True)

    # Phase 2: load per-seed trajectories (raw; run() resamples to the step count).
    by_seed = {}
    for line in open(TRAJ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("embedding_weight_0") is None or row["seed"] in by_seed:
            continue
        by_seed[row["seed"]] = np.stack(
            [row[f"embedding_weight_{i}"] for i in range(4)], axis=1
        )
    print(
        f"[phase2] trajectories for seeds {sorted(by_seed)} "
        f"shapes {[by_seed[s].shape for s in sorted(by_seed)]}",
        flush=True,
    )

    # Phase 3a: -WL with a per-step per-level LR equal to base_lr * w_l(t).
    todo = [s for s in SEEDS if s in by_seed and s not in _seeds_in(SCHED_OUT)]
    if todo:
        print(f"[phase3a] lr-schedule experiment for seeds {todo}", flush=True)
        configs = [
            dict(
                embedding_type="lookup_concat",
                embedding_weight_factor=2,
                seed=s,
                log_path=SCHED_OUT,
                loc_lr_schedule=by_seed[s],
                **SHARED,
            )
            for s in todo
        ]
        run_configs(configs, devices=devices, num_workers=num_workers, path=SCHED_OUT)
    else:
        print("[phase3a] complete, skipping", flush=True)

    # Phase 3b: -WL with a fixed (non-learnable) per-level gain = per-seed mean w_l.
    todo = [s for s in SEEDS if s in by_seed and s not in _seeds_in(FIXED_OUT)]
    if todo:
        print(f"[phase3b] fixed-gain experiment for seeds {todo}", flush=True)
        configs = [
            dict(
                embedding_type="lookup_weighted_concat",
                embedding_weight_factor=2,
                seed=s,
                log_path=FIXED_OUT,
                fixed_loc_weights=by_seed[s].mean(axis=0).tolist(),
                **SHARED,
            )
            for s in todo
        ]
        run_configs(configs, devices=devices, num_workers=num_workers, path=FIXED_OUT)
    else:
        print("[phase3b] complete, skipping", flush=True)

    print("[done] all phases complete", flush=True)


if __name__ == "__main__":
    main()
