"""Why does -WL not reach the same per-level forward gain on its own?

Tests the hypothesis that MiLES's learnable level weights w_l are a
well-conditioned, single-parameter handle on each level's forward scale, whereas
the -WL model can only reach the same scale by coherently moving the whole first
RNN input block W_l (d_l x 4*n_hidden parameters per level) -- a much
worse-conditioned target that lags within a single online pass.

We track, over the online pass, the identifiable per-level forward contribution
magnitude

    g_l(t) = w_l(t) * mean_row_norm(Z_l)(t) * mean_col_norm(W_l)(t)

for two conditions on Foursquare-NYC (800 users, BiTULER, seeds 0-4):

- MiLES (lookup_weighted_concat): w_l is learnable; expect it to differentiate
  the per-level g_l early.
- -WL   (lookup_concat): w_l == 1 throughout; the only route to a per-level
  scale is via W_l (and the embedding norms). Expect g_l to stay closer to its
  initialization ratios / move there more slowly, leaving the ~0.6 pt on the
  table.

w_l is read from the tracked weights (MiLES) or taken as 1 (-WL); the row norms
and W_l column norms come from the new log_emb_norms / log_input_weight_norms
tracking flags.

Outputs:
  reports/level_gain_miles.jsonl
  reports/level_gain_nowl.jsonl
"""

import json
from pathlib import Path

from torch.optim import Adam

from run import get_config_grid, run_configs, setup_device_config
from src.models import BiTULER


REPORTS = Path(__file__).parent.parent / "reports"
MILES_OUT = REPORTS / "level_gain_miles.jsonl"
NOWL_OUT = REPORTS / "level_gain_nowl.jsonl"
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
    embedding_weight_factor=2,
    log_emb_norms=True,
    log_input_weight_norms=True,
)


def _seeds_in(path):
    if not path.exists():
        return set()
    seen = set()
    for line in open(path):
        if line.strip():
            seen.add(json.loads(line)["seed"])
    return seen


def main():
    devices, num_workers, _ = setup_device_config()

    todo = [s for s in SEEDS if s not in _seeds_in(MILES_OUT)]
    if todo:
        print(f"[miles] seeds {todo}", flush=True)
        configs = get_config_grid(
            embedding_type="lookup_weighted_concat",
            track_embeddings=True,  # records w_l(t)
            seed=todo,
            log_path=MILES_OUT,
            **SHARED,
        )
        run_configs(configs, devices=devices, num_workers=num_workers, path=MILES_OUT)
    else:
        print("[miles] complete, skipping", flush=True)

    todo = [s for s in SEEDS if s not in _seeds_in(NOWL_OUT)]
    if todo:
        print(f"[nowl] seeds {todo}", flush=True)
        configs = get_config_grid(
            embedding_type="lookup_concat",  # no learnable weights; w_l == 1
            seed=todo,
            log_path=NOWL_OUT,
            **SHARED,
        )
        run_configs(configs, devices=devices, num_workers=num_workers, path=NOWL_OUT)
    else:
        print("[nowl] complete, skipping", flush=True)

    print("[done] all conditions complete", flush=True)


if __name__ == "__main__":
    main()
