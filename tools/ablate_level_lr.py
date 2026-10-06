"""Per-level learning-rate ablation for the MiLES level weights.

This experiment tests hypothesis that the learnable level weights
``w_l`` act as an effective per-level learning rate (an optimization effect)
rather than adding representational capacity. Because the levels are
concatenated and both the embedding tables and the downstream weights are
learnable, ``w_l`` is non-identifiable: rescaling ``w_l`` and ``Z_l`` in
opposite directions leaves the model unchanged. What ``w_l`` does change is the
effective per-level learning rate, since a gradient step on ``Z_l`` moves the
level's contribution ``w_l * Z_l`` by a factor of ``w_l``.

We compare the -WL model (``w_l`` fixed at 1) with explicit per-level learning
rates against the full MiLES model (learnable ``w_l``). Only the two new
conditions are run here:

- ``no_wl_uniform_lr`` : lookup_concat + a single embedding LR multiplier set to
                         the overall mean learned weight (~1.45), testing whether
                         a roughly uniform increase in embedding LR reproduces the
                         gain.
- ``no_wl_matched_lr`` : lookup_concat + per-level embedding LR multipliers set
                         to the time-averaged learned weights, testing the
                         precise per-level match.

The two reference conditions are reused from ``reports/ablation_all.jsonl``,
which already contains, with an identical configuration and seeds 0-9:

- full MiLES : lookup_weighted_concat, embedding_weight_factor=2 (learnable w_l).
- -WL        : lookup_concat, embedding_weight_factor=2 (w_l = 1, single LR).

The matched multipliers are the time-averaged learned weights from
``reports/level_weights.jsonl`` (per level: [1.44, 1.36, 1.45, 1.55]).

Output: reports/level_lr_ablation.jsonl
"""

from pathlib import Path

from torch.optim import Adam
from tqdm import tqdm

from run import (
    get_config_grid,
    get_missing_configs,
    run_configs,
    run_with_kwargs,
    setup_device_config,
)
from src.models import BiTULER


BASEPATH = Path(__file__).parent.parent.joinpath("reports")

# Time-averaged learned level weights from reports/level_weights.jsonl.
MATCHED_LR_MULTIPLIERS = [1.44, 1.36, 1.45, 1.55]
# Single uniform multiplier at the overall mean learned weight (~1.45).
UNIFORM_LR_MULTIPLIERS = [1.45]


if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()

    path = BASEPATH.joinpath("level_lr_ablation.jsonl")
    seeds = [0, 1, 2, 3, 4]

    shared = dict(
        model_cls=BiTULER,
        dataset="foursquare_NYC",
        loc_levels=4,
        time_levels=1,
        optimizer_cls=Adam,
        embedding_weight_factor=2,
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
        n_users=[400, 800],
        seed=seeds,
        log_path=path,
    )

    configs = []
    # C. -WL with a uniform embedding learning-rate multiplier (sweep).
    configs += get_config_grid(
        embedding_type="lookup_concat",
        embedding_lr_multiplier=UNIFORM_LR_MULTIPLIERS,
        loc_lr_multipliers=None,
        **shared,
    )
    # D. -WL with per-level embedding learning rates matched to learned weights.
    configs += get_config_grid(
        fixed_kwargs={"loc_lr_multipliers": MATCHED_LR_MULTIPLIERS},
        embedding_type="lookup_concat",
        embedding_lr_multiplier=1.0,
        **shared,
    )

    relevant_params = [
        "dataset",
        "model_cls",
        "n_users",
        "loc_levels",
        "time_levels",
        "optimizer_cls",
        "embedding_type",
        "embedding_weight_factor",
        "embedding_lr_multiplier",
        "loc_lr_multipliers",
        "discretization_rows",
        "discretization_shape",
        "aggregation_mode",
        "grow_factor",
        "lr",
        "n_hidden",
        "n_layers",
        "loc_embedding_factor",
        "time_embedding_factor",
        "subsample",
        "seed",
    ]
    if path.exists():
        configs = get_missing_configs(configs, path, relevant_params)

    if num_workers == 1:
        for config in tqdm(configs):
            config["device"] = devices[0]
            config["verbose"] = False
            run_with_kwargs(config)
    else:
        run_configs(configs, devices=devices, num_workers=num_workers, path=path)
