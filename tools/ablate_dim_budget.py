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

# (dataset, n_users, discretization_rows) for the high user-count variants.
DATASETS = [
    ("foursquare_NYC", 800, 200),
]
# total loc embedding dim = loc_embedding_factor * n_hidden (n_hidden = 1024).
LOC_EMBEDDING_FACTORS = [0.125, 0.25, 0.5, 1.0, 2.0]


if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()

    path = BASEPATH.joinpath("dim_budget_sweep.jsonl")
    seeds = [0, 1, 2]

    configs = []
    for dataset, n_users, discretization_rows in DATASETS:
        shared = dict(
            model_cls=BiTULER,
            dataset=dataset,
            n_users=n_users,
            time_levels=1,
            optimizer_cls=Adam,
            discretization_rows=discretization_rows,
            discretization_shape="hex",
            aggregation_mode="grow",
            grow_factor=2,
            lr=2e-4,
            n_hidden=1024,
            n_layers=1,
            loc_embedding_factor=LOC_EMBEDDING_FACTORS,
            time_embedding_factor=1 / 16,
            subsample=None,
            seed=seeds,
            log_path=path,
        )
        # MiLES (four levels).
        configs += get_config_grid(
            loc_levels=4,
            embedding_type="lookup_weighted_concat",
            embedding_weight_factor=2,
            **shared,
        )
        # POI lookup (single level), the default embedding.
        configs += get_config_grid(
            loc_levels=1,
            embedding_type="lookup_concat",
            embedding_weight_factor=2,
            **shared,
        )
        # Fourier features (continuous-encoder stand-in).
        configs += get_config_grid(
            loc_levels=1,
            embedding_type="fourier",
            embedding_weight_factor=4000,
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
