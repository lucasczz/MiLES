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


if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()

    path = BASEPATH.joinpath("lookup_level0_embedding_dim_ablation.jsonl")
    seeds = [0, 1, 2, 3, 4]

    configs = get_config_grid(
        dataset="foursquare_NYC",
        model_cls=BiTULER,
        n_users=800,
        loc_levels=1,
        loc_level=0,
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="lookup_concat",
        embedding_weight_factor=2,
        discretization_rows=200,
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=4,
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=[0.25, 0.5, 1],
        time_embedding_factor=1 / 16,
        subsample=None,
        seed=seeds,
        log_path=path,
    )

    relevant_params = [
        "dataset",
        "model_cls",
        "n_users",
        "loc_levels",
        "loc_level",
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
