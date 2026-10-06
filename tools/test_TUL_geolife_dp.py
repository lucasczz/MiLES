from run import get_config_grid, run_configs, setup_device_config
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()

    path = BASEPATH.joinpath("TUL_geolife_dp.jsonl")
    seeds = [0]
    dataset = "geolife_dp"
    n_users = 150
    discretization_rows = 1200

    configs = []

    # MiLES
    configs += get_config_grid(
        dataset=dataset,
        model_cls=BiTULER,
        n_users=n_users,
        loc_levels=[4],
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="lookup_weighted_concat",
        discretization_rows=discretization_rows,
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=2,
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=None,
        seed=seeds,
        log_path=path,
    )

    # Default (lookup_concat)
    configs += get_config_grid(
        dataset=dataset,
        model_cls=BiTULER,
        n_users=n_users,
        loc_levels=[1],
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="lookup_concat",
        discretization_rows=discretization_rows,
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=2,
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=None,
        seed=seeds,
        log_path=path,
    )

    # Fourier
    configs += get_config_grid(
        dataset=dataset,
        model_cls=BiTULER,
        n_users=n_users,
        loc_levels=[1],
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="fourier",
        discretization_rows=discretization_rows,
        discretization_shape="hex",
        aggregation_mode="grow",
        embedding_weight_factor=4000,
        grow_factor=2,
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=None,
        seed=seeds,
        log_path=path,
    )

    print(f"Running {len(configs)} configs...")
    run_configs(configs, devices, num_workers, path, task="TUL")
