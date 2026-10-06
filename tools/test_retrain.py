from run import get_config_grid, run_configs, setup_device_config, get_missing_configs
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()

    seeds = [0, 1, 2, 3, 4]
    path = BASEPATH.joinpath("periodic_retrain.jsonl")
    configs = []

    for dataset in [
        "foursquare_NYC",
        "foursquare_TKY",
        "geolife",
    ]:
        n_users = 150 if dataset == "geolife" else 800
        discretization_rows = 800 if dataset == "geolife" else 400
        subsample = 8839 if dataset == "foursquare_TKY" else None

        for retrain_interval in [5000, 2500]:
            # MiLES
            configs += get_config_grid(
                dataset=dataset,
                model_cls=BiTULER,
                n_users=n_users,
                loc_levels=[5],
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type="lookup_weighted_concat",
                discretization_rows=discretization_rows,
                discretization_shape="hex",
                aggregation_mode="grow",
                grow_factor=4,
                lr=1e-4,
                n_hidden=1024,
                n_layers=1,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                embedding_weight_factor=2,
                subsample=subsample,
                subsample_mode="last",
                retrain_interval=retrain_interval,
                retrain_epochs=20,
                retrain_batch_size=64,
                seed=seeds,
                log_path=path,
            )

            # Default baseline
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
                embedding_weight_factor=4000,
                grow_factor=2,
                lr=1e-4,
                n_hidden=1024,
                n_layers=1,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                subsample=subsample,
                subsample_mode="last",
                retrain_interval=retrain_interval,
                retrain_epochs=20,
                retrain_batch_size=64,
                seed=seeds,
                log_path=path,
            )

    run_configs(
        configs,
        devices=devices,
        num_workers=num_workers,
        path=path,
        task="retrain",
    )