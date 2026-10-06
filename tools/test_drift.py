from run import get_config_grid, run_configs, setup_device_config
from torch.optim import Adam
from pathlib import Path
from src.optim import AdamHD, DoG

from src.models import BiTULER, TULHOR, DeepTUL, TULVAE, MainTUL, T3S

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    # num_workers = 8

    # Usage
    devices, num_workers, _ = setup_device_config()

    lrs = {T3S: 1e-4, TULHOR: 1e-4, DeepTUL: 4e-4}
    seeds = [0, 1, 2, 3, 4]

    drift_location = 30_000
    drift_users = [None, 40]
    # pre_drift_lon_limit = 0.8235
    pre_drift_lon_limit = 0.5

    path = BASEPATH.joinpath("TUL_drift_nyc.jsonl")
    configs = []
    for model in [BiTULER]:
        for dataset in ["foursquare_NYC"]:
            base_levels = 2 if model in [T3S, TULHOR] else 1
            n_users = 150 if dataset == "geolife" else 800
            discretization_rows = 1200 if dataset == "geolife" else 300
            lr = lrs.get(model, 2e-4)
            n_layers = 2 if model == DeepTUL else 1

            # Evaluate MiLES
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=[4],
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type="lookup_weighted_concat",
                discretization_rows=discretization_rows,
                discretization_shape="hex",
                aggregation_mode="grow",
                grow_factor=2,
                lr=lr,
                n_hidden=1024,
                n_layers=n_layers,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                subsample=None,
                seed=seeds,
                log_path=path,
                drift_location=drift_location,
                drift_users=drift_users,
                pre_drift_lon_limit=pre_drift_lon_limit,
            )

            # Evaluate different embedding methods
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=[base_levels],
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type=["lookup_concat"],
                discretization_rows=discretization_rows,
                discretization_shape="hex",
                aggregation_mode="grow",
                embedding_weight_factor=4000,
                grow_factor=2,
                lr=lr,
                n_hidden=1024,
                n_layers=n_layers,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                subsample=None,
                seed=seeds,
                log_path=path,
                drift_location=drift_location,
                drift_users=drift_users,
                pre_drift_lon_limit=pre_drift_lon_limit,
            )

    run_configs(configs, devices=devices, num_workers=num_workers, path=path)
