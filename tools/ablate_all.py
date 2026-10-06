from run import get_config_grid, run_configs, setup_device_config
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    path = BASEPATH.joinpath("ablation_all.jsonl")
    model = BiTULER
    lr = 2e-4

    configs = []

    for dataset in ["foursquare_NYC", "foursquare_TKY", "geolife"]:
        n_users = 75 if dataset == "geolife" else 400
        discretization_rows = 800 if dataset == "geolife" else 200

        # Test complete MiLES embeddings
        configs += get_config_grid(
            dataset=dataset,
            model_cls=model,
            n_users=n_users,
            loc_levels=4,
            time_levels=1,
            optimizer_cls=Adam,
            embedding_type="lookup_weighted_concat",
            embedding_weight_factor=2,
            discretization_rows=discretization_rows,
            discretization_shape="hex",
            aggregation_mode="grow",
            grow_factor=2,
            lr=lr,
            n_hidden=1024,
            n_layers=1,
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=None,
            seed=seeds,
            log_path=path,
        )

        # Test removing levels
        configs += get_config_grid(
            dataset=dataset,
            model_cls=model,
            n_users=n_users,
            loc_levels=4,
            loc_level=[-1, -2, -3, -4],
            time_levels=1,
            optimizer_cls=Adam,
            embedding_type="lookup_weighted_concat",
            embedding_weight_factor=2,
            discretization_rows=discretization_rows,
            discretization_shape="hex",
            aggregation_mode="grow",
            grow_factor=2,
            lr=lr,
            n_hidden=1024,
            n_layers=1,
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=None,
            seed=seeds,
            log_path=path,
        )

        # Test \wo embedding dim weighting
        configs += get_config_grid(
            dataset=dataset,
            model_cls=model,
            n_users=n_users,
            loc_levels=4,
            loc_level=[None],
            time_levels=1,
            optimizer_cls=Adam,
            embedding_type="lookup_weighted_concat",
            embedding_weight_factor=1,
            discretization_rows=discretization_rows,
            discretization_shape="hex",
            aggregation_mode="grow",
            grow_factor=2,
            lr=lr,
            n_hidden=1024,
            n_layers=1,
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=None,
            seed=seeds,
            log_path=path,
        )

        # Test \wo dynamic weighting
        configs += get_config_grid(
            dataset=dataset,
            model_cls=model,
            n_users=n_users,
            loc_levels=4,
            loc_level=[None],
            time_levels=1,
            optimizer_cls=Adam,
            embedding_type="lookup_concat",
            embedding_weight_factor=2,
            discretization_rows=discretization_rows,
            discretization_shape="hex",
            aggregation_mode="grow",
            grow_factor=2,
            lr=lr,
            n_hidden=1024,
            n_layers=1,
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=None,
            seed=seeds,
            log_path=path,
        )
    run_configs(configs, devices=devices, num_workers=num_workers, path=path)
