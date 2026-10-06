from run import get_config_grid, run_configs, setup_device_config, get_missing_configs
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER, TULHOR, DeepTUL, T3S

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    # devices = ["cuda:4", "cuda:5", "cuda:0", "cuda:2"]
    # num_workers = 8

    # Usage
    devices, num_workers, _ = setup_device_config()

    lrs = {T3S: 1e-4, TULHOR: 1e-4, DeepTUL: 4e-4}
    seeds = [0, 1, 2, 3, 4]
    path = BASEPATH.joinpath("online_v_batch_v2.jsonl")
    configs = []
    for model in [
        BiTULER,
        # TULHOR,
        # DeepTUL,
        # TULVAE,
        # MainTUL,
        # T3S,
    ]:
        for dataset in [
            "foursquare_NYC",
            "foursquare_TKY",
            "geolife",
        ]:
            base_levels = 2 if model in [T3S, TULHOR] else 1
            n_users = 150 if dataset == "geolife" else 800
            discretization_rows = 800 if dataset == "geolife" else 400
            lr = lrs.get(model, 1e-4)
            n_layers = 2 if model == DeepTUL else 1
            subsample = 8839 if dataset == "foursquare_TKY" else None

            # Evaluate MiLES
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=[5],
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type="lookup_weighted_concat",
                discretization_rows=discretization_rows,
                discretization_shape="hex",
                aggregation_mode="grow",
                grow_factor=4,
                lr=lr,
                n_hidden=1024,
                n_layers=n_layers,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                embedding_weight_factor=2,
                subsample=subsample,
                subsample_mode="last",
                seed=seeds,
                log_path=path,
            )

            # Evaluate different embedding methods
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=[base_levels],
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type=[
                    "lookup_concat",
                ],
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
                subsample=subsample,
                subsample_mode="last",
                seed=seeds,
                log_path=path,
            )
    # configs = get_missing_configs(
    #     configs, path, relevant_params=["dataset", "embedding_type", "seed"]
    # )

    run_configs(
        configs, devices=devices, num_workers=num_workers, path=path, task="batch"
    )
