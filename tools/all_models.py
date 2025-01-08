from run import get_config_grid, run_configs, get_missing_configs
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER, TULERG, TULERL, TULHOR, DeepTUL, TULVAE, MainTUL, T3S

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    devices = ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]
    path = BASEPATH.joinpath("all_models.jsonl")
    num_workers = 8
    lrs = {T3S: 1e-4, TULHOR: 1e-4, DeepTUL: 4e-4}

    configs = []
    for model in [
        BiTULER,
        TULHOR,
        TULERG,
        TULERL,
        DeepTUL,
        TULVAE,
        MainTUL,
        T3S,
    ]:
        for dataset in ["foursquare_NYC", "foursquare_TKY", "geolife"]:
            base_levels = 2 if model in [T3S, TULHOR] else 1
            n_users = 75 if dataset == "geolife" else 400
            discretization_rows = 800 if dataset == "geolife" else 200
            lr = lrs.get(model, 2e-4)
            n_layers = 2 if model == DeepTUL else 1
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
            )
            # Baselines
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=[base_levels],
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type="lookup_concat",
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
            )

    # configs = get_missing_configs(
    #     configs, path, relevant_params=["model_cls", "dataset", "loc_levels", "seed"]
    # )

    run_configs(configs, devices=devices, num_workers=num_workers, path=path)
