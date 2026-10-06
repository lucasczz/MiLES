from run import get_config_grid, get_missing_configs, run_configs, setup_device_config
from torch.optim import Adam
from pathlib import Path
from src.optim import AdamHD, DoG

from src.models import BiTULER, TULHOR, DeepTUL, TULVAE, MainTUL, T3S

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    # devices = ["cuda:4", "cuda:5", "cuda:0", "cuda:2"]
    # num_workers = 8

    # Usage
    devices, num_workers, _ = setup_device_config()

    lrs = {T3S: 1e-4, TULHOR: 1e-4, DeepTUL: 4e-4}
    seeds = [0, 1, 2, 3, 4]
    path = BASEPATH.joinpath("TUL.jsonl")
    # DeeprETA-style baseline (tab:emb_replay) is written to its own file so
    # the main results file above is untouched by it.
    path_deepreta = BASEPATH.joinpath("TUL_deepreta.jsonl")
    configs = []
    configs_deepreta = []
    for model in [
        BiTULER,
        TULHOR,
        DeepTUL,
        TULVAE,
        MainTUL,
        T3S,
    ]:
        for dataset in ["foursquare_NYC", "foursquare_TKY", "geolife"]:
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
                    "fourier",
                    "linear",
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
                subsample=None,
                seed=seeds,
                log_path=path,
            )

            # Evaluate replay methods (+ MiLES)
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
                replay_buffer_type=["recent", "balanced"],
                replay_batch_size=1,
                replay_buffer_size=1000,
            )
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
                replay_buffer_type=["recent", "balanced"],
                replay_batch_size=1,
                replay_buffer_size=1000,
            )

            # Evaluate adaptive optimizers
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=base_levels,
                time_levels=1,
                optimizer_cls=DoG,
                embedding_type="lookup_concat",
                discretization_rows=discretization_rows,
                discretization_shape="hex",
                aggregation_mode="grow",
                grow_factor=2,
                lr=1,
                n_hidden=1024,
                n_layers=n_layers,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                subsample=None,
                seed=seeds,
                log_path=path,
            )

            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=base_levels,
                time_levels=1,
                optimizer_cls=AdamHD,
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

            # DeeprETA-style embedding baseline: MiLES with -WL, -VD, and a
            # square grid (loc_levels=3, grow_factor=4, tuned via
            # tune_methods.py's "DeeprETA" grid).
            configs_deepreta += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=3,
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type="lookup_concat",
                discretization_shape="square",
                embedding_weight_factor=1,
                discretization_rows=discretization_rows,
                aggregation_mode="grow",
                grow_factor=4,
                lr=lr,
                n_hidden=1024,
                n_layers=n_layers,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                subsample=None,
                seed=seeds,
                log_path=path_deepreta,
            )

    # Fill in the run() default so every config carries an explicit value;
    # otherwise configs that don't override embedding_weight_factor merge as
    # NaN and get_missing_configs would treat already-completed runs as missing.
    for config in configs:
        config.setdefault("embedding_weight_factor", 2)

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
        "seed",
    ]
    # Note: "subsample" is deliberately excluded — run()'s _create_log_info
    # never logs it, so matching on it would KeyError against reports/*.jsonl.
    if path.exists():
        configs = get_missing_configs(configs, path, relevant_params)
    if path_deepreta.exists():
        configs_deepreta = get_missing_configs(configs_deepreta, path_deepreta, relevant_params)

    run_configs(configs, devices=devices, num_workers=num_workers, path=path)
    run_configs(configs_deepreta, devices=devices, num_workers=num_workers, path=path_deepreta)
