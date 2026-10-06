from torch.optim import Adam
from pathlib import Path
from run import get_config_grid, get_missing_configs, run_configs, setup_device_config

from src.models.next_location import BiLSTM, Transformer, MLP, CNN

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()

    path = BASEPATH.joinpath("DP.jsonl")
    lrs = {MLP: 2e-4}
    subsample = 50000
    n_hidden = 1024

    seeds = [0, 1, 2, 3, 4]
    n_layers = {Transformer: 1, MLP: 3}

    configs = []
    for model in [
        BiLSTM,
        Transformer,
        CNN,
        MLP,
    ]:
        for dataset in ["foursquare_NYC", "foursquare_TKY", "geolife_dp"]:
            n_users = 150 if dataset == "geolife_dp" else 800
            discretization_rows = 800 if dataset == "geolife" else 200
            lr = lrs.get(model, 1e-4)
            n_layer = n_layers.get(model, 2)
            if model != BiLSTM:
                # Evaluate MiLES
                configs += get_config_grid(
                    dataset=dataset,
                    model_cls=model,
                    n_users=n_users,
                    loc_levels=5,
                    time_levels=1,
                    optimizer_cls=Adam,
                    embedding_type="lookup_weighted_concat",
                    discretization_rows=discretization_rows,
                    discretization_shape="hex",
                    aggregation_mode="grow",
                    grow_factor=4,
                    lr=lr,
                    n_hidden=n_hidden,
                    n_layers=n_layer,
                    embedding_weight_factor=2,
                    loc_embedding_factor=1,
                    time_embedding_factor=1 / 16,
                    user_embedding_factor=1 / 4,
                    subsample=subsample,
                    seed=seeds,
                    log_path=path,
                )
                # Evaluate other embedding techniques
                configs += get_config_grid(
                    dataset=dataset,
                    model_cls=model,
                    n_users=n_users,
                    loc_levels=[1],
                    time_levels=1,
                    optimizer_cls=Adam,
                    embedding_type=[
                        "lookup_concat",
                        "fourier",
                        "linear",
                    ],
                    discretization_rows=discretization_rows,
                    embedding_weight_factor=2000,
                    discretization_shape="hex",
                    aggregation_mode="grow",
                    lr=lr,
                    n_hidden=n_hidden,
                    n_layers=n_layer,
                    loc_embedding_factor=1,
                    time_embedding_factor=1 / 16,
                    user_embedding_factor=1 / 4,
                    subsample=subsample,
                    seed=seeds,
                    log_path=path,
                )
                # Evaluate replay techniques (+ MiLES)
                configs += get_config_grid(
                    dataset=dataset,
                    model_cls=model,
                    n_users=n_users,
                    loc_levels=1,
                    time_levels=1,
                    optimizer_cls=Adam,
                    embedding_type="lookup_concat",
                    lr=1e-4,
                    n_hidden=1024,
                    n_layers=n_layer,
                    loc_embedding_factor=1,
                    time_embedding_factor=1 / 16,
                    user_embedding_factor=1 / 4,
                    replay_batch_size=4,
                    replay_buffer_size=1000,
                    subsample=subsample,
                    log_path=path,
                    replay_buffer_type=["random", "recent"],
                    seed=seeds,
                )
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=5,
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type="lookup_weighted_concat",
                discretization_rows=discretization_rows,
                discretization_shape="hex",
                aggregation_mode="grow",
                grow_factor=4,
                lr=1e-4,
                n_hidden=1024,
                n_layers=n_layer,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                user_embedding_factor=1 / 4,
                replay_batch_size=4,
                replay_buffer_size=1000,
                subsample=subsample,
                log_path=path,
                replay_buffer_type=["random", "recent"],
                seed=seeds,
            )

    # Fill in run()'s actual defaults for any field a config block left
    # unset (e.g. the replay blocks above don't override discretization_rows
    # or embedding_weight_factor); this exactly mirrors runtime behavior, so
    # get_missing_configs can match against reports/DP.jsonl without treating
    # already-completed runs as missing due to an unset-vs-NaN mismatch.
    run_defaults = {
        "n_users": None,
        "loc_levels": 1,
        "time_levels": 1,
        "embedding_type": "lookup_concat",
        "embedding_weight_factor": 2,
        "loc_embedding_factor": 1.0,
        "time_embedding_factor": 0.25,
        "discretization_rows": 100,
        "discretization_shape": "hex",
        "grow_factor": 2,
        "aggregation_mode": "grow",
        "seed": 42,
    }
    for config in configs:
        for key, default in run_defaults.items():
            config.setdefault(key, default)

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
        "user_embedding_factor",
        "seed",
    ]
    # Note: "subsample" is deliberately excluded — run()'s _create_log_info
    # never logs it, so matching on it would KeyError against reports/DP.jsonl.
    if path.exists():
        configs = get_missing_configs(configs, path, relevant_params)

    run_configs(configs, devices=devices, num_workers=num_workers, path=path, task="DP")
