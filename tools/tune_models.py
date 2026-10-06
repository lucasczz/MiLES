from run import get_config_grid, run_configs, setup_device_config
from torch.optim import Adam
from src.models import (
    TULHOR,
    T3S,
    BiTULER,
    MainTUL,
    DeepTUL,
    TULVAE,
    BiLSTM,
    Transformer,
    CNN,
    MLP,
)
from pathlib import Path
from copy import copy


BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    configs = []

    task = "DP"
    devices, num_workers, run_dp = setup_device_config()
    task = "DP" if run_dp else "TUL"


    # Define model-specific configurations
    if task == "TUL":
        path = BASEPATH.joinpath("model_hparams_TUL.jsonl")
        MODEL_CONFIGS = {
            TULHOR: {
                "lr": [5e-5 * 2**i for i in range(5)],
                "n_heads": [8, 16],
            },
            T3S: {
                "lr": [5e-5 * 2**i for i in range(5)],
                "n_layers": 2,
                "n_heads": [8, 16],
            },
            MainTUL: {
                "n_layers": 2,
                "n_heads": [8, 16],
                "lambduh": [2, 10],
            },
            BiTULER: {},
            TULVAE: {
                "n_layers": 2,
                "latent_dim": [50, 100],
            },
            DeepTUL: {},
        }
    else:
        path = BASEPATH.joinpath("model_hparams_DP.jsonl")
        MODEL_CONFIGS = {
            BiLSTM: {},
            Transformer: {
                "lr": [5e-5 * 2**i for i in range(5)],
                "n_heads": [8, 16],
                "n_hidden": 1024,
            },
            CNN: {
                "n_layers": [2, 3, 4],
                "n_filters": [128, 256],
            },
            MLP: {},
        }

    BASE_CONFIG = {
        "dataset": "foursquare_TKY",
        "n_users": 400,
        "optimizer_cls": Adam,
        "loc_levels": 1,
        "time_levels": 1,
        "n_layers": [1, 2],
        "lr": [1e-4 * 2**i for i in range(4)],
        "n_hidden": [512, 1024],
        "loc_embedding_factor": [1],
        "time_embedding_factor": 1 / 16,
        "subsample": 5000,
        "subsample_mode": "first",
        "log_path": path,
    }

    # Generate all configurations
    for model_cls, model_params in MODEL_CONFIGS.items():
        config = copy(BASE_CONFIG)
        config.update(model_params)
        configs += get_config_grid(model_cls=model_cls, **config)
    run_configs(configs, devices=devices, num_workers=num_workers, path=path, task=task)
