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

    devices, num_workers, run_dp = setup_device_config()

    path = BASEPATH.joinpath("method_hparams_batch.jsonl")

    BASE_CONFIG = {
        "model_cls": BiTULER,
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
    BASE_CONFIG = {
        "model_cls": BiTULER,
        "dataset": "foursquare_TKY",
        "n_users": 400,
        "optimizer_cls": Adam,
        "time_levels": 1,
        "n_layers": 1,
        "lr": 1e-4,
        "n_hidden": 1024,
        "loc_embedding_factor": 1,
        "time_embedding_factor": 1 / 16,
        "subsample": 5000,
        "subsample_mode": "first",
        "log_path": path,
        "embedding_type": "lookup_weighted_concat",
        "loc_levels": [3, 4, 5],
        "discretization_rows": [100, 200, 300, 400, 500],
        "grow_factor": [2, 3, 4],
        "embedding_weight_factor": [1, 2],
    }

    # Generate all configurations

    configs += get_config_grid(**BASE_CONFIG)
    run_configs(
        configs, devices=devices, num_workers=num_workers, path=path, task="batch"
    )
