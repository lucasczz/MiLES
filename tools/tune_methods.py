from run import get_config_grid, run_configs, setup_device_config
from copy import copy
from torch.optim import Adam
from src.models import BiTULER, BiLSTM
from pathlib import Path


BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    configs = []
    devices, num_workers, run_dp = setup_device_config()
    task = "DP" if run_dp else "TUL"

    METHOD_CONFIGS = {
        "MiLES": {
            "embedding_type": "lookup_weighted_concat",
            "loc_levels": [3, 4, 5],
            "discretization_rows": [100, 200, 300, 400, 500],
            "grow_factor": [2, 3, 4],
            "embedding_weight_factor": [1, 2],
        },
        "Fourier": {
            "embedding_type": "fourier",
            "embedding_weight_factor": [500, 1000, 2000, 4000, 8000, 16000],
        },
        "DeeprETA": {
            "embedding_type": "lookup_concat",
            "discretization_shape": "square",
            "loc_levels": [3, 4, 5],
            "discretization_rows": [100, 200, 300, 400, 500],
            "grow_factor": [2, 3, 4],
            "embedding_weight_factor": 1,
        },
        "Space2Vec": {
            "embedding_type": "space2vec",
            "loc_levels": 1,
            "embedding_weight_factor": [500, 1000, 2000, 4000, 8000, 16000],
        },
    }
    if task == "TUL":
        path = BASEPATH.joinpath("method_hparams_TUL.jsonl")
        BASE_CONFIG = {
            "model_cls": BiTULER,
            "dataset": "foursquare_TKY",
            "n_users": 400,
            "loc_levels": 1,
            "time_levels": 1,
            "optimizer_cls": Adam,
            "n_hidden": 1024,
            "n_layers": 1 if task == "TUL" else 2,
            "loc_embedding_factor": 1,
            "time_embedding_factor": 1 / 16,
            "lr": 2e-4 if task == "TUL" else 1e-4,
            "subsample": 5000,
            "log_path": path,
            "embedding_type": "lookup_concat",
        }
        METHOD_CONFIGS["Balanced"] = (
            {
                "lr": [1e-4 * 2**i for i in range(4)],
                "replay_batch_size": [1, 2, 4],
                "replay_buffer_size": 1000,
                "replay_buffer_type": "balanced",
            },
        )
    else:
        path = BASEPATH.joinpath("method_hparams_DP.jsonl")
        BASE_CONFIG = {
            "model_cls": BiLSTM,
            "dataset": "foursquare_TKY",
            "n_users": 400,
            "loc_levels": 1,
            "time_levels": 1,
            "optimizer_cls": Adam,
            "n_hidden": 1024,
            "n_layers": 2,
            "loc_embedding_factor": 1,
            "time_embedding_factor": 1 / 16,
            "user_embedding_factor": 1 / 4,
            "lr": 1e-4,
            "subsample": 5000,
            "log_path": path,
            "embedding_type": "lookup_concat",
        }
        METHOD_CONFIGS["Random"] = (
            {
                "lr": [1e-4 * 2**i for i in range(4)],
                "replay_batch_size": [1, 2, 4],
                "replay_buffer_size": 1000,
                "replay_buffer_type": "random",
            },
        )

    # Generate all configurations
    for method, model_params in METHOD_CONFIGS.items():
        config = copy(BASE_CONFIG)
        config.update(model_params)
        configs += get_config_grid(**config)

    run_configs(configs, devices=devices, num_workers=num_workers, path=path, task=task)
