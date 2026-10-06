from run import get_config_grid, run_configs, setup_device_config
from torch.optim import Adam
from src.models import BiTULER
from pathlib import Path


BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()
    path = BASEPATH.joinpath("window_robustness_eval.jsonl")

    CONFIGS = {
        "window1": {
            "loc_levels": 4,
            "discretization_rows": 200,
            "grow_factor": 2,
            "embedding_weight_factor": 2,
        },
        "window2": {
            "loc_levels": 5,
            "discretization_rows": 400,
            "grow_factor": 4,
            "embedding_weight_factor": 2,
        },
    }

    configs = []
    for params in CONFIGS.values():
        configs += get_config_grid(
            model_cls=BiTULER,
            dataset="foursquare_TKY",
            n_users=400,
            time_levels=1,
            optimizer_cls=Adam,
            n_hidden=1024,
            n_layers=1,
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            lr=2e-4,
            log_path=path,
            embedding_type="lookup_weighted_concat",
            seed=[0, 1, 2, 3, 4],
            **params,
        )

    run_configs(configs, devices=devices, num_workers=num_workers, path=path, task="TUL")
