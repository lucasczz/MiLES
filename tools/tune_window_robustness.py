from run import get_config_grid, run_configs, setup_device_config
from torch.optim import Adam
from src.models import BiTULER
from pathlib import Path


BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()
    path = BASEPATH.joinpath("method_hparams_TUL_window2.jsonl")

    configs = get_config_grid(
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
        subsample=5000,
        subsample_start=20000,
        log_path=path,
        embedding_type="lookup_weighted_concat",
        loc_levels=[3, 4, 5],
        discretization_rows=[100, 200, 300, 400, 500],
        grow_factor=[2, 3, 4],
        embedding_weight_factor=[1, 2],
    )

    run_configs(configs, devices=devices, num_workers=num_workers, path=path, task="TUL")
