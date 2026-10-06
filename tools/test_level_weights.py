from run import get_config_grid, run_configs, setup_device_config
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()

    path = BASEPATH.joinpath("level_weights.jsonl")
    model = BiTULER
    dataset = "foursquare_NYC"
    discretization_rows = 800 if dataset == "geolife" else 200

    seeds = [0, 1, 2, 3, 4]
    configs = get_config_grid(
        dataset=dataset,
        model_cls=model,
        n_users=800,
        loc_levels=4,
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="lookup_weighted_concat",
        discretization_rows=discretization_rows,
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=2,
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=None,
        seed=seeds,
        log_path=path,
        track_embeddings=True,
    )

    run_configs(configs, devices=devices, num_workers=num_workers, path=path)
