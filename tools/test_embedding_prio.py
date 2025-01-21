from run import get_config_grid, run_configs
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    devices = ["cuda:4", "cuda:5"]
    path = BASEPATH.joinpath("emb_prio_new.jsonl")
    num_workers = 4

    configs = []
    for model in [
        BiTULER,
    ]:
        for dataset in ["foursquare_NYC"]:
            n_users = 75 if dataset == "geolife" else 400
            discretization_rows = 800 if dataset == "geolife" else 200
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=[4],
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type=["lookup_weighted_concat", "lookup_concat"],
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
