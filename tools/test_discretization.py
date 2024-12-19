from run import get_config_grid, run_configs
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER, TULHOR

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    devices = ["cuda:1", "cuda:2", "cuda:3", "cuda:7"]
    path = BASEPATH.joinpath("discretization_grid.jsonl")
    num_workers = 12
    configs = []
    for embedding_type in ["lookup_concat", "lookup_sum"]:
        configs += get_config_grid(
            dataset="foursquare_NYC",
            model_cls=BiTULER,
            n_users=400,
            loc_levels=[3, 4],
            time_levels=1,
            optimizer_cls=Adam,
            embedding_type=embedding_type,
            embedding_weight_factor=2,
            discretization_rows=[100, 200, 300, 400, 500],
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
        )
        # configs += get_config_grid(
        #     dataset="foursquare_NYC",
        #     model_cls=BiTULER,
        #     n_users=400,
        #     loc_levels=1,
        #     time_levels=1,
        #     optimizer_cls=Adam,
        #     discretization_rows=100,
        #     embedding_type=embedding_type,
        #     embedding_weight_factor=1,
        #     discretization_shape="hex",
        #     aggregation_mode="grow",
        #     grow_factor=4,
        #     lr=2e-4,
        #     n_hidden=1024,
        #     n_layers=1,
        #     loc_embedding_factor=1,
        #     time_embedding_factor=1 / 16,
        #     subsample=None,
        #     seed=seeds,
        #     log_path=path,
        # )

    # Concat TULHOR
    configs += get_config_grid(
        dataset="foursquare_NYC",
        model_cls=TULHOR,
        n_users=400,
        loc_levels=[2, 3, 4],
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="lookup_concat",
        embedding_weight_factor=1,
        discretization_rows=[100, 200, 300, 400, 500],
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=4,
        lr=1e-4,
        n_hidden=1024,
        n_layers=1,
        n_heads=16,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=None,
        seed=seeds,
        log_path=path,
    )

    # Baselines
    configs += get_config_grid(
        dataset="foursquare_NYC",
        model_cls=TULHOR,
        n_users=400,
        loc_levels=1,
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="lookup_concat",
        embedding_weight_factor=1,
        discretization_rows=400,
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=4,
        lr=1e-4,
        n_hidden=1024,
        n_layers=1,
        n_heads=16,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        seed=seeds,
        subsample=None,
        log_path=path,
    )
    run_configs(configs, devices=devices, num_workers=num_workers, path=path)
