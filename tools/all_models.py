from run import get_config_grid, run_configs
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER, TULERG, TULERL, TULHOR, DeepTUL, TULVAE, MainTUL, T3S

BASEPATH = Path(__file__).parent.parent.joinpath("reports")
# TODO: model specific hparams

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    devices = ["cuda:1", "cuda:2", "cuda:3", "cuda:7"]
    path = BASEPATH.joinpath("all_models.jsonl")
    num_workers = 12
    configs = []
    for model in [
        BiTULER, TULERG, TULERL, 
        TULHOR, DeepTUL, TULVAE, MainTUL, T3S]:
        configs += get_config_grid(
            dataset=["foursquare_NYC", "foursquare_TKY"],
            model_cls=model,
            n_users=[800],
            loc_levels=4,
            time_levels=1,
            optimizer_cls=Adam,
            embedding_type="lookup_concat",
            discretization_rows=200,
            discretization_shape="hex",
            aggregation_mode="grow",
            grow_factor=4,
            lr=2e-4,
            n_hidden=1024,
            n_layers=1,
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=None,
            seed=seeds,
            log_path=path,
        )
        configs += get_config_grid(
            dataset=["foursquare_NYC", "foursquare_TKY"],
            model_cls=model,
            n_users=[800],
            loc_levels=1,
            time_levels=1,
            optimizer_cls=Adam,
            embedding_type="lookup_concat",
            discretization_rows=200,
            discretization_shape="hex",
            aggregation_mode="grow",
            grow_factor=4,
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
        #     dataset=["geolife"],
        #     model_cls=BiTULER,
        #     n_users=[150],
        #     loc_levels=4,
        #     time_levels=1,
        #     optimizer_cls=Adam,
        #     embedding_type="lookup_concat",
        #     discretization_rows=200,
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
        # configs += get_config_grid(
        #     dataset=["geolife"],
        #     model_cls=BiTULER,
        #     n_users=[150],
        #     loc_levels=1,
        #     time_levels=1,
        #     optimizer_cls=Adam,
        #     embedding_type="lookup_concat",
        #     discretization_rows=200,
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
    run_configs(configs, devices=devices, num_workers=num_workers, path=path)
