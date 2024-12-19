from run import get_config_grid, run_configs
from torch.optim import Adam
from src.models import TULHOR, T3S, BiTULER, MainTUL, DeepTUL, TULVAE
from pathlib import Path


BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    configs = []
    devices = ["cuda:1", "cuda:2", "cuda:3", "cuda:7"]
    # devices= ['cuda:0']
    num_workers = 12
    path = BASEPATH.joinpath("hparam_grid.jsonl")
    # configs += get_config_grid(
    #     dataset="foursquare_TKY",
    #     model_cls=TULHOR,
    #     n_users=400,
    #     loc_levels=1,
    #     time_levels=1,
    #     optimizer_cls=Adam,
    #     lr=[5e-5 * 2**i for i in range(5)],
    #     n_hidden=[512, 1024],
    #     n_layers=[1, 2],
    #     loc_embedding_factor=[0.5, 1],
    #     time_embedding_factor=[1 / 16, 1 / 8],
    #     n_heads=[8, 16],
    #     subsample=5000,
    #     log_path=path,
    # )
    configs += get_config_grid(
        dataset="foursquare_TKY",
        model_cls=T3S,
        n_users=400,
        loc_levels=1,
        time_levels=1,
        optimizer_cls=Adam,
        lr=[5e-5 * 2**i for i in range(5)],
        n_hidden=[512, 1024],
        n_layers=2,
        loc_embedding_factor=[0.5, 1],
        time_embedding_factor=[1 / 16, 1 / 8],
        n_heads=[8, 16],
        subsample=5000,
        log_path=path,
    )
    configs += get_config_grid(
        dataset="foursquare_TKY",
        model_cls=MainTUL,
        n_users=400,
        loc_levels=1,
        time_levels=1,
        optimizer_cls=Adam,
        lr=[1e-4 * 2**i for i in range(4)],
        n_hidden=[512, 1024],
        n_layers=2,
        loc_embedding_factor=[0.5, 1],
        time_embedding_factor=[1 / 16, 1 / 8],
        n_heads=[8, 16],
        lambduh=[2, 10],
        subsample=5000,
        log_path=path,
    )
    # configs += get_config_grid(
    #     dataset="foursquare_TKY",
    #     model_cls=BiTULER,
    #     n_users=400,
    #     loc_levels=1,
    #     time_levels=1,
    #     optimizer_cls=Adam,
    #     lr=[1e-4 * 2**i for i in range(4)],
    #     n_hidden=[512, 1024],
    #     n_layers=[1, 2],
    #     loc_embedding_factor=[0.5, 1],
    #     time_embedding_factor=[1 / 16, 1 / 8],
    #     subsample=5000,
    #     log_path=path,
    # )
    configs += get_config_grid(
        dataset="foursquare_TKY",
        model_cls=TULVAE,
        n_users=400,
        loc_levels=1,
        time_levels=1,
        optimizer_cls=Adam,
        lr=[1e-4 * 2**i for i in range(4)],
        n_hidden=[512, 1024],
        n_layers=2,
        loc_embedding_factor=[0.5, 1],
        time_embedding_factor=[1 / 16, 1 / 8],
        latent_dim=[50, 100],
        subsample=5000,
        log_path=path,
    )
    # configs += get_config_grid(
    #     dataset="foursquare_TKY",
    #     model_cls=DeepTUL,
    #     n_users=400,
    #     loc_levels=1,
    #     time_levels=1,
    #     optimizer_cls=Adam,
    #     lr=[1e-4 * 2**i for i in range(4)],
    #     n_hidden=[512, 1024],
    #     n_layers=[1, 2],
    #     loc_embedding_factor=[0.5, 1],
    #     time_embedding_factor=[1 / 16, 1 / 8],
    #     subsample=5000,
    #     log_path=path,
    # )

    run_configs(configs, devices=devices, num_workers=num_workers, path=path)
