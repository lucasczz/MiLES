from run import grid_search
from src.models.deeptul import DeepTUL
from src.models.maintul import MainTUL
from torch.optim import Adam

from src.models.t3s import T3S
from src.models.tuler import BiTULER
from src.models.tulvae import TULVAE

if __name__ == "__main__":

    grid_search(
        dataset="foursquare_TKY",
        model_cls=T3S,
        n_users=400,
        loc_levels=1,
        time_levels=1,
        optimizer_cls=Adam,
        lr=[1e-4 * 2**i for i in range(6)],
        n_hidden=[128, 256, 512, 1024],
        n_layers=[1, 2],
        loc_embedding_factor=[0.5, 1],
        time_embedding_factor=0.25,
        n_heads=[8, 16],
        dropout=[0.0, 0.1],
        subsample=5000,
        device="cuda:0",
        log_path="t3s_grid.jsonl",
    )
    grid_search(
        dataset="foursquare_TKY",
        model_cls=MainTUL,
        n_users=400,
        loc_levels=1,
        time_levels=1,
        optimizer_cls=Adam,
        lr=[1e-4 * 2**i for i in range(6)],
        n_hidden=[128, 256, 512, 1024],
        n_layers=1,
        loc_embedding_factor=[0.5, 1],
        time_embedding_factor=0.25,
        n_heads=[8, 16],
        lambduh=[2, 10],
        dropout=[0.0, 0.1],
        subsample=5000,
        device="cuda:0",
        log_path="maintul_grid.jsonl",
    )
    grid_search(
        dataset="foursquare_TKY",
        model_cls=BiTULER,
        n_users=400,
        loc_levels=1,
        time_levels=1,
        optimizer_cls=Adam,
        lr=[1e-4 * 2**i for i in range(6)],
        n_hidden=[128, 256, 512, 1024],
        n_layers=[1, 2],
        loc_embedding_factor=[0.5, 1],
        time_embedding_factor=0.25,
        dropout=[0.0, 0.1],
        subsample=5000,
        device="cuda:0",
        log_path="bituler_grid.jsonl",
    )
    grid_search(
        dataset="foursquare_TKY",
        model_cls=TULVAE,
        n_users=400,
        loc_levels=1,
        time_levels=1,
        optimizer_cls=Adam,
        lr=[1e-4 * 2**i for i in range(6)],
        n_hidden=[128, 256, 512, 1024],
        n_layers=1,
        loc_embedding_factor=[0.5, 1],
        time_embedding_factor=0.25,
        latent_dim=[50, 100],
        dropout=[0.0, 0.1],
        subsample=5000,
        device="cuda:0",
        log_path="tulvae_grid.jsonl",
    )
