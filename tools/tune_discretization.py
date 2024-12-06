from run import grid_search
from torch.optim import Adam

from src.models import BiTULER, TULHOR

if __name__ == "__main__":
    for model, lr in [(BiTULER, 2e-4), (TULHOR, 1e-4)]:
        grid_search(
            dataset="foursquare_TKY",
            model_cls=model,
            n_users=400,
            loc_levels=[2, 3, 4],
            time_levels=1,
            optimizer_cls=Adam,
            discretization_rows=[100, 200, 300, 400, 500],
            discretization_shape=["hex"],
            aggregation_mode="grow",
            grow_factor=[2, 3, 4],
            lr=lr,
            n_hidden=1024,
            n_layers=1,
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=5000,
            device="cuda:0",
            log_path="discretization_tune_grid_cat.jsonl",
        )
        # Baselines
        grid_search(
            dataset="foursquare_TKY",
            model_cls=model,
            n_users=400,
            loc_levels=1,
            time_levels=1,
            optimizer_cls=Adam,
            discretization_rows=100,
            discretization_shape="hex",
            aggregation_mode="grow",
            lr=lr,
            n_hidden=1024,
            n_layers=1,
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=5000,
            device="cuda:0",
            log_path="discretization_tune_grid_cat.jsonl",
        )