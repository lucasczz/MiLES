from run import grid_search
from torch.optim import Adam

from src.models import BiTULER

if __name__ == "__main__":
    grid_search(
        dataset="foursquare_TKY",
        model_cls=BiTULER,
        n_users=400,
        loc_levels=[2, 3, 4],
        time_levels=1,
        optimizer_cls=Adam,
        discretization_rows=[100, 200, 300, 400, 500, 600],
        discretization_shape=["diamond", "square"],
        aggregation_mode="group",
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=5000,
        device="cuda:0",
        log_path="discretization_grid_square.jsonl",
    )
    grid_search(
        dataset="foursquare_TKY",
        model_cls=BiTULER,
        n_users=400,
        loc_levels=[2, 3, 4],
        time_levels=1,
        optimizer_cls=Adam,
        discretization_rows=[100, 200, 300, 400, 500, 600],
        discretization_shape=["diamond", "square"],
        aggregation_mode="grow",
        grow_factor=[2, 3, 4],
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=5000,
        device="cuda:0",
        log_path="discretization_grid.jsonl",
    )
