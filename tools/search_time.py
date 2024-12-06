from run import grid_search
from torch.optim import Adam

from src.models import BiTULER, TULHOR

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]

    grid_search(
        dataset="foursquare_NYC",
        model_cls=BiTULER,
        n_users=400,
        loc_levels=1,
        time_levels=[1, 2, 3, 4],
        optimizer_cls=Adam,
        embedding_type="lookup_weighted_sum",
        discretization_rows=500,
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=3,
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=None,
        seed=seeds,
        device="cuda:7",
        log_path="time_grid.jsonl",
    )

    # Baselines
    grid_search(
        dataset="foursquare_NYC",
        model_cls=TULHOR,
        n_users=400,
        loc_levels=1,
        time_levels=[1, 2, 3, 4],
        optimizer_cls=Adam,
        discretization_rows=500,
        embedding_type="lookup_weighted_sum",
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=3,
        lr=1e-4,
        n_heads=16,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=None,
        seed=seeds,
        device="cuda:7",
        log_path="time_grid.jsonl",
    )
