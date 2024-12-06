from run import get_config_grid, run_with_kwargs
from torch.optim import Adam
from multiprocessing import Pool
from tqdm import tqdm

from src.models import BiTULER, TULHOR

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    devices = ["cuda:4", "cuda:5", "cuda:7"]
    num_workers = 12
    configs = []
    for embedding_type in ["lookup_concat", "lookup_sum"]:
        # Concat
        configs += get_config_grid(
            dataset="foursquare_NYC",
            model_cls=BiTULER,
            n_users=400,
            loc_levels=[2, 3, 4],
            time_levels=1,
            optimizer_cls=Adam,
            embedding_type=embedding_type,
            discretization_rows=[100, 200, 300, 400, 500],
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
            log_path="discretization_grid_new.jsonl",
        )
        configs += get_config_grid(
            dataset="foursquare_NYC",
            model_cls=BiTULER,
            n_users=400,
            loc_levels=1,
            time_levels=1,
            optimizer_cls=Adam,
            discretization_rows=100,
            embedding_type=embedding_type,
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
            log_path="discretization_grid_new.jsonl",
        )

    # Concat TULHOR
    configs += get_config_grid(
        dataset="foursquare_NYC",
        model_cls=TULHOR,
        n_users=400,
        loc_levels=[2, 3, 4],
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="lookup_concat",
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
        log_path="discretization_grid_new.jsonl",
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
        discretization_rows=100,
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
        log_path="discretization_grid_new.jsonl",
    )

    # Distribute accross GPUs
    for i, entry in enumerate(configs):
        entry["device"] = devices[i % len(devices)]
        entry['verbose'] = False

    with Pool(processes=num_workers) as pool:
        # Use tqdm for progress bar
        list(tqdm(pool.imap(run_with_kwargs, configs), total=len(configs)))
