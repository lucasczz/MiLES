from run import get_config_grid, run_with_kwargs
from torch.optim import Adam
from multiprocessing import Pool, Manager, Process
from pathlib import Path
from tqdm import tqdm

from src.models import BiTULER, TULHOR
from src.data.tracker import handle_queue

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    devices = ["cuda:4", "cuda:5", "cuda:7"]
    path = BASEPATH.joinpath("discretization_grid_new.jsonl")
    num_workers = 2
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
            subsample=500,
            seed=seeds,
            log_path=path,
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
            subsample=500,
            seed=seeds,
            log_path=path,
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
        subsample=500,
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
        subsample=500,
        log_path=path,
    )
    manager = Manager()
    q = manager.Queue()

    # Distribute accross GPUs
    for i, entry in enumerate(configs):
        entry["device"] = devices[i % len(devices)]
        entry["verbose"] = False
        entry["write_queue"] = q

    pool = Pool(processes=num_workers)

    # Start a dedicated process for the queue
    queue_process = Process(target=handle_queue, args=(q, path))
    queue_process.start()

    # Use tqdm for progress bar
    results = list(tqdm(pool.imap(run_with_kwargs, configs), total=len(configs)))

    q.put("kill")
    pool.close()
    pool.join()
