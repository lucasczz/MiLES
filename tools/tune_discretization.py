from run import get_config_grid, run_with_kwargs
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, Process, Manager
from torch.optim import Adam
from src.data.tracker import handle_queue

from src.models import BiTULER, TULHOR

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    devices = ["cuda:1", "cuda:2", "cuda:3", "cuda:7"]
    path = BASEPATH.joinpath("discretization_grid_tune.jsonl")
    num_workers = 8
    configs = []
    for model, lr in [(BiTULER, 2e-4), (TULHOR, 1e-4)]:
        configs += get_config_grid(
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
            embedding_type=["lookup_sum", "lookup_concat"],
            embedding_weight_factor=[1],
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=5000,
            device="cuda:0",
            log_path=path,
        )
        # Baselines
        configs += get_config_grid(
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
            embedding_type=["lookup_sum", "lookup_concat"],
            embedding_weight_factor=[1, 2, 3],
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=5000,
            device="cuda:0",
            log_path=path,
        )
        configs += get_config_grid(
            dataset="foursquare_TKY",
            model_cls=model,
            n_users=400,
            loc_levels=4,
            time_levels=1,
            optimizer_cls=Adam,
            discretization_rows=400,
            discretization_shape="hex",
            aggregation_mode="grow",
            grow_factor=4,
            lr=lr,
            n_hidden=1024,
            n_layers=1,
            embedding_type=["lookup_sum", "lookup_concat"],
            embedding_weight_factor=[0.5, 1, 2, 4],
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=5000,
            device="cuda:0",
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
