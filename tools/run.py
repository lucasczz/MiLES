from collections import deque
from copy import copy, deepcopy
from itertools import product
from typing import Dict, Optional, List
import torch
from tqdm import tqdm
from torch.optim import Adam
import multiprocessing as mp
import pathlib
import pandas as pd

from src.data.tracker import JSONTracker, handle_queue, EmbeddingWeightTracker
from src.data.utils import get_dataloader


def get_config_grid(search_space=None, fixed_kwargs=None, **kwargs):
    if search_space is None:
        search_space = [{}]
    if fixed_kwargs is None:
        fixed_kwargs = {}
    results = []
    for row in search_space:
        row.update(kwargs)
        fixed_attributes = fixed_kwargs.copy()
        variable_keys = []
        variable_values = []
        for key, value in row.items():
            if isinstance(value, (tuple, list)):
                variable_values.append(value)
                variable_keys.append(key)
            else:
                fixed_attributes[key] = value
        configs = [
            fixed_attributes | dict(zip(variable_keys, prod_values))
            for prod_values in product(*variable_values)
        ]
        results.extend(configs)
    return results


def get_missing_configs(configs, path_done, relevant_params):
    df_planned = pd.DataFrame.from_records(configs)
    df_planned["model_cls"] = df_planned["model_cls"].apply(lambda x: x.__name__)
    df_done = pd.read_json(path_done, orient="records", lines=True)

    # Match on specific columns
    diff_df = df_planned.merge(df_done, on=relevant_params, how="left", indicator=True)
    df_missing = diff_df[diff_df["_merge"] == "left_only"].drop("_merge", axis=1)
    return [configs[i] for i in df_missing.index]


def run_configs(
    configs: List[Dict], devices: List[str], num_workers: int, path: pathlib.Path
):
    manager = mp.Manager()
    q = manager.Queue()

    # Distribute accross GPUs
    for i, entry in enumerate(configs):
        entry["device"] = devices[i % len(devices)]
        entry["verbose"] = False
        entry["write_queue"] = q

    pool = mp.Pool(processes=num_workers)

    # Start a dedicated process for the queue
    queue_process = mp.Process(target=handle_queue, args=(q, path))
    queue_process.start()

    # Use tqdm for progress bar
    list(tqdm(pool.imap(run_with_kwargs, configs), total=len(configs)))

    q.put("kill")
    pool.close()
    pool.join()


def run(
    dataset: str,
    model_cls: torch.nn.Module,
    n_users: int,
    loc_levels: int = 1,
    time_levels: int = 1,
    optimizer_cls: torch.optim.Optimizer = Adam,
    lr: float = 1e-3,
    n_hidden: int = 128,
    n_layers: int = 1,
    embedding_type: str = "lookup_concat",
    embedding_weight_factor: float = 2,
    loc_embedding_factor: float = 1.0,
    time_embedding_factor: float = 0.25,
    dropout: float = 0.0,
    device: torch.device = "cuda:0",
    log_path: str = "test.jsonl",
    verbose: bool = True,
    history_length: int = 1000,
    batch_size: int = 1,
    subsample: Optional[int] = None,
    discretization_rows: int = 100,
    discretization_shape: str = "hex",
    grow_factor: int = 2,
    aggregation_mode: str = "group",
    loc_level: int = None,
    seed: int = 42,
    write_queue=None,
    track_embeddings: bool = False,
    **model_params: Dict,
):
    torch.set_float32_matmul_precision("high")
    dataloader, n_locs, n_times = get_dataloader(
        dataset=dataset,
        n_users=n_users,
        discretization_rows=discretization_rows,
        discretization_shape=discretization_shape,
        batch_size=batch_size,
        subsample=subsample,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
    )
    torch.manual_seed(seed)
    loc_embedding_dim = int(loc_embedding_factor * n_hidden)
    time_embedding_dim = int(time_embedding_factor * loc_embedding_dim)
    model = model_cls(
        n_users=n_users,
        n_locs=n_locs[:loc_levels],
        n_times=n_times[:time_levels],
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout=dropout,
        embedding_type=embedding_type,
        embedding_weight_factor=embedding_weight_factor,
        loc_embedding_dim=loc_embedding_dim,
        time_embedding_dim=time_embedding_dim,
        loc_level=loc_level,
        device=device,
        **model_params,
    ).to(device)

    xh, th, llh, uh = (
        deque([], maxlen=history_length),
        deque([], maxlen=history_length),
        deque([], maxlen=history_length),
        torch.empty(0, device=device, dtype=torch.int32),
    )

    optimizer = optimizer_cls(model.parameters(), lr=lr)

    log_info = dict(
        dataset=dataset,
        n_users=n_users,
        loc_levels=loc_levels,
        loc_level=loc_level,
        time_levels=time_levels,
        model_cls=model_cls.__name__,
        optimizer_cls=optimizer_cls.__name__,
        lr=lr,
        n_hidden=n_hidden,
        n_layers=n_layers,
        embedding_type=embedding_type,
        embedding_weight_factor=embedding_weight_factor,
        loc_embedding_factor=loc_embedding_factor,
        time_embedding_factor=time_embedding_factor,
        discretization_rows=discretization_rows,
        discretization_shape=discretization_shape,
        grow_factor=grow_factor,
        aggregation_mode=aggregation_mode,
        dropout=dropout,
        seed=seed,
        **model_params,
    )
    if track_embeddings:
        tracker = JSONTracker(
            save_path=log_path,
            parameters=log_info,
            write_queue=write_queue,
            module=model,
        )
    else:
        tracker = JSONTracker(
            save_path=log_path, parameters=log_info, write_queue=write_queue
        )
    if verbose:
        iterator = tqdm(dataloader, leave=False)
    else:
        iterator = dataloader
    for xc, tc, llc, uc in iterator:
        with torch.inference_mode():
            model.eval()
            logits = model.pred_step(
                xc=xc, tc=tc, llc=llc, xh=list(xh), th=list(th), llh=list(llh), uh=uh
            )
        model.train()
        loss = model.train_step(
            xc=xc, tc=tc, llc=llc, uc=uc, xh=list(xh), th=list(th), llh=list(llh), uh=uh
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tracker.update(logits, uc)

        xh += xc
        th += tc
        llh += llc
        uh = torch.cat([uh[-history_length + 1 :], uc.to(uh.device)])
    tracker.save()


def run_with_kwargs(kwargs):
    try:
        return run(**kwargs)
    except Exception as e:
        print("Error: ", kwargs)
        print(e)
