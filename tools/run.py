from collections import deque
from itertools import product
from typing import Dict, Optional
import torch
from tqdm import tqdm
from torch.optim import Adam
from multiprocessing import Pool

from src.data.tracker import JSONTracker
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


def grid_search(
    dataset: str,
    model_cls: torch.nn.Module,
    n_users: int,
    loc_levels: 1,
    time_levels: 1,
    optimizer_cls: torch.optim.Optimizer = Adam,
    lr: float = 1e-3,
    n_hidden: int = 128,
    n_layers: int = 1,
    embedding_type: str = "lookup_sum",
    loc_embedding_factor: float = 1.0,
    time_embedding_factor: float = 0.25,
    dropout: float = 0.0,
    subsample: Optional[int] = None,
    discretization_rows: int = 100,
    discretization_shape: str = "hex",
    aggregation_mode: str = "group",
    grow_factor: int = 2,
    device: torch.device = "cuda:0",
    seed: int = 42,
    log_path: str = "test.jsonl",
    debug: bool = False,
    **model_kwargs,
):
    batch_size = 1
    configs = get_config_grid(
        model_cls=model_cls,
        loc_levels=loc_levels,
        time_levels=time_levels,
        optimizer_cls=optimizer_cls,
        lr=lr,
        n_hidden=n_hidden,
        embedding_type=embedding_type,
        loc_embedding_factor=loc_embedding_factor,
        time_embedding_factor=time_embedding_factor,
        dropout=dropout,
        n_layers=n_layers,
        discretization_rows=discretization_rows,
        discretization_shape=discretization_shape,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
        subsample=subsample,
        seed=seed,
        **model_kwargs,
    )
    # Get the dataloader and other dataset-related information
    if debug:
        configs = configs[2:5]
    # Iterate over all combinations and run the model
    for config in tqdm(configs):
        # Run the model with the current combination of parameters
        run(
            dataset=dataset,
            batch_size=batch_size,
            n_users=n_users,
            device=device,
            log_path=log_path,
            verbose=True,
            **config,
        )


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
    embedding_type: str = "lookup_sum",
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
    seed: int = 42,
    write_queue=None,
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
        loc_embedding_dim=loc_embedding_dim,
        time_embedding_dim=time_embedding_dim,
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
        time_levels=time_levels,
        model_cls=model_cls.__name__,
        optimizer_cls=optimizer_cls.__name__,
        lr=lr,
        n_hidden=n_hidden,
        n_layers=n_layers,
        embedding_type=embedding_type,
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
    return run(**kwargs)


def grid_search_parallel(
    dataset: str,
    model_cls: torch.nn.Module,
    n_users: int,
    loc_levels: 1,
    time_levels: 1,
    optimizer_cls: torch.optim.Optimizer = Adam,
    lr: float = 1e-3,
    n_hidden: int = 128,
    n_layers: int = 1,
    embedding_type: str = "lookup_sum",
    loc_embedding_factor: float = 1.0,
    time_embedding_factor: float = 0.25,
    dropout: float = 0.0,
    subsample: Optional[int] = None,
    discretization_rows: int = 100,
    discretization_shape: str = "hex",
    aggregation_mode: str = "group",
    grow_factor: int = 2,
    device: torch.device = "cuda:0",
    seed: int = 42,
    log_path: str = "test.jsonl",
    debug: bool = False,
    num_workers: int = 6,  # Number of workers for parallel processing
    **model_kwargs,
):
    batch_size = 1
    configs = get_config_grid(
        dataset_name=dataset,
        batch_size=batch_size,
        n_users=n_users,
        device=device,
        log_path=log_path,
        verbose=False,
        model_cls=model_cls,
        loc_levels=loc_levels,
        time_levels=time_levels,
        optimizer_cls=optimizer_cls,
        lr=lr,
        n_hidden=n_hidden,
        embedding_type=embedding_type,
        loc_embedding_factor=loc_embedding_factor,
        time_embedding_factor=time_embedding_factor,
        dropout=dropout,
        n_layers=n_layers,
        discretization_rows=discretization_rows,
        discretization_shape=discretization_shape,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
        subsample=subsample,
        seed=seed,
        **model_kwargs,
    )

    if debug:
        configs = configs[2:5]

    # Use multiprocessing Pool for parallel execution
    with Pool(processes=num_workers) as pool:
        # Use tqdm for progress bar
        list(tqdm(pool.imap(run_with_kwargs, configs), total=len(configs)))
