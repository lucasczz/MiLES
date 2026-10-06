import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from pathlib import Path
import numpy as np
import traceback
import pathlib
from collections import deque
from itertools import product
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
from dog import PolynomialDecayAverager
from torch.optim import Adam
from tqdm import tqdm

from src.data.tracker import JSONTracker, handle_queue
from src.data.utils import get_dataloader, get_dest_pred_dataloader
from src.data.buffer import RecentBuffer, BalancedBuffer, RandomBuffer

import argparse


def setup_device_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--workers-per-device", type=int, default=2)
    parser.add_argument("--dp", type=bool, default=False)
    args = parser.parse_args()

    # Priority: CLI args > auto-detect
    if args.gpus is not None:
        devices = [f"cuda:{i}" for i in args.gpus]
    else:
        devices = auto_detect_devices()

    num_workers = len(devices) * args.workers_per_device

    print(f"Using devices: {devices}")
    print(f"Using {num_workers} workers")
    return devices, num_workers, args.dp


def auto_detect_devices():
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


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
    df_planned["optimizer_cls"] = df_planned["optimizer_cls"].apply(
        lambda x: x.__name__
    )
    df_done = pd.read_json(path_done, orient="records", lines=True)

    # Match on specific columns
    diff_df = df_planned.merge(df_done, on=relevant_params, how="left", indicator=True)
    missing_mask = diff_df["_merge"] == "left_only"
    result = [config for config, is_missing in zip(configs, missing_mask) if is_missing]
    # Convert back to list of configs using boolean indexing
    return result


def run_configs(
    configs: List[Dict],
    devices: List[str],
    num_workers: int,
    path: pathlib.Path,
    task: str = "TUL",
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
    if task == "TUL":
        list(tqdm(pool.imap(run_with_kwargs, configs), total=len(configs)))
    elif task == "batch":
        list(tqdm(pool.imap(run_batch_with_kwargs, configs), total=len(configs)))
    elif task == "retrain":
        list(tqdm(pool.imap(run_retrain_with_kwargs, configs), total=len(configs)))
    else:
        list(tqdm(pool.imap(run_next_loc_with_kwargs, configs), total=len(configs)))

    q.put("kill")
    pool.close()
    pool.join()


def _setup_replay_buffer(buffer_type: str, buffer_size: int):
    """Initialize replay buffer based on type and size."""
    if buffer_size <= 0:
        return None

    buffer_map = {
        "balanced": BalancedBuffer,
        "recent": RecentBuffer,
        "random": RandomBuffer,
    }

    if buffer_type not in buffer_map:
        return None

    return buffer_map[buffer_type](buffer_size)


def _setup_model_and_optimizer(
    model_cls: torch.nn.Module,
    optimizer_cls: torch.optim.Optimizer,
    lr: float,
    n_users: int,
    n_locs: list,
    n_times: list,
    n_hidden: int,
    loc_embedding_factor: float,
    time_embedding_factor: float,
    user_embedding_factor: float = None,
    day_embedding_factor: float = None,
    embedding_lr_multiplier: float = 1.0,
    loc_lr_multipliers: Optional[List[float]] = None,
    build_loc_lr_groups: bool = False,
    fixed_loc_weights: Optional[List[float]] = None,
    device: torch.device = "cuda:0",
    **model_params,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Any]:
    """Setup model, optimizer, and averager."""

    # Calculate embedding dimensions
    loc_embedding_dim = int(loc_embedding_factor * n_hidden)
    time_embedding_dim = int(time_embedding_factor * loc_embedding_dim)

    model_kwargs = {
        "n_users": n_users,
        "n_locs": n_locs,
        "n_times": n_times,
        "n_hidden": n_hidden,
        "loc_embedding_dim": loc_embedding_dim,
        "time_embedding_dim": time_embedding_dim,
        "device": device,
        **model_params,
    }

    # Add user and day embeddings if provided (for destination prediction task)
    if user_embedding_factor is not None:
        model_kwargs["user_embedding_dim"] = int(
            user_embedding_factor * loc_embedding_dim
        )
    if day_embedding_factor is not None:
        model_kwargs["day_embedding_dim"] = int(
            day_embedding_factor * loc_embedding_dim
        )

    model = model_cls(**model_kwargs).to(device)
    # model = torch.compile(model, mode="default", fullgraph=True)

    # Optionally freeze the location level weights at fixed values, turning the
    # learnable per-level gain into a fixed forward gain. Used to test whether a
    # non-learnable per-level gain reproduces the effect of the learnable weights.
    if fixed_loc_weights is not None:
        weights = model.embedding.weights
        if len(fixed_loc_weights) != weights.numel():
            raise ValueError(
                f"Expected {weights.numel()} fixed_loc_weights, got "
                f"{len(fixed_loc_weights)}."
            )
        with torch.no_grad():
            weights.copy_(
                torch.tensor(fixed_loc_weights, dtype=weights.dtype, device=weights.device)
            )
        weights.requires_grad_(False)

    # Optionally assign per-level learning rates to the location embedding tables.
    # This tests whether MiLES's learnable level weights w_l act through the
    # optimization (an effective per-level learning rate) rather than adding
    # representational capacity. loc_lr_multipliers sets a static multiplier per
    # level; embedding_lr_multiplier applies a single multiplier to all levels;
    # build_loc_lr_groups creates per-level groups whose rates the caller updates
    # over time (a per-step schedule). Per-level groups are param_groups[:n_levels]
    # in level order, followed by a group with the remaining parameters.
    use_level_groups = (
        loc_lr_multipliers is not None
        or embedding_lr_multiplier != 1.0
        or build_loc_lr_groups
    )
    if use_level_groups:
        loc_embeddings = model.embedding.loc_embedding
        if not isinstance(loc_embeddings, torch.nn.ModuleList):
            raise ValueError(
                "Per-level learning rates require a multi-level location embedding "
                "(loc_embedding must be a ModuleList)."
            )
        if loc_lr_multipliers is not None:
            multipliers = list(loc_lr_multipliers)
            if len(multipliers) != len(loc_embeddings):
                raise ValueError(
                    f"Expected {len(loc_embeddings)} loc_lr_multipliers, "
                    f"got {len(multipliers)}."
                )
        else:
            multipliers = [embedding_lr_multiplier] * len(loc_embeddings)
        level_param_ids = set()
        param_groups = []
        for emb, mult in zip(loc_embeddings, multipliers):
            params = [p for p in emb.parameters() if p.requires_grad]
            level_param_ids.update(id(p) for p in params)
            param_groups.append({"params": params, "lr": lr * mult})
        rest = [
            p
            for p in model.parameters()
            if id(p) not in level_param_ids and p.requires_grad
        ]
        param_groups.append({"params": rest, "lr": lr})
        optimizer = optimizer_cls(param_groups, lr=lr)
    else:
        optimizer = optimizer_cls(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
    averager = (
        PolynomialDecayAverager(model) if optimizer_cls.__name__ == "DoG" else None
    )

    return model, optimizer, averager


def _prepare_batch_with_replay(
    current_batch: tuple,
    replay_buffer,
    replay_batch_size: int,
    replay_interval: int,
    step_counter: int,
    is_next_loc: bool = False,
):
    """Prepare training batch with optional replay data."""
    if is_next_loc:
        xc, tc, llc, uc, ll_target = current_batch
    else:
        xc, tc, llc, uc = current_batch
        ll_target = None

    # Check if replay should be used
    if (
        replay_buffer is not None
        and len(replay_buffer) > 0
        and step_counter % replay_interval == 0
    ):
        replay_data = replay_buffer.sample(replay_batch_size)

        if is_next_loc:
            x_rep, t_rep, ll_rep, u_rep, target_rep = replay_data
            return (
                xc + x_rep,
                tc + t_rep,
                llc + ll_rep,
                torch.cat((uc, u_rep), dim=0),
                torch.cat((ll_target, target_rep), dim=0),
            )
        else:
            x_rep, t_rep, ll_rep, u_rep = replay_data
            return (xc + x_rep, tc + t_rep, llc + ll_rep, torch.cat((uc, u_rep), dim=0))

    return current_batch


def _create_log_info(locals_dict: dict, model_params: dict) -> dict:
    """Create logging information dictionary."""
    base_keys = {
        "dataset",
        "n_users",
        "loc_levels",
        "loc_level",
        "time_levels",
        "lr",
        "n_hidden",
        "n_layers",
        "embedding_type",
        "embedding_weight_factor",
        "loc_embedding_factor",
        "time_embedding_factor",
        "discretization_rows",
        "discretization_shape",
        "grow_factor",
        "aggregation_mode",
        "dropout",
        "seed",
        "replay_batch_size",
        "replay_buffer_size",
        "replay_buffer_type",
        "replay_interval",
        "drift_location",
        "drift_users",
        "pre_drift_lon_limit",
    }

    # Add task-specific keys
    if "user_embedding_factor" in locals_dict:
        base_keys.update({"user_embedding_factor", "day_embedding_factor"})

    log_info = {k: v for k, v in locals_dict.items() if k in base_keys}

    # Add class names
    log_info["model_cls"] = locals_dict["model_cls"].__name__
    log_info["optimizer_cls"] = locals_dict["optimizer_cls"].__name__

    return {**log_info, **model_params}


def run(
    dataset: str,
    model_cls: torch.nn.Module,
    n_users: int = None,
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
    subsample_mode: str = "first",
    subsample_start: int = 0,
    discretization_rows: int = 100,
    discretization_shape: str = "hex",
    grow_factor: int = 2,
    aggregation_mode: str = "grow",
    loc_level: int = None,
    seed: int = 42,
    write_queue=None,
    track_embeddings: bool = False,
    log_emb_norms: bool = False,
    log_input_weight_norms: bool = False,
    loc_lr_schedule=None,
    fixed_loc_weights=None,
    replay_buffer_type: str = "none",
    replay_buffer_size: int = 0,
    replay_batch_size: int = 16,
    replay_interval: int = 1,
    drift_location=None,
    drift_users=None,
    pre_drift_lon_limit=None,
    **model_params: Dict,
):
    """Run prequential evaluation for trajectory-user-linking."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    # Setup data
    dataloader, n_locs, n_times = get_dataloader(
        dataset=dataset,
        n_users=n_users,
        discretization_rows=discretization_rows,
        discretization_shape=discretization_shape,
        batch_size=batch_size,
        subsample=subsample,
        subsample_mode=subsample_mode,
        subsample_start=subsample_start,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
        drift_location=drift_location,
        drift_users=drift_users,
        pre_drift_lon_limit=pre_drift_lon_limit,
    )

    # Setup replay buffer
    replay_buffer = _setup_replay_buffer(replay_buffer_type, replay_buffer_size)

    # Setup model and optimizer
    model, optimizer, averager = _setup_model_and_optimizer(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        lr=lr,
        n_users=n_users,
        n_locs=n_locs[:loc_levels],
        n_times=n_times[:time_levels],
        n_hidden=n_hidden,
        loc_embedding_factor=loc_embedding_factor,
        time_embedding_factor=time_embedding_factor,
        device=device,
        n_layers=n_layers,
        dropout=dropout,
        embedding_type=embedding_type,
        embedding_weight_factor=embedding_weight_factor,
        loc_level=loc_level,
        build_loc_lr_groups=(loc_lr_schedule is not None),
        fixed_loc_weights=fixed_loc_weights,
        **model_params,
    )

    # Resample a per-step per-level learning-rate schedule (e.g. a recorded w_l(t)
    # trajectory) to the number of training steps. The first loc_levels parameter
    # groups are the per-level embedding tables, in level order.
    lr_schedule = None
    if loc_lr_schedule is not None:
        sched = np.asarray(loc_lr_schedule, dtype=float)
        if sched.ndim != 2:
            raise ValueError("loc_lr_schedule must be 2D (steps, levels).")
        n_steps = len(dataloader)
        idx = np.linspace(0, sched.shape[0] - 1, n_steps).round().astype(int)
        lr_schedule = sched[idx]

    # Setup history
    xh, th, llh = (deque([], maxlen=history_length) for _ in range(3))
    uh = torch.empty((0,), device=device, dtype=torch.int32)

    # Setup logging
    log_info = _create_log_info(locals(), model_params)
    tracker_kwargs = {
        "save_path": log_path,
        "parameters": log_info,
        "write_queue": write_queue,
    }
    if track_embeddings or log_emb_norms or log_input_weight_norms:
        tracker_kwargs["module"] = model
        tracker_kwargs["log_emb_norms"] = log_emb_norms
        tracker_kwargs["log_input_weight_norms"] = log_input_weight_norms

    tracker = JSONTracker(**tracker_kwargs)
    iterator = tqdm(dataloader, leave=False) if verbose else dataloader

    # Training loop
    step_counter = 0
    for batch in iterator:
        step_counter += 1
        xc, tc, llc, uc = batch

        # Online prediction
        with torch.no_grad():
            model.eval()
            logits = model.pred_step(
                xc=xc, tc=tc, llc=llc, xh=list(xh), th=list(th), llh=list(llh), uh=uh
            )

        # Prepare training batch with replay
        xc_all, tc_all, llc_all, uc_all = _prepare_batch_with_replay(
            batch, replay_buffer, replay_batch_size, replay_interval, step_counter
        )

        # Training step
        model.train()
        loss = model.train_step(
            xc=xc_all,
            tc=tc_all,
            llc=llc_all,
            uc=uc_all,
            xh=list(xh),
            th=list(th),
            llh=list(llh),
            uh=uh,
        )
        loss.backward()
        if lr_schedule is not None:
            mults = lr_schedule[min(step_counter - 1, len(lr_schedule) - 1)]
            for group, mult in zip(optimizer.param_groups, mults):
                group["lr"] = lr * float(mult)
        optimizer.step()
        optimizer.zero_grad()

        # Update tracking and averaging
        tracker.update(logits, uc)
        if averager is not None:
            averager.step()

        # Update history
        xh.extend(xc)
        th.extend(tc)
        llh.extend(llc)
        uh = torch.cat([uh[-history_length + 1 :], uc.to(uh.device)])

        # Add to replay buffer
        if replay_buffer is not None:
            replay_buffer.add(batch)

    tracker.save()


def run_batch(
    dataset: str,
    model_cls: torch.nn.Module,
    n_users: int = None,
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
    batch_size: int = 64,
    subsample: Optional[int] = None,
    subsample_mode="first",
    discretization_rows: int = 100,
    discretization_shape: str = "hex",
    grow_factor: int = 2,
    aggregation_mode: str = "grow",
    loc_level: int = None,
    seed: int = 42,
    write_queue=None,
    track_embeddings: bool = False,
    train_ratio: float = 0.8,
    n_epochs: int = 20,
    drift_location=None,
    drift_users=None,
    pre_drift_lon_limit=None,
    **model_params: Dict,
):
    """Run conventional batch (train/test split) evaluation for trajectory-user-linking."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    # Setup data
    train_loader, test_loader, n_locs, n_times = get_dataloader(
        dataset=dataset,
        n_users=n_users,
        discretization_rows=discretization_rows,
        discretization_shape=discretization_shape,
        batch_size=batch_size,
        subsample=subsample,
        subsample_mode=subsample_mode,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
        split=train_ratio,
        seed=seed,
        drift_location=drift_location,
        drift_users=drift_users,
        pre_drift_lon_limit=pre_drift_lon_limit,
    )

    # Setup model and optimizer
    model, optimizer, averager = _setup_model_and_optimizer(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        lr=lr,
        n_users=n_users,
        n_locs=n_locs[:loc_levels],
        n_times=n_times[:time_levels],
        n_hidden=n_hidden,
        loc_embedding_factor=loc_embedding_factor,
        time_embedding_factor=time_embedding_factor,
        device=device,
        n_layers=n_layers,
        dropout=dropout,
        embedding_type=embedding_type,
        embedding_weight_factor=embedding_weight_factor,
        loc_level=loc_level,
        **model_params,
    )

    # Setup logging
    log_info = _create_log_info(locals(), model_params)
    tracker_kwargs = {
        "save_path": log_path,
        "parameters": log_info,
        "write_queue": write_queue,
    }
    if track_embeddings:
        tracker_kwargs["module"] = model

    tracker = JSONTracker(**tracker_kwargs)

    # ── Training phase ──
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        iterator = (
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
            if verbose
            else train_loader
        )

        for xc, tc, llc, uc in iterator:
            loss = model.train_step(
                xc=xc,
                tc=tc,
                llc=llc,
                uc=uc,
                xh=[],
                th=[],
                llh=[],
                uh=torch.empty((0,), device=device, dtype=torch.int32),
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        if averager is not None:
            averager.step()

        if verbose:
            print(
                f"Epoch {epoch+1}/{n_epochs} — train loss: {epoch_loss / len(train_loader):.4f}"
            )

    # ── Evaluation phase ──
    model.eval()
    all_logits, all_labels = [], []
    eval_iterator = (
        tqdm(test_loader, desc="Evaluating", leave=False) if verbose else test_loader
    )

    with torch.no_grad():
        for xc, tc, llc, uc in eval_iterator:
            logits = model.pred_step(
                xc=xc,
                tc=tc,
                llc=llc,
                xh=[],
                th=[],
                llh=[],
                uh=torch.empty((0,), device=device, dtype=torch.int32),
            )
            all_logits.append(logits.cpu())
            all_labels.append(uc.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute metrics
    preds = all_logits.argmax(axis=-1)
    top_5 = np.array(
        [
            int(label in np.argsort(logit)[-5:])
            for logit, label in zip(all_logits, all_labels)
        ]
    )

    f1 = f1_score(y_true=all_labels, y_pred=preds, average="macro", zero_division=0)
    results = {
        **_create_log_info(locals(), model_params),
        "top_1_accuracy": float(np.mean(preds == all_labels)),
        "top_5_accuracy": float(np.mean(top_5)),
        "macro_f1": f1,
        "n_train": len(train_loader.dataset),
        "n_test": len(all_labels),
    }

    if verbose:
        print(
            f"Accuracy: {results['accuracy']:.4f} | Top-5: {results['top_5_accuracy']:.4f}"
        )

    # Save
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        json.dump(results, f)
        f.write("\n")


def run_periodic_retrain(
    dataset: str,
    model_cls: torch.nn.Module,
    n_users: int = None,
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
    subsample: Optional[int] = None,
    subsample_mode: str = "first",
    discretization_rows: int = 100,
    discretization_shape: str = "hex",
    grow_factor: int = 2,
    aggregation_mode: str = "grow",
    loc_level: int = None,
    seed: int = 42,
    write_queue=None,
    track_embeddings: bool = False,
    retrain_interval: int = 1000,
    retrain_epochs: int = 10,
    retrain_batch_size: int = 64,
    drift_location=None,
    drift_users=None,
    pre_drift_lon_limit=None,
    **model_params: Dict,
):
    """Run prequential evaluation with periodic batch retraining."""
    torch.set_float32_matmul_precision("high")

    # Setup data — stream with batch_size=1 for prequential evaluation
    dataloader, n_locs, n_times = get_dataloader(
        dataset=dataset,
        n_users=n_users,
        discretization_rows=discretization_rows,
        discretization_shape=discretization_shape,
        batch_size=1,
        subsample=subsample,
        subsample_mode=subsample_mode,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
        drift_location=drift_location,
        drift_users=drift_users,
        pre_drift_lon_limit=pre_drift_lon_limit,
    )

    # Store model init args for reinitialisation at each retrain
    model_init_kwargs = dict(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        lr=lr,
        n_users=n_users,
        n_locs=n_locs[:loc_levels],
        n_times=n_times[:time_levels],
        n_hidden=n_hidden,
        loc_embedding_factor=loc_embedding_factor,
        time_embedding_factor=time_embedding_factor,
        device=device,
        n_layers=n_layers,
        dropout=dropout,
        embedding_type=embedding_type,
        embedding_weight_factor=embedding_weight_factor,
        loc_level=loc_level,
        **model_params,
    )

    # Initial model (untrained)
    torch.manual_seed(seed)
    model, optimizer, averager = _setup_model_and_optimizer(**model_init_kwargs)

    # Setup history (for models that use it during prediction)
    xh, th, llh = (deque([], maxlen=history_length) for _ in range(3))
    uh = torch.empty((0,), device=device, dtype=torch.int32)

    # Accumulate training data
    all_xc, all_tc, all_llc, all_uc = [], [], [], []

    # Setup logging
    log_info = _create_log_info(locals(), model_params)
    tracker_kwargs = {
        "save_path": log_path,
        "parameters": log_info,
        "write_queue": write_queue,
    }
    if track_embeddings:
        tracker_kwargs["module"] = model

    tracker = JSONTracker(**tracker_kwargs)
    iterator = tqdm(dataloader, leave=False) if verbose else dataloader

    # Training loop
    stream_data = list(dataloader)  # list of (xc, tc, llc, uc) single instances
    n_total = len(stream_data)

    # Unwrap single-instance batches for accumulation
    all_xc = [xc[0] for xc, _, _, _ in stream_data]
    all_tc = [tc[0] for _, tc, _, _ in stream_data]
    all_llc = [llc[0] for _, _, llc, _ in stream_data]
    all_uc = [uc[0] for _, _, _, uc in stream_data]

    eval_batch_size = 64

    for segment_start in range(0, n_total, retrain_interval):
        segment_end = min(segment_start + retrain_interval, n_total)
        segment = list(zip(
            all_xc[segment_start:segment_end],
            all_tc[segment_start:segment_end],
            all_llc[segment_start:segment_end],
            all_uc[segment_start:segment_end],
        ))

        # Batched prequential prediction
        eval_loader = DataLoader(
            segment,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=_retrain_collate_fn,
        )

        model.eval()
        with torch.no_grad():
            for xc_b, tc_b, llc_b, uc_b in eval_loader:
                logits = model.pred_step(
                    xc=xc_b, tc=tc_b, llc=llc_b,
                    xh=[], th=[], llh=[],
                    uh=torch.empty((0,), device=device, dtype=torch.int32),
                )
                # Track per-instance
                for i in range(len(uc_b)):
                    tracker.update(
                        logits[i:i+1], uc_b[i:i+1]
                    )

        # Retrain on all data seen so far
        if segment_end < n_total:
            retrain_data = list(zip(
                all_xc[:segment_end],
                all_tc[:segment_end],
                all_llc[:segment_end],
                all_uc[:segment_end],
            ))

            torch.manual_seed(seed)
            model, optimizer, averager = _setup_model_and_optimizer(
                **model_init_kwargs
            )

            retrain_loader = DataLoader(
                retrain_data,
                batch_size=retrain_batch_size,
                shuffle=True,
                collate_fn=_retrain_collate_fn,
            )

            model.train()
            for epoch in range(retrain_epochs):
                for xc_b, tc_b, llc_b, uc_b in retrain_loader:
                    loss = model.train_step(
                        xc=xc_b, tc=tc_b, llc=llc_b, uc=uc_b,
                        xh=[], th=[], llh=[],
                        uh=torch.empty(
                            (0,), device=device, dtype=torch.int32
                        ),
                    )
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            if averager is not None:
                averager.step()

            if verbose:
                print(
                    f"Retrained at {segment_end} on "
                    f"{len(retrain_data)} trajectories"
                )

    tracker.save()

def _retrain_collate_fn(batch):
    xc = [item[0] for item in batch]
    tc = [item[1] for item in batch]
    llc = [item[2] for item in batch]
    uc = torch.stack([item[3] for item in batch])
    return xc, tc, llc, uc


def run_next_loc(
    dataset: str,
    model_cls: torch.nn.Module,
    n_users: int = None,
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
    user_embedding_factor: float = 0.25,
    day_embedding_factor: float = 0.25,
    dropout: float = 0.0,
    device: torch.device = "cuda:0",
    log_path: str = "test.jsonl",
    verbose: bool = True,
    batch_size: int = 1,
    subsample: Optional[int] = None,
    subsample_mode: str = "random",
    min_subtraj_len: int = 2,
    discretization_rows: int = 100,
    discretization_shape: str = "hex",
    grow_factor: int = 2,
    aggregation_mode: str = "grow",
    loc_level: int = None,
    seed: int = 42,
    write_queue=None,
    replay_buffer_type: str = "none",
    replay_buffer_size: int = 0,
    replay_batch_size: int = 16,
    replay_interval: int = 1,
    **model_params: Dict,
):
    """Run prequential evaluation for next-location prediction."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    # Setup data
    dataloader, n_locs, n_times, ll_scale = get_dest_pred_dataloader(
        dataset=dataset,
        n_users=n_users,
        discretization_rows=discretization_rows,
        discretization_shape=discretization_shape,
        batch_size=batch_size,
        subsample=subsample,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
        min_subtraj_len=min_subtraj_len,
        subsample_mode=subsample_mode,
    )

    # Setup replay buffer
    replay_buffer = _setup_replay_buffer(replay_buffer_type, replay_buffer_size)

    # Setup model and optimizer
    model, optimizer, _ = _setup_model_and_optimizer(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        lr=lr,
        n_users=n_users,
        n_locs=n_locs[:loc_levels],
        n_times=n_times[:time_levels],
        n_hidden=n_hidden,
        loc_embedding_factor=loc_embedding_factor,
        time_embedding_factor=time_embedding_factor,
        user_embedding_factor=user_embedding_factor,
        day_embedding_factor=day_embedding_factor,
        device=device,
        n_layers=n_layers,
        dropout=dropout,
        embedding_type=embedding_type,
        embedding_weight_factor=embedding_weight_factor,
        loc_level=loc_level,
        **model_params,
    )

    # Setup logging
    log_info = _create_log_info(locals(), model_params)
    tracker = JSONTracker(
        save_path=log_path,
        parameters=log_info,
        write_queue=write_queue,
        ll_scale=ll_scale,
    )
    iterator = tqdm(dataloader, leave=False) if verbose else dataloader

    # Training loop
    step_counter = 0
    for batch in iterator:
        step_counter += 1
        xc, tc, llc, uc, ll_target = batch

        # Online prediction
        with torch.no_grad():
            model.eval()
            preds = model.pred_step(xc=xc, tc=tc, llc=llc, uc=uc)

        # Prepare training batch with replay
        xc_all, tc_all, llc_all, uc_all, target_all = _prepare_batch_with_replay(
            batch,
            replay_buffer,
            replay_batch_size,
            replay_interval,
            step_counter,
            is_next_loc=True,
        )

        # Training step
        model.train()
        loss = model.train_step(
            xc=xc_all, tc=tc_all, llc=llc_all, uc=uc_all, ll_target=target_all
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update tracking
        tracker.update_next_loc(preds, ll_target)

        # Add to replay buffer
        if replay_buffer is not None:
            replay_buffer.add(batch)
    tracker.save()


def run_with_kwargs(kwargs):
    try:
        return run(**kwargs)

    except Exception as e:
        print("Error: ", kwargs)
        print(traceback.format_exc())
        print(e)


def run_retrain_with_kwargs(kwargs):
    try:
        return run_periodic_retrain(**kwargs)

    except Exception as e:
        print("Error: ", kwargs)
        print(traceback.format_exc())
        print(e)


def run_batch_with_kwargs(kwargs):
    try:
        return run_batch(**kwargs)

    except Exception as e:
        print("Error: ", kwargs)
        print(traceback.format_exc())
        print(e)


def run_next_loc_with_kwargs(kwargs):
    try:
        return run_next_loc(**kwargs)

    except Exception as e:
        print("Error: ", kwargs)
        print(traceback.format_exc())
        print(e)
