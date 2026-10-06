import random
from typing import List, Optional, Tuple
from src.data.discretization import discretize_coordinates, group_cells
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from functools import cache


BASEPATH = Path(__file__).parent.parent.parent


def get_dataloader(
    dataset: str,
    n_users: int,
    discretization_rows: int = 100,
    discretization_shape: str = "hex",
    aggregation_mode: str = "group",
    batch_size: int = 1,
    grow_factor: float = 2,
    subsample: Optional[int] = None,
    subsample_mode: Optional[str] = None,
    subsample_start: int = 0,
    drift_location=None,
    drift_users=None,
    pre_drift_lon_limit=None,
    use_offset: bool = False,
    split=None,
    seed=42,
):
    def collate_fn(batch):
        x = [torch.tensor(row[0]).int() for row in batch]
        t = [torch.tensor(row[1]).int() for row in batch]
        ll = [torch.tensor(row[2]).float() for row in batch]
        u = torch.tensor([row[3] for row in batch]).long()
        return x, t, ll, u

    # Create a dataset and dataloader
    df, x_features, t_features, _ = load_data(
        dataset=dataset,
        n_users=n_users,
    )
    df_quantized = get_discretization(
        df,
        n_rows=discretization_rows,
        shape=discretization_shape,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
        use_offset=use_offset,
    )
    n_locs = [df_quantized[x_feature].nunique() + 1 for x_feature in x_features]
    n_times = [df_quantized[t_feature].nunique() + 1 for t_feature in t_features[:-1]]
    trajectories = get_trajectories(
        df=df_quantized,
        x_features=x_features,
        t_features=t_features,
        subsample=subsample,
        subsample_mode=subsample_mode,
        subsample_start=subsample_start,
    )

    if drift_location is not None:
        if drift_users is not None:
            trajectories = insert_user_drift(
                trajectories,
                drift_location=drift_location,
                n_users=n_users,
                n_users_insert=drift_users,
                seed=42,
            )
        elif pre_drift_lon_limit is not None:
            trajectories = insert_location_drift(
                trajectories,
                drift_location=drift_location,
                limits_pre_drift=[pre_drift_lon_limit, 1.0],
            )

    if split is not None:
        # Random train/test split at the trajectory level
        n_total = len(trajectories)
        n_train = int(n_total * split)
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=rng).tolist()

        train_trajectories = [trajectories[i] for i in perm[:n_train]]
        test_trajectories = [trajectories[i] for i in perm[n_train:]]
        # train_trajectories = trajectories[:n_train]
        # test_trajectories = trajectories[n_train:]

        train_loader = DataLoader(
            TrajectoryDataset(train_trajectories),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            TrajectoryDataset(test_trajectories),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        return train_loader, test_loader, n_locs, n_times

    dataloader = DataLoader(
        TrajectoryDataset(trajectories),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return dataloader, n_locs, n_times


def get_discretization(
    df, n_rows, shape="hex", aggregation_mode="grow", grow_factor=2, use_offset=False
):
    dfd = discretize_coordinates(df, n_rows=n_rows, shape=shape, col_suffix=0)
    dfd["cell0"] = dfd.groupby(["q0", "r0"], sort=False).ngroup() + 1

    agg_levels = range(1, 4)
    if aggregation_mode == "group":
        for level in agg_levels:
            q_new, r_new = group_cells(
                qr=dfd[[f"q{level-1}", f"r{level-1}"]].values, shape=shape
            ).T
            dfd[f"q{level}"] = q_new
            dfd[f"r{level}"] = r_new
            if use_offset:
                offset = dfd[f"cell{level-1}"].max()
            else:
                offset = 0
            dfd[f"cell{level}"] = (
                dfd.groupby([f"q{level}", f"r{level}"], sort=False).ngroup()
                + 1
                + offset
            )
    elif aggregation_mode == "grow":
        for level in agg_levels:
            dfd = discretize_coordinates(
                dfd,
                n_rows=n_rows // (grow_factor**level),
                shape=shape,
                col_suffix=level,
            )
            if use_offset:
                offset = dfd[f"cell{level-1}"].max()
            else:
                offset = 0
            dfd[f"cell{level}"] = (
                dfd.groupby([f"q{level}", f"r{level}"], sort=False).ngroup()
                + 1
                + offset
            )
    return dfd


def merge_random(a, b):
    result = []
    i = j = 0
    total_len = len(a) + len(b)
    slots = list(range(total_len))
    random.shuffle(slots)

    # pick random positions for a
    a_positions = set(slots[: len(a)])

    for k in range(total_len):
        if k in a_positions:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    return result


def insert_user_drift(trajectories, drift_location, n_users, n_users_insert, seed):
    np.random.seed(seed)
    u_insert = np.random.choice(np.arange(n_users), size=n_users_insert)
    ts_base = []
    ts_insert = []
    for ti in trajectories:
        if ti[3] in (u_insert):
            ts_insert.append(ti)
        else:
            ts_base.append(ti)

    ts_base_post = ts_base[drift_location:]
    ts_pre = ts_base[:drift_location]
    ts_post = merge_random(ts_base_post, ts_insert)
    return ts_pre + ts_post


def insert_location_drift(trajectories, drift_location, limits_pre_drift=[0.5, 1.0]):
    from copy import copy

    trajectories_pre = list(
        filter(
            lambda x: (
                (x[2][:, 0] < limits_pre_drift[0]) & (x[2][:, 1] < limits_pre_drift[1])
            ).sum()
            > 0,
            trajectories,
        )
    )
    trajectories_pre = copy(trajectories_pre)

    for i, ti in enumerate(trajectories_pre[:drift_location]):
        x, t, ll, u = ti
        mask = (ll[:, 0] < limits_pre_drift[0]) & (ll[:, 1] < limits_pre_drift[1])
        trajectories_pre[i] = (x[mask], t[mask], ll[mask], u)
    return trajectories_pre + trajectories[drift_location:]


def get_dest_pred_dataloader(
    dataset: str,
    n_users: int,
    discretization_rows: int = 100,
    discretization_shape: str = "hex",
    aggregation_mode: str = "group",
    batch_size: int = 1,
    grow_factor: float = 2,
    subsample: Optional[int] = None,
    subsample_mode: str = "random",
    use_offset: bool = False,
    min_subtraj_len: int = 5,
):
    # Create a dataset and dataloader
    df, x_features, t_features, llscale = load_data(
        dataset=dataset,
        n_users=n_users,
    )
    df_quantized = get_discretization(
        df,
        n_rows=discretization_rows,
        shape=discretization_shape,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
        use_offset=use_offset,
    )
    n_locs = [df_quantized[x_feature].nunique() + 1 for x_feature in x_features]
    n_times = [df_quantized[t_feature].nunique() + 1 for t_feature in t_features[:-1]]
    trajectories = get_trajectories(df_quantized, x_features, t_features)
    subtrajectories = get_subtrajectories(trajectories, min_subtraj_len)
    if subsample_mode == "random":
        subsample_idcs = np.random.choice(
            np.arange(len(subtrajectories)), replace=False, size=subsample
        )
        subsample_idcs = np.sort(subsample_idcs)
        subtrajectories = [subtrajectories[idx] for idx in subsample_idcs]
    else:
        subtrajectories = subtrajectories[:subsample]

    def collate_fn(batch):
        x = [torch.tensor(row[0]).int() for row in batch]
        t = [torch.tensor(row[1]).int() for row in batch]
        ll = [torch.tensor(row[2]).float() for row in batch]
        ll_target = torch.tensor(np.array([row[3] for row in batch])).float()
        u = torch.tensor([row[4] for row in batch]).int()
        return x, t, ll, u, ll_target

    dataloader = DataLoader(
        TrajectoryDataset(subtrajectories),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return dataloader, n_locs, n_times, llscale


def get_subtrajectories(trajectories, min_length=5):
    """
    Transforms a list of complete trajectories into a chronologically sorted
    stream of (history, target) samples for next location prediction.

    Args:
        trajectories: The output from the get_trajectories function.

    Returns:
        A list of lists, where each list represents one
        training sample, sorted by the timestamp of the target point.
    """
    subtrajectories = []

    # Pre-allocate and use numpy for slicing where possible
    for xs, ts, lls, user in trajectories:
        # lls = np.diff(np.vstack(([lls[0]], lls)), axis=0)
        n = len(xs)
        if n < min_length:
            continue
        # Use numpy slicing for efficiency
        for i in range(min_length - 1, n):
            subtrajectories.append([xs[:i], ts[:i], lls[:i], lls[-1], user, ts[i][-1]])

    # Use numpy argsort for faster sorting if many samples
    if subtrajectories:
        # Extract timestamps for sorting
        timestamps = np.array([sample[-1] for sample in subtrajectories])
        sorted_indices = np.argsort(timestamps)
        subtrajectories = [subtrajectories[i] for i in sorted_indices]

    return subtrajectories


def get_trajectories(
    df,
    x_features,
    t_features,
    subsample=None,
    remove_duplicates=False,
    subsample_mode="first",
    subsample_start=0,
):
    # Remove consecutive entries with the same x_features[0] within each t_idx group
    def remove_consecutive_duplicates(group):
        # Keep the first row, then only keep rows where x_features[0] changes
        mask = group[x_features[0]].shift() != group[x_features[0]]
        mask.iloc[0] = True  # Always keep the first row
        return group[mask]

    if remove_duplicates:
        df = df.groupby("t_idx", group_keys=False).apply(remove_consecutive_duplicates)
        df = df.sort_values(by=["t_idx", "timestamp"]).reset_index(drop=True)

    # Group trajectories by 't_idx' and determine start indices for splits
    groups = df.groupby("t_idx", sort=False)
    traj_starts = groups["t_idx"].idxmin().values[1:]

    # Extract features and user data as arrays
    df_x = df[x_features].values
    df_t = df[t_features].values
    df_ll = df[["lon", "lat"]].values
    df_u = df["user"].values

    # Split arrays into trajectories using trajectory start indices
    xs = np.split(df_x, traj_starts)
    ts = np.split(df_t, traj_starts)
    lls = np.split(df_ll, traj_starts)
    us = df_u[[0, *traj_starts]]

    # Combine the split arrays into trajectory tuples
    trajectories = list(zip(xs, ts, lls, us))

    # Subsample trajectories if required
    if subsample:
        if subsample_mode == "first":
            trajectories = trajectories[subsample_start : subsample_start + subsample]
        else:
            trajectories = trajectories[subsample_start + subsample :]

    return trajectories


DEFAULT_N_USERS = {
    "geolife": 150,
    "geolife_dp": 150,
}


@cache
def load_data(dataset: str, n_users: Optional[int] = None):
    t_features = ["hour", "3hour", "6hour", "weekday", "is_workday", "timestamp"]
    x_features = [f"cell{i}" for i in range(4)]
    if n_users is None:
        n_users = DEFAULT_N_USERS.get(dataset, 800)

    # load and preprocess the data
    df = pd.read_csv(
        BASEPATH.joinpath("data", "processed", f"{dataset}_{n_users}.csv.gz")
    )
    if dataset.startswith("foursquare"):
        df["point"] = df.groupby("venueId").ngroup() + 1
        x_features = ["point"] + x_features

    df["3hour"] = df["hour"] // 3
    df["6hour"] = df["hour"] // 6
    min_lon, max_lon = df["lon"].min(), df["lon"].max()
    min_lat, max_lat = df["lat"].min(), df["lat"].max()
    lon_scale = max_lon - min_lon
    lat_scale = max_lat - min_lat
    df["lon"] = (df["lon"] - min_lon) / lon_scale
    df["lat"] = (df["lat"] - min_lat) / lat_scale

    return df, x_features, t_features, np.array([lon_scale, lat_scale])


class TrajectoryDataset(Dataset):
    def __init__(self, data: List[Tuple]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
