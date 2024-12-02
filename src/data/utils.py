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
    discretization_shape: str = 'hex',
    aggregation_mode: str = "group",
    batch_size: int = 1,
    grow_factor: float = 2,
    subsample: Optional[int] = None,
):
    def collate_fn(batch):
        x = [torch.tensor(row[0]).int() for row in batch]
        t = [torch.tensor(row[1]).int() for row in batch]
        ll = [torch.tensor(row[2]).float() for row in batch]
        u = torch.tensor([row[3] for row in batch]).long()
        return x, t, ll, u

    # Create a dataset and dataloader
    df, x_features, t_features = load_data(
        dataset=dataset,
        n_users=n_users,
    )
    df_quantized = get_discretization(
        df,
        n_rows=discretization_rows,
        shape=discretization_shape,
        aggregation_mode=aggregation_mode,
        grow_factor=grow_factor,
    )
    n_locs = [df_quantized[x_feature].max() + 1 for x_feature in x_features]
    n_times = [df_quantized[t_feature].max() + 1 for t_feature in t_features[:-1]]
    trajectories = get_trajectories(df_quantized, x_features, t_features, subsample)
    dataloader = DataLoader(
        TrajectoryDataset(trajectories),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return dataloader, n_locs, n_times


def get_discretization(
    df, n_rows, shape="hex", aggregation_mode="group", grow_factor=2
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
            dfd[f"cell{level}"] = (
                dfd.groupby([f"q{level}", f"r{level}"], sort=False).ngroup() + 1
            )
    elif aggregation_mode == "grow":
        for level in agg_levels:
            dfd = discretize_coordinates(
                dfd,
                n_rows=n_rows // (grow_factor**level),
                shape=shape,
                col_suffix=level,
            )
            dfd[f"cell{level}"] = (
                dfd.groupby([f"q{level}", f"r{level}"], sort=False).ngroup() + 1
            )
    return dfd


def get_trajectories(df, x_features, t_features, subsample=None):
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
        trajectories = trajectories[:subsample]

    return trajectories


@cache
def load_data(dataset: str, n_users: int):
    t_features = ["hour", "3hour", "6hour", "weekday", "is_workday", "timestamp"]
    x_features = [f"cell{i}" for i in range(4)]

    # load and preprocess the data
    df = pd.read_csv(
        BASEPATH.joinpath("data", "processed", f"{dataset}_{n_users}.csv.gz")
    )
    if dataset.startswith("foursquare"):
        df["point"] = df.groupby("venueId").ngroup() + 1
        x_features = ["point"] + x_features

    df["3hour"] = df["hour"] // 3
    df["6hour"] = df["hour"] // 6
    df["lon"] = (df["lon"] - df["lon"].min()) / (df["lon"].max() - df["lon"].min())
    df["lat"] = (df["lat"] - df["lat"].min()) / (df["lat"].max() - df["lat"].min())

    return df, x_features, t_features


class TrajectoryDataset(Dataset):
    def __init__(self, data: List[Tuple]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
