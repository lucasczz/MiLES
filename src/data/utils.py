from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd


BASEPATH = Path(__file__).parent.parent.parent


def get_dataloader(
    dataset: str,
    loc_levels: List[int],
    time_levels: List[int],
    n_users: int,
    batch_size: int,
    device: torch.device,
    debug: bool = False,
):
    def collate_fn(batch):
        x = [torch.tensor(row[0]).int().to(device) for row in batch]
        t = [torch.tensor(row[1]).int().to(device) for row in batch]
        ll = [torch.tensor(row[2]).float().to(device) for row in batch]
        u = torch.tensor([row[3] for row in batch]).long().to(device)
        return x, t, ll, u

    # Create a dataset and dataloader
    print("Loading data...")
    trajectories, n_locs, n_times = load_trajectories(
        dataset,
        n_users,
        loc_levels=loc_levels,
        time_levels=time_levels,
        subsample=10000 if debug else None,
    )
    dataloader = DataLoader(
        TrajectoryDataset(trajectories),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return dataloader, n_locs, n_times


def load_trajectories(
    dataset: str, n_users: int, loc_levels: int, time_levels: int, subsample: int = None
):
    t_features = ["hour", "weekday", "is_workday"][:time_levels] + ["timestamp"]
    x_features = [f"cell{i}" for i in range(loc_levels)]

    # load and preprocess the data
    df = pd.read_csv(
        BASEPATH.joinpath("data", "processed", f"{dataset}_{n_users}.csv.gz")
    )
    if dataset.startswith("foursquare"):
        df["point"] = df.groupby("venueId").ngroup() + 1
        x_features = ["point"] + x_features
    if subsample:
        df = df.iloc[:subsample]

    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["lon"] = (df["lon"] - df["lon"].min()) / (df["lon"].max() - df["lon"].min())
    df["lat"] = (df["lat"] - df["lat"].min()) / (df["lat"].max() - df["lat"].min())

    grouped = df.groupby("t_idx", sort=False)
    data = [
        (
            group[x_features].values,
            group[t_features].values,
            group[["lon", "lat"]].values,
            group["user"].iloc[0],
        )
        for _, group in grouped
    ]
    n_locs = [df[x_feature].max() + 1 for x_feature in x_features]
    n_times = [df[t_feature].max() + 1 for t_feature in t_features[:-1]]
    return data, n_locs, n_times


class TrajectoryDataset(Dataset):
    def __init__(self, data: List[Tuple]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
