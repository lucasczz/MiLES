import pandas as pd
import lightning as L
import torch
from torch.utils import data
import pickle
import pathlib

DATAPATH = pathlib.Path(__file__).parent.parent.parent.joinpath("data", "processed")


class GeolifeModule(L.LightningDataModule):
    def collate_fn(batch):
        x, y, t = zip(*batch)
        return x, torch.tensor(y), t

    def __init__(
        self,
        n_hex_rows=100,
        n_hex_levels=3,
        n_users=90,
        val_split=0.2,
        test_split=0.2,
        batch_size=128,
        n_workers=4,
    ):
        super().__init__()
        self.n_hex_levels = n_hex_levels
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.n_users = n_users
        self.user_weight = []
        self.cell_keys = [f"cell{i}" for i in range(n_hex_levels)]
        self.trajectories = {}
        with open(DATAPATH.joinpath(f"geolife_hex_{n_hex_rows}.pkl"), "rb") as f:
            self.df = pickle.load(f)

        users = self.df.groupby("user")["t_idx"].nunique().nlargest(n_users).index
        self.df = self.df[self.df["user"].isin(users)]

    def setup(self, stage: str) -> None:
        stages = ["train", "val", "test"]
        results = {stage: [] for stage in stages}

        for _, df_user in self.df.groupby("user"):
            trajectory_ids = df_user["t_idx"].unique()

            n_trajectories = len(trajectory_ids)
            n_val = int(self.val_split * n_trajectories)
            n_test = int(self.test_split * n_trajectories)
            n_train = n_trajectories - n_val - n_test

            df_user["user"] = len(results["train"])
            self.user_weight.append(n_train / self.df["t_idx"].nunique())

            results["train"].append(
                df_user[df_user["t_idx"].isin(trajectory_ids[:n_train])]
            )
            results["val"].append(
                df_user[
                    df_user["t_idx"].isin(trajectory_ids[n_train : n_train + n_val])
                ]
            )
            results["test"].append(
                df_user[df_user["t_idx"].isin(trajectory_ids[n_train + n_val :])]
            )

        self.n_users = len(results["train"])

        for stage in stages:
            data_stage = pd.concat(results[stage])
            trajs = data_stage.groupby("t_idx")
            x = [torch.tensor(traj[self.cell_keys].values) for _, traj in trajs]
            y = torch.tensor([traj.iloc[0]["user"] for _, traj in trajs])
            t = [torch.tensor(traj["time_label"].values) for _, traj in trajs]
            self.trajectories[stage] = list(zip(x, y, t))

    def get_info(self):
        return {
            "n_users": self.n_users,
            "n_cells": list(self.df[self.cell_keys].nunique()),
            "n_time_intervals": self.df["time_label"].nunique(),
            "user_weight": self.user_weight,
        }

    def train_dataloader(self):
        return data.DataLoader(
            self.trajectories["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=GeolifeModule.collate_fn,
            num_workers=self.n_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.trajectories["val"],
            batch_size=self.batch_size,
            collate_fn=GeolifeModule.collate_fn,
            num_workers=self.n_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.trajectories["test"],
            batch_size=self.batch_size,
            collate_fn=GeolifeModule.collate_fn,
            num_workers=self.n_workers,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.trajectories["test"],
            batch_size=self.batch_size,
            collate_fn=GeolifeModule.collate_fn,
            num_workers=self.n_workers,
        )
