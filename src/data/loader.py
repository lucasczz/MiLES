from torch.utils.data import DataLoader, Dataset
from torch import nn
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from src.data.tracker import ExperimentTracker


def micro_f1(labels, preds):
    return f1_score(labels, preds, average="micro")


def micro_recall(labels, preds):
    return recall_score(labels, preds, average="micro")


def micro_precision(labels, preds):
    return precision_score(labels, preds, average="micro")


BASEPATH = Path(__file__).parent.parent.parent
METRICS = [micro_f1, micro_precision, micro_recall, accuracy_score]
LOGGING_INTERVAL = 100


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        n_users: int = 400,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.n_users = n_users
        self.batch_size = batch_size
        self.prep_data()

    def prep_data(self):
        t_features = ["hour", "weekday", "timestamp", "is_workday"]
        x_features = [f"cell{i}" for i in range(4)]
        if self.dataset.startswith("foursquare"):
            x_features = ["POI"] + x_features

        # Load and preprocess the data
        df = pd.read_csv(
            BASEPATH.joinpath(
                "data", "processed", f"{self.dataset}_{self.n_users}.csv.gz"
            )
        )
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()
        df["lon"] = (df["lon"] - df["lon"].min()) / (df["lon"].max() - df["lon"].min())
        df["lat"] = (df["lat"] - df["lat"].min()) / (df["lat"].max() - df["lat"].min())
        data = []

        # Group the data by trajectory index and extract features
        for _, trajectory in df.groupby("t_idx"):
            row = (
                trajectory[x_features].values,
                trajectory[t_features].values,
                trajectory[["lon", "lat"]].values,
                trajectory["user"].iloc[0],
            )
            data.append(row)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(
    dataset: str,
    n_users: int = 400,
    batch_size: int = 1,
):
    # Create a dataset and dataloader
    dataset = TrajectoryDataset(dataset, n_users, batch_size)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )


def collate_fn(data):
    x, t, ll, u = data
    return (
        torch.tensor(x).int(),
        torch.tensor(t).int(),
        torch.tensor(ll).float(),
        torch.tensor(u).int(),
    )


def train(
    model_cls: torch.nn.Module,
    dataset: str,
    buffer_cls: callable,
    buffer_size: int,
    replay_sample_size: int,
    optimizer_cls: torch.optim.optimizer.Optimizer,
    learning_rate: float,
    model_params: dict,
    device: torch.device,
    log_path: str,
):
    dataloader = get_dataloader(dataset)
    model = model_cls(model_params).to(device)
    buffer = buffer_cls(size=buffer_size, sample_size=replay_sample_size)
    tracker = ExperimentTracker(
        metric_fns=METRICS,
        parameters=model_params,
        logging_interval=LOGGING_INTERVAL,
        save_path=BASEPATH.joinpath("reports", log_path),
    )
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)
    for sample in dataloader:
        buffer.update(sample)
        xc, tc, llc, uc, xh, th, llh, uh = buffer.get()
        with torch.inference_mode():
            preds = model.predict_step(xc, tc, llc, uc, xh, th, llh, uh)

        loss = model.train_step(xc, tc, llc, uc, xh, th, llh, uh)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tracker.update(uc[0], preds[0])
