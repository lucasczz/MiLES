from pathlib import Path
from typing import Dict, List
import torch
from torch.optim import Adam
from tqdm import tqdm

from src.data.utils import get_dataloader
from src.data.tracker import ExperimentTracker
from src.models.tuler import BiTULER


BASEPATH = Path(__file__).parent.parent
LOGGING_INTERVAL = 100

import itertools


def search():
    model_cls = BiTULER
    dataset = "foursquare_NYC"
    n_users = 400
    loc_levels = 2
    time_levels = 1
    batch_size = 1
    device = "cuda:0"
    log_path = "tuler_gridsearch.csv"
    optimizer_cls = Adam
    learning_rate = [1e-3 * 2**i for i in range(6)]
    model_params = dict(
        n_hidden=[64, 128, 256],
        embedding_type=["lookup", "rotary", "cosine"],
        loc_embedding_dim=[32, 64],
        time_embedding_dim=8,
        dropout=0.0,
        n_layers=1,
    )

    # Extract the parameters that have multiple values
    n_hidden_values = model_params["n_hidden"]
    embedding_type_values = model_params["embedding_type"]
    embedding_dim_values = model_params["loc_embedding_dim"]

    # Generate all combinations of the parameters with multiple values

    param_combinations = list(
        itertools.product(
            n_hidden_values, embedding_type_values, embedding_dim_values, learning_rate
        )
    )

    # Get the dataloader and other dataset-related information
    dataloader, n_locs, n_times = get_dataloader(
        dataset, loc_levels, time_levels, n_users, batch_size, device
    )

    # Iterate over all combinations and run the model
    for n_hidden, embedding_type, embedding_dim, lr in param_combinations:
        # Create a new model_params dictionary for each combination
        current_model_params = {
            "n_hidden": n_hidden,
            "embedding_type": embedding_type,
            "loc_embedding_dim": embedding_dim,
            "time_embedding_dim": model_params["time_embedding_dim"],
            "dropout": model_params["dropout"],
            "n_layers": model_params["n_layers"],
        }
        log_info = {
            "dataset": dataset + f"_{n_users}",
            "batch_size": batch_size,
            "lr": learning_rate,
            "model": model_cls.__name__,
            "optimizer": optimizer_cls.__name__,
            "loc_levels": loc_levels,
            "time_levels": time_levels,
        }
        try:
            # Run the model with the current combination of parameters
            run(
                model_cls=model_cls,
                n_users=n_users,
                optimizer_cls=optimizer_cls,
                learning_rate=lr,
                dataloader=dataloader,
                n_locs=n_locs,
                n_times=n_times,
                current_model_params,
                device,
                log_path,
            )

        except:
            pass


def run(
    model_cls: torch.nn.Module,
    n_users: int,
    optimizer_cls: torch.optim.Optimizer,
    learning_rate: float,
    dataloader,
    n_locs: List[int],
    n_times: List[int],
    model_params: dict,
    device: torch.device,
    log_info: Dict,
    log_path: str,
):

    model = model_cls(
        n_locs=n_locs,
        n_users=n_users,
        n_times=n_times,
        device=device,
        **model_params,
    ).to(device)
    xh, th, llh, uh = [], [], [], torch.empty(0, device=device)
    tracker = ExperimentTracker(
        parameters=model_params | log_info,
        logging_interval=LOGGING_INTERVAL,
        save_path=BASEPATH.joinpath("reports", log_path),
    )
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)
    for xc, tc, llc, uc in tqdm(dataloader):

        with torch.inference_mode():
            preds = model.pred_step(xc=xc, tc=tc, llc=llc, xh=xh, th=th, llh=llh, uh=uh)

        loss = model.train_step(
            xc=xc, tc=tc, llc=llc, uc=uc, xh=xh, th=th, llh=llh, uh=uh
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tracker.update(uc, preds, loss)
        xh.append(xc)
        th.append(tc)
        llh.append(llc)
        uh = torch.concat([uh, uc])


if __name__ == "__main__":
    search()
