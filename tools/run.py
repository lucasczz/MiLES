from pathlib import Path
from typing import Dict, List
import torch
from tqdm import tqdm

from src.data.tracker import ExperimentTracker


BASEPATH = Path(__file__).parent.parent
LOGGING_INTERVAL = 100


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
    verbose: bool = True,
):

    model = model_cls(
        n_locs=n_locs,
        n_users=n_users,
        n_times=n_times,
        device=device,
        **model_params,
    ).to(device)
    # model = torch.compile(model)
    xh, th, llh, uh = [], [], [], torch.empty(0, device=device)
    tracker = ExperimentTracker(
        parameters=model_params | log_info,
        logging_interval=LOGGING_INTERVAL,
        save_path=BASEPATH.joinpath("reports", log_path),
    )
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)
    if verbose:
        iterator = tqdm(dataloader)
    else:
        dataloader
    for xc, tc, llc, uc in iterator:

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

    tracker.log_step()
