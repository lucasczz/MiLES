from tqdm import tqdm
from run import run
from src.data.utils import get_dataloader
from src.models.maintul import MainTUL
from torch.optim import Adam

import itertools


def search():
    model_cls = MainTUL
    dataset = "foursquare_TKY"
    subsample = 5_000
    n_users = 400
    loc_levels = 1
    time_levels = 1
    batch_size = 1
    device = "cuda:0"
    log_path = "maintul_gridsearch.csv"
    optimizer_cls = Adam
    lrs = [1e-4 * 2**i for i in range(6)]

    n_hiddens = [64, 128, 256, 512]
    embedding_type = "lookup"
    loc_embedding_factors = [0.5, 1]
    time_embedding_factor = 0.25
    dropout = 0.0
    n_layerss = [1, 2, 3]
    n_headss = [4, 8]

    # Generate all combinations of the parameters with multiple values
    param_combinations = list(
        itertools.product(
            lrs,
            n_hiddens,
            n_layerss,
            n_headss,
            loc_embedding_factors,
        )
    )

    # Get the dataloader and other dataset-related information
    dataloader, n_locs, n_times = get_dataloader(
        dataset, n_users, batch_size, device, subsample
    )

    # Iterate over all combinations and run the model
    for (
        lr,
        n_hidden,
        n_layers,
        n_heads,
        loc_embedding_factor,
    ) in tqdm(param_combinations):
        # Create a new model_params dictionary for each combination
        loc_embedding_dim = int(loc_embedding_factor * n_hidden)
        time_embedding_dim = int(time_embedding_factor * loc_embedding_dim)
        current_model_params = {
            "n_hidden": n_hidden,
            "embedding_type": embedding_type,
            "loc_embedding_dim": loc_embedding_dim,
            "time_embedding_dim": time_embedding_dim,
            "dropout": dropout,
            "n_layers": n_layers,
            "n_heads": n_heads,
        }
        log_info = {
            "dataset": dataset + f"_{n_users}",
            "batch_size": batch_size,
            "lr": lr,
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
                n_locs=n_locs[:loc_levels],
                n_times=n_times[:time_levels],
                model_params=current_model_params,
                device=device,
                log_path=log_path,
                log_info=log_info,
                verbose=False,
            )
        except:
            pass


def main():
    model_cls = MainTUL
    dataset = "foursquare_NYC"
    n_users = 400
    loc_levels = 1
    time_levels = 1
    batch_size = 1
    device = "cuda:0"
    log_path = "maintul_debug"
    optimizer_cls = Adam
    learning_rate = 4e-4
    model_params = dict(
        n_hidden=2048,
        embedding_type="lookup",
        loc_embedding_dim=1024,
        time_embedding_dim=256,
        dropout=0.0,
        n_layers=1,
        n_heads=8,
        lambduh=2,
        distill_temp=10,
        n_augment=4,
    )

    # Get the dataloader and other dataset-related information
    dataloader, n_locs, n_times = get_dataloader(
        dataset, n_users, batch_size, device, subsample=5_000
    )

    # Iterate over all combinations and run the model
    log_info = {
        "dataset": dataset + f"_{n_users}",
        "batch_size": batch_size,
        "lr": learning_rate,
        "model": model_cls.__name__,
        "optimizer": optimizer_cls.__name__,
        "loc_levels": loc_levels,
        "time_levels": time_levels,
    }
    # Run the model with the current combination of parameters
    run(
        model_cls=model_cls,
        n_users=n_users,
        optimizer_cls=optimizer_cls,
        learning_rate=learning_rate,
        dataloader=dataloader,
        n_locs=n_locs[:loc_levels],
        n_times=n_times[:loc_levels],
        model_params=model_params,
        device=device,
        log_path=log_path,
        log_info=log_info,
    )


if __name__ == "__main__":
    dataset = "foursquare_NYC"
    n_users = 400
    batch_size = 1
    device = "cuda:0"

    # Get the dataloader and other dataset-related information
    dataloader, n_locs, n_times = get_dataloader(
        dataset, n_users, batch_size, device, subsample=100
    )

    # Run the model with the current combination of parameters
    run(
        dataset_name="foursquare_NYC",
        dataloader=dataloader,
        model_cls=MainTUL,
        n_users=400,
        n_locs=n_locs,
        n_times=n_times,
        loc_levels=1,
        time_levels=1,
        optimizer_cls=Adam,
        lr=1e-3,
        n_hidden=128,
        n_layers=1,
        n_heads=8,
        lambduh=2,
        device=device,
        log_path="test.jsonl",
    )
