from run import run
from src.data.utils import get_dataloader
from src.models.tuler import BiTULER
from torch.optim import Adam

import itertools


def search():
    model_cls = BiTULER
    dataset = "foursquare_NYC"
    n_users = 400
    loc_levelss = [1, 2, 3]
    time_levelss = [1, 2, 3]
    batch_size = 1
    device = "cuda:0"
    log_path = "tuler_gridsearch.csv"
    optimizer_cls = Adam
    lrs = [1e-3 * 2**i for i in range(6)]

    n_hiddens = [64, 128, 256]
    embedding_types = ["lookup", "rotary", "cosine"]
    loc_embedding_dims = [32, 64]
    time_embedding_dim = 8
    dropout = 0.0
    n_layers = 1

    # Extract the parameters that have multiple values

    # Generate all combinations of the parameters with multiple values

    param_combinations = list(
        itertools.product(
            loc_levelss,
            time_levelss,
            lrs,
            n_hiddens,
            embedding_types,
            loc_embedding_dims,
        )
    )

    # Get the dataloader and other dataset-related information
    dataloader, n_locs, n_times = get_dataloader(dataset, n_users, batch_size, device)

    # Iterate over all combinations and run the model
    for (
        loc_levels,
        time_levels,
        lr,
        n_hidden,
        embedding_type,
        embedding_dim,
        lr,
    ) in param_combinations:
        # Create a new model_params dictionary for each combination
        current_model_params = {
            "n_hidden": n_hidden,
            "embedding_type": embedding_type,
            "loc_embedding_dim": embedding_dim,
            "time_embedding_dim": time_embedding_dim,
            "dropout": dropout,
            "n_layers": n_layers,
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
            )
        except:
            pass


def main():
    model_cls = BiTULER
    dataset = "foursquare_NYC"
    n_users = 400
    loc_levels = 1
    time_levels = 1
    batch_size = 1
    device = "cpu"
    log_path = "tuler_debug.csv"
    optimizer_cls = Adam
    learning_rate = 1e-2
    model_params = dict(
        n_hidden=64,
        embedding_type="rotary",
        loc_embedding_dim=64,
        time_embedding_dim=8,
        dropout=0.0,
        n_layers=1,
    )

    # Get the dataloader and other dataset-related information
    dataloader, n_locs, n_times = get_dataloader(
        dataset, n_users, batch_size, device, subsample=True
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
    main()
