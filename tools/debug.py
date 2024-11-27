from src.data.utils import get_dataloader
from src.models import BiTULER
from run import run


if __name__ == "__main__":
    batch_size = 1
    dataset = "foursquare_TKY"
    n_users = 400
    device = "cuda:1"
    subsample = 5000
    log_path = "debug_embeddings.jsonl"
    # Get the dataloader and other dataset-related information
    dataloader, n_locs, n_times = get_dataloader(
        dataset, n_users, batch_size, subsample
    )

    run(
        model_cls=BiTULER,
        dataset_name=dataset,
        dataloader=dataloader,
        n_locs=n_locs,
        n_hidden=1024,
        n_layers=1,
        loc_levels=2,
        time_levels=2,
        lr=2e-4,
        embedding_type="lookup_sum",
        loc_embedding_factor=1.0,
        time_embedding_factor=2 / 16,
        dropout=0.0,
        n_times=n_times,
        n_users=n_users,
        device=device,
        log_path=log_path,
        verbose=True,
    )
