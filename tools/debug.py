from src.models import BiTULER, TULHOR
from run import run


if __name__ == "__main__":
    batch_size = 1
    dataset = "foursquare_TKY"
    n_users = 400
    device = "cuda:4"
    subsample = 5000
    log_path = "debug_tuler.jsonl"

    run(
        model_cls=BiTULER,
        dataset_name=dataset,
        n_hidden=1024,
        n_layers=1,
        loc_levels=3,
        time_levels=2,
        lr=2e-4,
        embedding_type="lookup_weighted_sum",
        loc_embedding_factor=1.0,
        time_embedding_factor=1 / 8,
        discretization_rows=600,
        aggregation_mode="grow",
        grow_factor=2,
        dropout=0.0,
        n_users=n_users,
        device=device,
        subsample=subsample,
        log_path=log_path,
        verbose=True,
    )
