from src.models import BiTULER, DeepTUL, MainTUL
from run import run
from torch.optim import Adam


if __name__ == "__main__":
    batch_size = 1
    dataset = "foursquare_NYC"
    n_users = 400
    subsample = 500
    log_path = "debug_emb_tracking.jsonl"

    run(
        dataset="foursquare_TKY",
        model_cls=BiTULER,
        n_users=n_users,
        loc_levels=4,
        loc_level=None,
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="lookup_concat",   
        embedding_weight_factor=2,
        discretization_rows=200,
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=2,
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=subsample,
        seed=2,
        device="cuda:5",
        log_path=log_path,
        track_embeddings=True,
    )
