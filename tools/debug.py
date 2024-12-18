from src.models import BiTULER, DeepTUL
from run import run
from torch.optim import Adam



if __name__ == "__main__":
    batch_size = 1
    dataset = "foursquare_NYC"
    n_users = 400
    subsample = 5000
    log_path = "debug_ablation.jsonl"

    run(
        dataset="foursquare_TKY",
        model_cls=BiTULER,
        n_users=400,
        loc_levels=4,
        loc_level=1,
        time_levels=1,
        optimizer_cls=Adam,
        embedding_type="lookup_concat",
        discretization_rows=200,
        discretization_shape="hex",
        aggregation_mode="grow",
        grow_factor=4,
        lr=2e-4,
        n_hidden=1024,
        n_layers=1,
        loc_embedding_factor=1,
        time_embedding_factor=1 / 16,
        subsample=5000,
        seed=42,
        device="cuda:0",
        log_path=log_path,
    )
