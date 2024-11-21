from run import run
from src.models.tuler import BiTULER
from torch.optim import Adam


def main():
    run(
        model_cls=BiTULER,
        dataset="foursquare_NYC",
        n_users=400,
        loc_levels=2,
        time_levels=1,
        batch_size=1,
        device="cuda:0",
        log_path="tuler_test.csv",
        optimizer_cls=Adam,
        learning_rate=.25e-2,
        model_params=dict(
            n_hidden=128,
            embedding_type="lookup",
            loc_embedding_dim=32,
            time_embedding_dim=8,
            dropout=0.0,
            n_layers=1,
        ),
    )


if __name__ == "__main__":
    main()
