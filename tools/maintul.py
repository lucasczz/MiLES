from run import run
from src.models.maintul import MainTUL
from torch.optim import Adam


def main():
    run(
        model_cls=MainTUL,
        dataset="foursquare_NYC",
        n_users=400,
        loc_levels=1,
        time_levels=1,
        batch_size=4,
        device="cpu",
        log_path="maintul_test.csv",
        optimizer_cls=Adam,
        learning_rate=1e-3,
        model_params=dict(
            n_hidden=64,
            embedding_type="lookup",
            loc_embedding_dim=32,
            time_embedding_dim=8,
            lambduh=10,
            distill_temp=10,
            dropout=0.1,
            n_heads=4,
            n_layers=2,
            n_augment=16,
        ),
        # debug=True,
    )


if __name__ == "__main__":
    main()
