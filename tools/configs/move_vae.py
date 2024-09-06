from types import SimpleNamespace

config = SimpleNamespace(
    gpus=[0],
    batch_size=128,
    val_split=0.1,
    test_split=0.2,
    n_workers=4,
    model_kwargs=dict(
        n_layers=2,
        loss_fn="cross_entropy",
        user_embedding_dim=8,
        hidden_size=512,
        latent_dim=256,
        lr=3e-4,
        dropout=0.5,
    ),
    save_dir="rvae",
    max_epochs=None,
)
