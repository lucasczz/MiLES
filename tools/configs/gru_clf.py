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
        coordinate_embedding_dim=8,
        hidden_size=128,
        lr=1e-3,
        dropout=0.0,
    ),
    save_dir="gru",
    max_epochs=None,
)
