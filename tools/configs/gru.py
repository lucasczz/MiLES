from types import SimpleNamespace

config = SimpleNamespace(
    gpus=[0],
    batch_size=64,
    n_hex_rows=100,
    val_split=0.1,
    test_split=0.2,
    n_workers=4,
    n_layers=1,
    n_hex_levels=3,
    loss_fn="cross_entropy",
    cell_embedding_dim=32,
    hidden_size=512,
    lr=5e-4,
    dropout=0.0,
    save_dir="gru",
    max_epochs=None,
    bidirectional=False
)
