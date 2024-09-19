import pathlib
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger

import wandb
from src.data.loader import GeolifeModule
from src.models.gru import GRUClassifier
from utils import load_config_from_file


def run(config):
    wandb.init(project="lbn-trajectories")
    seed = 42

    seed_everything(seed, workers=True)

    # model = torch.compile(model)
    basepath = pathlib.Path(__file__).parent.parent.joinpath("reports")
    save_dir = basepath.joinpath(config.save_dir)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=6)
    logger = WandbLogger(save_dir=save_dir)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        default_root_dir=save_dir,
        logger=logger,
        max_epochs=config.max_epochs,
        devices=config.gpus,
        accelerator="gpu",
        inference_mode=True,
        callbacks=[checkpoint, early_stop],
        gradient_clip_val=0.5,
    )

    data = GeolifeModule(
        batch_size=config.batch_size,
        test_split=config.test_split,
        val_split=config.val_split,
        n_workers=config.n_workers,
        n_hex_rows=config.n_hex_rows,
    )
    data_info = data.get_info()
    model = GRUClassifier(
        optim_fn="AdamW",
        weight_decay=1e-5,
        n_cells=data_info["n_cells"],
        n_users=data_info["n_users"],
        user_weight=data_info["user_weight"],
        n_time_intervals=data_info["n_time_intervals"],
        cell_embedding_dim=config.cell_embedding_dim,
        hidden_size=config.hidden_size,
        lr=config.lr,
        n_layers=config.n_layers,
        bidirectional=config.bidirectional,
    )
    trainer.fit(model, datamodule=data)
    wandb.finish()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python tools/train.py tools/configs/config_file.py")
        sys.exit(1)

    file_path = sys.argv[1]
    config = load_config_from_file(file_path)
    run(config)
