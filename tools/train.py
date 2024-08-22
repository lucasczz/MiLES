import pathlib
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger

import wandb
from src.data.loader import GeolifeModule
from src.models.rvae import GRUVariationalAutoencoder
from utils import MODELS
from utils import load_config_from_file


def run(config):
    wandb.init(project="lbn-trajectories")
    seed = 42

    seed_everything(seed, workers=True)

    # model = torch.compile(model)
    basepath = pathlib.Path(__file__).parent.parent.joinpath("reports")
    save_dir = basepath.joinpath(config.save_dir)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min")
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
    )
    data_info = data.get_info()
    model = GRUVariationalAutoencoder(
        optim_fn="AdamW",
        weight_decay=1e-5,
        time_embedding_dim=data_info["n_timeslots"],
        n_cols=data_info["n_cols"],
        n_rows=data_info["n_rows"],
        n_users=data_info["n_users"],
        **config.model_kwargs,
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
