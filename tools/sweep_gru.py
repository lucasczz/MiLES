import pathlib
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger

import wandb
from src.data.loader import GeolifeModule
from src.models.tuler import GRUClassifier


def run():
    # Initialize wandb if a sweep is running
    with wandb.init():
        config = wandb.config

        seed = 42
        seed_everything(seed, workers=True)

        basepath = pathlib.Path(__file__).parent.parent.joinpath("reports")
        save_dir = basepath.joinpath(config.save_dir)
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
            callbacks=[early_stop],
            gradient_clip_val=0.5,
        )

        data = GeolifeModule(
            n_hex_rows=config.n_hex_rows,
            batch_size=config.batch_size,
            test_split=config.test_split,
            val_split=config.val_split,
            n_workers=config.n_workers,
        )
        data_info = data.get_info()

        model = GRUClassifier(
            optim_fn="AdamW",
            weight_decay=1e-5,
            n_cells=data_info["n_cells"],
            n_users=data_info["n_users"],
            user_weight=data_info["user_weight"],
            cell_embedding_dim=config.cell_embedding_dim,
            hidden_size=config.hidden_size,
            lr=config.lr,
            n_layers=config.n_layers,
            bidirectional=config.bidirectional,
        )
        trainer.fit(model, datamodule=data)
        wandb.finish()


if __name__ == "__main__":
    # Define sweep configuration
    sweep_config = {
        "method": "bayes",  # You can use 'grid' or 'random' for grid/random search
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "n_hex_rows": {"value": 50},
            "hidden_size": {"values": [64, 128, 256, 512]},
            "n_layers": {"values": [1, 2, 3]},
            "cell_embedding_dim": {"values": [16, 32, 64]},
            "lr": {"min": 1e-5, "max": 1e-3},
            'n_hex_levels': {'values': []},
            "bidirectional": {"values": [True, False]},
            "max_epochs": {"value": 50},
            "gpus": {"value": 1},
            "batch_size": {"value": 64},
            "test_split": {"value": 0.2},
            "val_split": {"value": 0.1},
            "n_workers": {"value": 4},
            "save_dir": {"value": "gru_sweep"},
        },
    }
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="lbn-trajectories")

    # Launch the sweep
    wandb.agent(sweep_id, function=run, count=20)  # Adjust count as needed
