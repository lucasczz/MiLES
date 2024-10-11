from typing import List
from torchmetrics.functional import accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import lightning as L


class MultilevelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init="default") -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.init = init
        # TODO: add option to initialize lower levels to zero
        self.levels = nn.ModuleList(
            [nn.Embedding(n, embedding_dim) for n in num_embeddings]
        )

    def forward(self, x):
        level_embeddings = torch.stack(
            [level(x[..., idx]) for idx, level in enumerate(self.levels)]
        )
        return level_embeddings.sum(dim=0)


class GRUClassifier(L.LightningModule):
    def __init__(
        self,
        n_cells: List[int],
        n_users: int,
        n_time_intervals: int,
        cell_embedding_dim: int = 16,
        time_embedding_dim: int = 16,
        hidden_size: int = 256,
        n_layers: int = 3,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        loss_fn: str = "cross_entropy",
        optim_fn: str = "AdamW",
        bidirectional: bool = False,
        dropout: float = 0,
        user_weight: List = None,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.n_users = n_users
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = getattr(F, loss_fn)
        self.optim_fn = getattr(torch.optim, optim_fn)
        self.dropout = dropout
        self.user_weight = user_weight if user_weight else [1 / n_users] * n_users
        self.n_time_intervals = n_time_intervals

        # Model parameters
        self.save_hyperparameters()

        # Embeddings
        self.cell_embedding = MultilevelEmbedding(n_cells, cell_embedding_dim)

        self.time_embedding = nn.Embedding(n_time_intervals, time_embedding_dim)

        # Encoder
        self.fc_enc = nn.Linear(cell_embedding_dim + time_embedding_dim, hidden_size)
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc_clf = nn.Linear(hidden_size, self.n_users)

        # Optimizer and loss function
        self.loss_fn = getattr(F, loss_fn)
        self.optim_fn = getattr(torch.optim, optim_fn)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ):
        # Embedding lookups
        traj_lens = torch.tensor([len(xi) for xi in x])
        x_pad = pad_sequence(x, batch_first=True)
        t_pad = pad_sequence(t, batch_first=True)
        t_embed = self.time_embedding(t_pad)
        x_embed = self.cell_embedding(x_pad)
        traj_embed = torch.cat([x_embed, t_embed], dim=-1)

        # Encode trajectory
        traj_enc = self.fc_enc(traj_embed)
        x_packed = pack_padded_sequence(
            traj_enc, traj_lens, batch_first=True, enforce_sorted=False
        )
        _, h_enc = self.encoder(x_packed)

        y_pred = self.fc_clf(h_enc[-1])

        return y_pred

    def configure_optimizers(self):
        optim = self.optim_fn(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [optim], []

    def training_step(self, batch, batch_idx=0):
        x, y, t = batch
        y_pred = self(x, t)

        user_weight = torch.tensor(self.user_weight, device=self.device)
        loss = self.loss_fn(y_pred, y, weight=user_weight)

        acc = accuracy(
            y_pred, y, task="multiclass", num_classes=self.n_users, average="micro"
        )
        self.log("train_top1_accuracy", acc, batch_size=len(x))
        self.log("train_loss", loss, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx=0):
        x, y, t = batch
        y_pred = self(x, t)

        user_weight = torch.tensor(self.user_weight, device=self.device)
        loss = self.loss_fn(y_pred, y, weight=user_weight)

        top1_acc = accuracy(y_pred, y, task="multiclass", num_classes=self.n_users)
        top5_acc = accuracy(
            y_pred, y, task="multiclass", num_classes=self.n_users, top_k=5
        )
        self.log("val_top1_accuracy", top1_acc, batch_size=len(x))
        self.log("val_top5_accuracy", top5_acc, batch_size=len(x))
        self.log("val_loss", loss, batch_size=len(x))
        return loss
