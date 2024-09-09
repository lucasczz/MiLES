from typing import List
from torchmetrics.functional import accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import lightning as L


class GRUClassifier(L.LightningModule):
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        n_users: int = 181,
        coordinate_embedding_dim: int = 8,
        hidden_size: int = 256,
        latent_dim: int = 128,
        n_layers: int = 3,
        lr: float = 0.001,
        beta: float = 1.0,
        weight_decay: float = 0.0001,
        loss_fn: str = "cross_entropy",
        optim_fn: str = "AdamW",
        bidirectional: bool = False,
        dropout: float = 0,
        user_weight: List = None,
    ):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_users = n_users
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.loss_fn = getattr(F, loss_fn)
        self.optim_fn = getattr(torch.optim, optim_fn)
        self.dropout = dropout
        self.user_weight = user_weight if user_weight else [1 / n_users] * n_users

        # Model parameters
        self.save_hyperparameters()

        # Embeddings
        self.col_embedding = nn.Embedding(n_cols, coordinate_embedding_dim)
        self.row_embedding = nn.Embedding(n_rows, coordinate_embedding_dim)
        # Encoder
        self.fc_enc = nn.Linear(2 * coordinate_embedding_dim, hidden_size)
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc_clf = nn.Linear(latent_dim, self.n_users)

        # Optimizer and loss function
        self.loss_fn = getattr(F, loss_fn)
        self.optim_fn = getattr(torch.optim, optim_fn)

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
    ):
        # Embedding lookups
        traj_lens = torch.tensor([len(qi) for qi in q])
        q_pad = pad_sequence(q, batch_first=True)
        r_pad = pad_sequence(r, batch_first=True)
        q_embed = self.col_embedding(q_pad)
        r_embed = self.row_embedding(r_pad)
        traj = torch.cat([q_embed, r_embed], dim=-1)

        # Encode trajectory
        traj_enc = self.fc_enc(traj)
        traj_packed = pack_padded_sequence(
            traj_enc, traj_lens, batch_first=True, enforce_sorted=False
        )
        _, h_enc = self.encoder(traj_packed)

        y_pred = self.fc_clf(h_enc[-1])

        return y_pred

    def configure_optimizers(self):
        optim = self.optim_fn(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [optim], []

    def training_step(self, batch, batch_idx=0):
        q, r, y = batch
        y_pred = self(q, r)

        user_weight = torch.tensor(self.user_weight, device=self.device)
        loss = self.loss_fn(y_pred, y, weight=user_weight)

        acc = accuracy(y_pred, y, task="multiclass", num_classes=self.n_users, average='micro')
        self.log("train_top1_accuracy", acc, batch_size=len(q))
        self.log("train_loss", loss, batch_size=len(q))
        return loss

    def validation_step(self, batch, batch_idx=0):
        q, r, y = batch
        y_pred = self(q, r)

        user_weight = torch.tensor(self.user_weight, device=self.device)
        loss = self.loss_fn(y_pred, y, weight=user_weight)

        top1_acc = accuracy(y_pred, y, task="multiclass", num_classes=self.n_users)
        top5_acc = accuracy(
            y_pred, y, task="multiclass", num_classes=self.n_users, top_k=5
        )
        self.log("val_top1_accuracy", top1_acc, batch_size=len(q))
        self.log("val_top5_accuracy", top5_acc, batch_size=len(q))
        self.log("val_loss", loss, batch_size=len(q))
        return loss
