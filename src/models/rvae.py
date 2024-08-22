from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import lightning as L
from pathlib import Path


class GRUVariationalAutoencoder(L.LightningModule):
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        n_users: int = 181,
        coordinate_embedding_dim: int = 8,
        user_embedding_dim: int = 8,
        time_embedding_dim: int = 4,
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
        val_users: List = [0, 1, 2],
        offset: int = 0,
    ):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_users = n_users
        self.user_embedding_dim = user_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.loss_fn = getattr(F, loss_fn)
        self.optim_fn = getattr(torch.optim, optim_fn)
        self.dropout = dropout
        self.offset = offset
        self.val_users = val_users

        # Model parameters
        self.save_hyperparameters()

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, user_embedding_dim)
        self.col_embedding = nn.Embedding(n_cols, coordinate_embedding_dim)
        self.row_embedding = nn.Embedding(n_rows, coordinate_embedding_dim)
        moves = torch.tensor(
            [[0, 0], [], [0, 1], [0, -1], [1, 0], [-1, 0], [1, -1], [-1, 1]]
        )
        self.move_embedding = nn.Embedding.from_pretrained(moves, freeze=True)
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
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_log_var = nn.Linear(hidden_size, latent_dim)

        # Decoder
        condition_input_size = latent_dim + user_embedding_dim
        self.fc_condition = nn.Linear(condition_input_size, hidden_size)
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.q_output = nn.Linear(hidden_size, n_cols)
        self.r_output = nn.Linear(hidden_size, n_rows)

        # Optimizer and loss function
        self.loss_fn = getattr(F, loss_fn)
        self.optim_fn = getattr(torch.optim, optim_fn)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        y: torch.Tensor,
        traj_lens: List[int],
    ):
        # Embedding lookups
        q_embed = self.col_embedding(q)
        r_embed = self.row_embedding(r)
        traj = torch.cat([q_embed, r_embed], dim=-1)

        # Encode trajectory
        traj_enc = self.fc_enc(traj)
        traj_packed = pack_padded_sequence(
            traj_enc, traj_lens, batch_first=True, enforce_sorted=False
        )
        _, h_enc = self.encoder(traj_packed)

        mu = self.fc_mu(h_enc[-1])
        log_var = self.fc_log_var(h_enc[-1])
        traj_embed = self.reparameterize(mu, log_var)

        # Prepare condition vector for decoding
        user_embed = self.user_embedding(y)
        condition_vector = torch.cat([traj_embed, user_embed], dim=-1)
        condition_vector = self.fc_condition(condition_vector)
        h0 = torch.concat(
            [
                condition_vector[None],
                torch.zeros(
                    self.n_layers - 1, *condition_vector.shape, device=self.device
                ),
            ],
            dim=0,
        )

        # Decode trajectory (shifted for next timestep prediction)
        traj_packed = pack_padded_sequence(
            traj_enc[:, :-1], traj_lens - 1, batch_first=True, enforce_sorted=False
        )
        out_dec, _ = self.decoder(traj_packed, h0)
        out_unpacked, _ = pad_packed_sequence(out_dec, batch_first=True)

        q_pred = self.q_output(out_unpacked)
        r_pred = self.r_output(out_unpacked)

        return q_pred, r_pred, mu, log_var

    def configure_optimizers(self):
        optim = self.optim_fn(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [optim], []

    def training_step(self, batch, batch_idx=0):
        q, r, kl_loss, loss = self.get_reconstructions(batch)
        self.log("train_loss", loss, batch_size=len(q))
        self.log("train_kl_loss", kl_loss, batch_size=len(q))
        return loss

    def validation_step(self, batch, batch_idx=0):
        q, r, kl_loss, loss = self.get_reconstructions(batch)
        self.log("val_loss", loss, batch_size=len(q))
        self.log("val_kl_loss", kl_loss, batch_size=len(q))
        return loss

    def on_validation_end(self) -> None:
        y = torch.tensor(self.val_users, device=self.device)
        trajectories = self.generate(y)
        df = pd.DataFrame(data=trajectories, columns=["q", "r", "y"])
        df.to_csv(
            Path(__file__).parent.parent.parent.joinpath("reports", "rvae", "traj.csv")
        )

    def get_reconstructions(self, batch):
        q, r, y = batch

        q_flip = [qi.flip(0) for qi in q]
        r_flip = [ri.flip(0) for ri in r]

        traj_lens = torch.tensor([len(qi) for qi in q])
        q_flip = pad_sequence(q_flip, batch_first=True)
        r_flip = pad_sequence(r_flip, batch_first=True)
        q_pred, r_pred, mu, log_var = self(q_flip, r_flip, y, traj_lens)

        q_pad = pad_sequence(q, batch_first=True)
        r_pad = pad_sequence(r, batch_first=True)

        # Reconstruction loss (shifted target)
        q_loss = self.loss_fn(q_pred.transpose(1, -1), q_pad[:, 1:])
        r_loss = self.loss_fn(r_pred.transpose(1, -1), r_pad[:, 1:])

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        loss = q_loss + r_loss + kl_loss
        return q, r, kl_loss, loss

    def generate(self, y: torch.Tensor, temperature: float = 1e-6, max_steps: int = 20):
        self.eval()
        with torch.inference_mode():
            user_embed = self.user_embedding(y)

            condition_vector = torch.cat(
                [torch.randn(len(y), self.latent_dim, device=self.device), user_embed],
                dim=-1,
            )
            condition_vector = self.fc_condition(condition_vector)
            h0 = torch.cat(
                [
                    condition_vector[None],
                    torch.zeros(
                        self.n_layers - 1, *condition_vector.shape, device=self.device
                    ),
                ],
                dim=0,
            )
            q_pred = torch.tensor([self.n_cols - 2] * len(y), device=self.device)
            r_pred = torch.ones([self.n_rows - 2] * len(y), device=self.device)
            traj_coords = [
                np.column_stack(
                    [
                        q_pred.numpy(force=True),
                        r_pred.numpy(force=True),
                        y.numpy(force=True),
                    ]
                )
            ]
            traj_embeddings = []
            while (len(traj_coords) < max_steps + 2) and (
                (q_pred != self.n_rows - 1).any() or (r_pred != self.n_rows - 1).any()
            ):
                q_embed = self.col_embedding(q_pred)
                r_embed = self.row_embedding(r_pred)
                new_embed = torch.cat([q_embed, r_embed], dim=-1)
                traj_embeddings.append(new_embed)

                # Encode trajectory
                traj_enc = self.fc_enc(torch.stack(traj_embeddings, dim=1))

                out_dec, _ = self.decoder(traj_enc, h0)

                q_logit = self.q_output(out_dec[:, -1])
                r_logit = self.r_output(out_dec[:, -1])

                q_proba = torch.softmax(q_logit / temperature, dim=-1)
                r_proba = torch.softmax(r_logit / temperature, dim=-1)

                q_pred = torch.multinomial(q_proba, 1)[:, 0]
                r_pred = torch.multinomial(r_proba, 1)[:, 0]
                traj_coords.append(
                    np.column_stack(
                        [
                            q_pred.numpy(force=True),
                            r_pred.numpy(force=True),
                            y.numpy(force=True),
                        ]
                    )
                )
        traj_coords = np.concatenate(traj_coords, axis=0)
        return traj_coords
