from typing import List, Literal
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from src.embedding import EMBEDDING_TYPES


class TULER(nn.Module):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 16,
        time_embedding_dim: int = 8,
        n_hidden: int = 256,
        n_layers: int = 3,
        bidirectional: bool = True,
        dropout: float = 0,
        rnn_type: Literal["LSTM", "GRU"] = "LSTM",
        device: torch.device = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        self.n_locs = n_locs
        self.n_users = n_users
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.n_dirs = 1 + bidirectional

        # Embeddings
        self.embedding = EMBEDDING_TYPES[embedding_type](
            num_embeddings_loc=n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=n_times,
            embedding_dim_time=time_embedding_dim,
            weight_factor=embedding_weight_factor,
        )
        self.rnn_type = rnn_type

        if rnn_type == "GRU":
            self.encoder = nn.GRU(
                input_size=self.embedding.dim,
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        else:
            self.encoder = nn.LSTM(
                input_size=self.embedding.dim,
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        self.fc_clf = nn.Linear(self.n_dirs * n_hidden, self.n_users)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Embedding lookups
        traj_lens = torch.tensor([len(xi) for xi in x])
        x_pad = pad_sequence(x, batch_first=True)
        t_pad = pad_sequence(t, batch_first=True)
        x_embedded = self.embedding(x_pad, t_pad)

        # Encode trajectory
        x_packed = pack_padded_sequence(
            x_embedded, traj_lens, batch_first=True, enforce_sorted=False
        )
        if self.rnn_type == "GRU":
            _, h = self.encoder(x_packed)
        else:
            _, (h, _) = self.encoder(x_packed)
        h = rearrange(h[-self.n_dirs :], "dirs batch hidden -> batch (dirs hidden)")
        y_pred = self.fc_clf(self.dropout(h))

        return y_pred

    def train_step(
        self, xc: List[torch.Tensor], tc: List[torch.Tensor], uc: torch.Tensor, **kwargs
    ):
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        preds = self(xc_padded, tc_padded)
        loss = F.cross_entropy(preds, uc.to(self.device))
        return loss

    def pred_step(self, xc: List[torch.Tensor], tc: List[torch.Tensor], **kwargs):
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        logits = self(xc_padded, tc_padded)
        return logits


class BiTULER(TULER):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 16,
        time_embedding_dim: int = 8,
        n_hidden=256,
        n_layers=3,
        dropout=0,
        rnn_type="LSTM",
        device="cuda:0",
    ):
        super().__init__(
            n_locs=n_locs,
            n_users=n_users,
            n_times=n_times,
            embedding_type=embedding_type,
            embedding_weight_factor=embedding_weight_factor,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            device=device,
            bidirectional=True,
        )


class TULERG(TULER):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 16,
        time_embedding_dim: int = 8,
        n_hidden=256,
        n_layers=3,
        dropout=0,
        device="cuda:0",
    ):
        super().__init__(
            n_locs=n_locs,
            n_users=n_users,
            n_times=n_times,
            embedding_type=embedding_type,
            embedding_weight_factor=embedding_weight_factor,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
            rnn_type="GRU",
            device=device,
            bidirectional=False,
        )


class TULERL(TULER):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 16,
        time_embedding_dim: int = 8,
        n_hidden=256,
        n_layers=3,
        dropout=0,
        device="cuda:0",
    ):
        super().__init__(
            n_locs=n_locs,
            n_users=n_users,
            n_times=n_times,
            embedding_type=embedding_type,
            embedding_weight_factor=embedding_weight_factor,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
            rnn_type="LSTM",
            device=device,
            bidirectional=False,
        )
