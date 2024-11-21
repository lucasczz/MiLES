import math
from typing import List
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from einops import rearrange

from src.embedding import EMBEDDING_TYPES


class CurrentEncoder(nn.Module):
    def __init__(
        self,
        n_locs,
        n_times,
        n_hidden: int = 64,
        embedding_type: str = "lookup",
        loc_embedding_dim: int = 64,
        time_embedding_dim: int = 32,
        dropout: float = 0.6,
        n_layers: int = 1,
        device: torch.device = "cuda:0",
        bidirectional: bool = True,
    ):
        super().__init__()
        self.device = device
        self.n_hidden = n_hidden
        self.n_locs = n_locs
        self.n_times = n_times
        self.n_layers = n_layers
        self.n_dirs = 1 + bidirectional
        self.embedding = EMBEDDING_TYPES[embedding_type](
            num_embeddings_loc=n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=n_times,
            embedding_dim_time=time_embedding_dim,
        )
        self.lstm = nn.LSTM(
            self.embedding.dim,
            n_hidden,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: List[torch.Tensor], t: List[torch.Tensor]):
        traj_lens = torch.tensor([len(xi) for xi in x])
        x_pad = pad_sequence(x, batch_first=True).to(self.device)
        t_pad = pad_sequence(t, batch_first=True).to(self.device)
        xt_embed = self.dropout(self.embedding(x_pad, t_pad))

        # Encode trajectory
        xt_packed = pack_padded_sequence(
            xt_embed, traj_lens, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(xt_packed)

        return rearrange(h[-self.n_dirs :], "dirs batch hidden -> batch (dirs hidden)")


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_times: int,
        n_users: int,
        n_hidden: int = 64,
        embedding_type: str = "lookup",
        loc_embedding_dim: int = 64,
        time_embedding_dim: int = 32,
        user_embedding_dim: int = 32,
        dropout: float = 0.6,
        n_layers: int = 1,
        device: torch.device = "cuda:0",
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_locs = n_locs
        self.n_times = n_times
        self.n_layers = n_layers
        self.embedding = EMBEDDING_TYPES[embedding_type](
            num_embeddings_loc=n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=n_times,
            embedding_dim_time=time_embedding_dim,
        )
        self.user_embedding_dim = user_embedding_dim
        self.embedding_dim = self.embedding.dim + user_embedding_dim

        self.user_embed = nn.Embedding(n_users + 1, user_embedding_dim, padding_idx=0)
        self.fc_xtu = nn.Linear(self.embedding_dim, 2 * n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u: torch.Tensor,
    ):
        # x.shape = (seq_len, n_features)

        # Step 1: Get unique combinations of location and time IDs
        xt = torch.stack([x[..., 0], t[..., 0]], dim=-1)  # shape = (sum(seq_len), 2)
        xt_unique, inv_idcs, counts = torch.unique(
            xt, dim=0, return_inverse=True, return_counts=True
        )

        xt_unique = xt_unique.to(self.device)
        inv_idcs = inv_idcs.to(self.device)
        counts = counts.to(self.device)

        n_unique = xt_unique.shape[0]

        # Step 2: Embed locations and times
        xt_embed = self.dropout(
            self.embedding(xt_unique[..., 0, None], xt_unique[..., 1, None])
        )
        ui_embed = self.dropout(self.user_embed(u.to(self.device)))

        # Step 3: Aggregate user embeddings for each unique (loc, time) pair
        u_emb_sum = torch.zeros(n_unique, self.user_embedding_dim, device=self.device)
        # Use scatter_add_ to efficiently add u_embeds to emb_sum
        u_emb_sum = u_emb_sum.scatter_add_(
            0,
            inv_idcs.unsqueeze(-1).expand(-1, self.user_embedding_dim),
            ui_embed,
        )
        u_embed = u_emb_sum / counts[..., None]

        # Step 4: Combine embeddings
        xtu_embed = torch.cat([xt_embed, u_embed], dim=-1)
        xtu_embed = self.dropout(xtu_embed)

        return F.tanh(self.fc_xtu(xtu_embed))  # shape = (n_unique, 2 *n_hidden)


def history_attention(current, history):
    # current.shape = (batch_size, 2*n_hidden)
    # history.shape = ( n_unique, 2*n_hidden)

    n_unique, n_hidden = history.shape

    # Compute logits using batch matrix multiplication
    logits = torch.einsum("ij,tj->it", current, history) / math.sqrt(n_hidden)
    # logits.shape = (batch_size, n_unique)

    # Compute attention weights using softmax
    attn = torch.softmax(logits, dim=-1)
    return attn


class DeepTUL(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_times: int,
        n_users: int,
        n_hidden: int = 64,
        embedding_type: str = "lookup",
        loc_embedding_dim: int = 64,
        time_embedding_dim: int = 32,
        user_embedding_dim: int = 32,
        dropout: float = 0.6,
        n_layers: int = 1,
        device: torch.device = "cuda:0",
        bidirectional: bool = True,
    ):
        super().__init__()
        self.c_encoder = CurrentEncoder(
            n_locs=n_locs,
            n_times=n_times,
            n_hidden=n_hidden,
            embedding_type=embedding_type,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
            n_layers=n_layers,
            device=device,
            bidirectional=bidirectional,
        )
        self.h_encoder = HistoryEncoder(
            n_locs=n_locs,
            n_times=n_times,
            n_hidden=n_hidden,
            n_users=n_users,
            embedding_type=embedding_type,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            dropout=dropout,
            n_layers=n_layers,
            device=device,
        )
        self.clf = nn.Linear(4 * n_hidden, n_users)
        self.device = device

    def forward(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        xh: List[torch.Tensor] = [],
        th: List[torch.Tensor] = [],
        uh: torch.Tensor = torch.empty(0),
    ):
        c_enc = self.c_encoder(xc, tc)
        if len(uh) > 0:
            xh_cat = torch.cat(xh)
            th_cat = torch.cat(th)
            uh_repeat = uh.repeat_interleave(torch.tensor([len(xhi) for xhi in xh]))
            h_enc = self.h_encoder(xh_cat, th_cat, uh_repeat)
            hc_attn = history_attention(c_enc, h_enc)

            # c_enc.shape = (batch_size, 2 * n_hidden)
            # h_attn.shape = (batch_size, n_xt_unique)
            # h_enc.shape = (n_xt_unique, 2 * n_hidden)

            # Return the weighted sum of history with attention applied
            h_context = torch.einsum("it,tj->ij", hc_attn, h_enc)
            # h_context.shape = (batch_size, 2 * n_hidden)
        else:
            h_context = torch.zeros_like(c_enc)

        return self.clf(torch.cat([c_enc, h_context], dim=-1))

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        uc: torch.Tensor,
        xh: List[torch.Tensor],
        th: List[torch.Tensor],
        uh: torch.Tensor,
        **kwargs
    ):

        preds = self(xc, tc, xh, th, uh)
        loss = F.cross_entropy(preds, uc.to(self.device))
        return loss

    def pred_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        xh: List[torch.Tensor],
        th: List[torch.Tensor],
        uh: List[torch.Tensor],
        **kwargs
    ):

        logits = self(xc, tc, xh, th, uh)
        return logits
