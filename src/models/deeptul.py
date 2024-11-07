from typing import List
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from einops import rearrange


class CurrentEncoder(nn.Module):
    def __init__(
        self,
        n_locs,
        n_times,
        n_hidden: int = 64,
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
        self.loc_embedding_dim = loc_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.embedding_dim = time_embedding_dim + loc_embedding_dim

        self.lstm = nn.LSTM(
            self.embedding_dim,
            n_hidden,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.loc_embed = nn.Embedding(n_locs + 1, loc_embedding_dim, padding_idx=0)
        self.time_embed = nn.Embedding(n_times + 1, time_embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: List[torch.Tensor], t: List[torch.Tensor]):
        traj_lens = torch.tensor([len(xi) for xi in x])
        x_pad = pad_sequence(x, batch_first=True).to(self.device)
        t_pad = pad_sequence(t, batch_first=True).to(self.device)
        x_embed = self.loc_embed(x_pad)
        t_embed = self.time_embed(t_pad)
        x_embed = self.dropout(x_embed)
        t_embed = self.dropout(t_embed)
        xt_embed = torch.cat([x_embed, t_embed], dim=-1)

        # Encode trajectory
        xt_packed = pack_padded_sequence(
            xt_embed, traj_lens, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(xt_packed)  # out.shape = (batch, seq, 2*n_hidden)

        return rearrange(h[-self.n_dirs :], "dirs batch hidden -> batch (dirs hidden)")


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_times: int,
        n_users: int,
        n_hidden: int = 64,
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
        self.loc_embedding_dim = loc_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.user_embedding_dim = user_embedding_dim
        self.embedding_dim = time_embedding_dim + loc_embedding_dim + user_embedding_dim

        self.loc_embed = nn.Embedding(n_locs + 1, loc_embedding_dim, padding_idx=0)
        self.time_embed = nn.Embedding(n_times + 1, time_embedding_dim, padding_idx=0)
        self.user_embed = nn.Embedding(n_users + 1, user_embedding_dim, padding_idx=0)
        self.fc_xtu = nn.Linear(self.embedding_dim, 2 * n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(
        self,
        x: List[torch.Tensor],
        t: List[torch.Tensor],
        u: List[torch.Tensor],
    ):
        # x.shape = [(len(seq), 1) for seq in seqs for seqs in x]
        all_xt_unique, all_inv_idcs, all_counts, all_u = [], [], [], []
        for xi, ti, ui in zip(x, t, u):
            # Step 1: Get unique combinations of location and time IDs
            xti = torch.stack([xi, ti], dim=-1)  # shape = (sum(seq_len), 2)
            xt_unique, inv_idcs, counts = torch.unique(
                xti, dim=0, return_inverse=True, return_counts=True
            )
            all_xt_unique.append(xt_unique)
            all_inv_idcs.append(inv_idcs)
            all_counts.append(counts)
            all_u.append(ui)

        n_unique = torch.tensor(
            [len(count) for count in all_counts], device=self.device, dtype=torch.int64
        )
        counts_padded = pad_sequence(all_counts, batch_first=True, padding_value=-1).to(
            self.device
        )
        xt_unique_padded = pad_sequence(all_xt_unique, batch_first=True).to(self.device)
        inv_idcs_padded = pad_sequence(all_inv_idcs, batch_first=True).to(self.device)
        u_padded = pad_sequence(all_u, batch_first=True).to(self.device)

        # Step 2: Embed locations and times
        x_embed = self.dropout(self.loc_embed(xt_unique_padded[..., 0]))
        t_embed = self.dropout(self.time_embed(xt_unique_padded[..., 1]))
        ui_embed = self.dropout(self.user_embed(u_padded))

        # Step 3: Aggregate user embeddings for each unique (loc, time) pair
        batch_size, n_unique_max, _ = xt_unique_padded.shape

        u_emb_sum = torch.zeros(
            batch_size, n_unique_max, self.user_embedding_dim, device=self.device
        )
        # Use scatter_add_ to efficiently add u_embeds to emb_sum
        u_emb_sum = u_emb_sum.scatter_add_(
            1,
            inv_idcs_padded.unsqueeze(-1).expand(-1, -1, self.user_embedding_dim),
            ui_embed,
        )
        u_embed = u_emb_sum / counts_padded[..., None]

        # Step 4: Combine embeddings
        xtu_embed = torch.cat([x_embed, t_embed, u_embed], dim=-1)
        xtu_embed = self.dropout(xtu_embed)

        return F.tanh(self.fc_xtu(xtu_embed)), n_unique


def history_attention(current, history, history_lens):
    # current.shape = (batch_size, 2*n_hidden)
    # history.shape = (batch_size, n_unique_max, 2*n_hidden)
    # history_lens.shape = (batch_size,)

    batch_size, n_unique_max, n_hidden = history.shape

    # Compute logits using batch matrix multiplication
    logits = torch.einsum("ij,itj->it", current, history) / torch.sqrt(
        torch.tensor(n_hidden, dtype=torch.float32)
    )

    # Create a mask based on history_lens
    mask = torch.arange(n_unique_max, device=history_lens.device).unsqueeze(
        0
    ) >= history_lens.unsqueeze(1)

    # Apply mask by setting invalid positions to -inf
    logits.masked_fill_(mask, float("-inf"))

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
        xh: List[List[torch.Tensor]],
        th: List[List[torch.Tensor]],
        uh: List[List[int]],
    ):
        c_enc = self.c_encoder(xc, tc)
        h_enc, h_lens = self.h_encoder(xh, th, uh)
        hc_attn = history_attention(c_enc, h_enc, h_lens)

        # Return the weighted sum of history with attention applied
        h_context = torch.einsum("it,itj->ij", hc_attn, h_enc)

        return self.clf(torch.cat([c_enc, h_context], dim=-1))

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        uc: torch.Tensor,
        xh: List[torch.Tensor],
        th: List[torch.Tensor],
        uh: List[torch.Tensor],
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

        with torch.inference_mode():
            return self(xc, tc, xh, th, uh)
