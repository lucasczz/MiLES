from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence


class CurrentEncoder(nn.Module):
    def __init__(
        self,
        n_hidden,
        embedding_dim_loc,
        embedding_dim_time,
        n_locs,
        n_times,
        dropout,
        n_layers,
        device,
    ):
        super().__init__()
        self.device = device
        self.n_hidden = n_hidden
        self.n_locs = n_locs
        self.n_times = n_times
        self.n_layers = n_layers
        self.embedding_dim_loc = embedding_dim_loc
        self.embedding_dim_time = embedding_dim_time
        self.embedding_dim = embedding_dim_time + embedding_dim_loc

        self.lstm = nn.LSTM(
            self.embedding_dim,
            n_hidden,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.loc_embed = nn.Embedding(n_locs+1, embedding_dim_loc, padding_idx=0)
        self.time_embed = nn.Embedding(n_times+1, embedding_dim_time, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        traj_lens = torch.tensor([len(xi) for xi in x])
        x_pad = pad_sequence(x, batch_first=True)
        t_pad = pad_sequence(t, batch_first=True)
        x_embed = self.loc_embed(x_pad)
        t_embed = self.time_embed(t_pad)
        x_embed = self.dropout(x_embed)
        t_embed = self.dropout(t_embed)
        xt_embed = torch.cat([x_embed, t_embed], dim=-1)

        # Encode trajectory
        x_packed = pack_padded_sequence(
            xt_embed, traj_lens, batch_first=True, enforce_sorted=False
        )
        out, (h, c) = self.lstm(x_packed)  # out.shape = (batch, seq, 2*n_hidden)
        out_unpacked, lens_unpacked = pad_packed_sequence(out, batch_first=True)

        return out_unpacked[torch.arange(len(out_unpacked)), lens_unpacked - 1]
        # shape = (batch_size, 2*n_hidden)


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        n_hidden,
        embedding_dim_loc,
        embedding_dim_time,
        embedding_dim_user,
        n_locs,
        n_times,
        n_users,
        dropout,
        n_layers,
        device,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_locs = n_locs
        self.n_times = n_times
        self.n_layers = n_layers
        self.embedding_dim_loc = embedding_dim_loc
        self.embedding_dim_time = embedding_dim_time
        self.embedding_dim_user = embedding_dim_user
        self.embedding_dim = embedding_dim_time + embedding_dim_loc + embedding_dim_user

        self.loc_embed = nn.Embedding(n_locs+1, embedding_dim_loc, padding_idx=0)
        self.time_embed = nn.Embedding(n_times+1, embedding_dim_time, padding_idx=0)
        self.user_embed = nn.Embedding(n_users+1, embedding_dim_user, padding_idx=0)
        self.fc_xtu = nn.Linear(self.embedding_dim, 2 * n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, t, u):
        # x.shape = [(len(seq), 1) for seq in seqs for seqs in x]
        all_xt_unique, all_inv_idcs, all_counts, all_u = [], [], [], []
        for xis, tis, uis in zip(x, t, u):
            xi = torch.cat(xis)  # shape = (sum(seq_len), 1)
            ti = torch.cat(tis)  # shape = (sum(seq_len), 1)
            ui = torch.cat(uis)  # shape = (sum(seq_len), 1)

            # Step 1: Get unique combinations of location and time IDs
            xti = torch.stack([xi, ti], dim=-1)  # shape = (sum(seq_len), 2)
            xt_unique, inv_idcs, counts = torch.unique(
                xti, dim=0, return_inverse=True, return_counts=True
            )
            all_xt_unique.append(xt_unique)
            all_inv_idcs.append(inv_idcs)
            all_counts.append(counts)
            all_u.append(ui)

        n_unique = torch.LongTensor([len(count) for count in all_counts])
        counts_padded = pad_sequence(all_counts, batch_first=True, padding_value=-1)
        xt_unique_padded = pad_sequence(all_xt_unique, batch_first=True)
        inv_idcs_padded = pad_sequence(all_inv_idcs, batch_first=True)
        u_padded = pad_sequence(all_u, batch_first=True)

        # Step 2: Embed locations and times
        x_embed = self.dropout(self.loc_embed(xt_unique_padded[..., 0]))
        t_embed = self.dropout(self.time_embed(xt_unique_padded[..., 1]))
        ui_embed = self.dropout(self.user_embed(u_padded))

        # Step 3: Aggregate user embeddings for each unique (loc, time) pair
        batch_size, n_unique_max, _ = xt_unique_padded.shape

        u_emb_sum = torch.zeros(batch_size, n_unique_max, self.embedding_dim_user)
        # Use scatter_add_ to efficiently add u_embeds to emb_sum
        u_emb_sum = u_emb_sum.scatter_add_(
            1,
            inv_idcs_padded.unsqueeze(-1).expand(-1, -1, self.embedding_dim_user),
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
    mask = torch.arange(n_unique_max).unsqueeze(0) >= history_lens.unsqueeze(1)

    # Apply mask by setting invalid positions to -inf
    logits.masked_fill_(mask, float("-inf"))

    # Compute attention weights using softmax
    attn = torch.softmax(logits, dim=-1)
    return attn


class DeepTUL(nn.Module):
    def __init__(
        self,
        n_hidden,
        embedding_dim_loc,
        embedding_dim_time,
        embedding_dim_user,
        n_locs,
        n_times,
        n_users,
        dropout,
        n_layers,
        device,
    ):
        super().__init__()
        self.c_encoder = CurrentEncoder(
            n_hidden,
            embedding_dim_loc,
            embedding_dim_time,
            n_locs,
            n_times,
            dropout,
            n_layers,
            device,
        )
        self.h_encoder = HistoryEncoder(
            n_hidden,
            embedding_dim_loc,
            embedding_dim_time,
            embedding_dim_user,
            n_locs,
            n_times,
            n_users,
            dropout,
            n_layers,
            device,
        )
        self.clf = nn.Linear(4 * n_hidden, n_users)

    def forward(self, xc, tc, xh, th, uh):
        c_enc = self.c_encoder(xc, tc)
        h_enc, h_lens = self.h_encoder(xh, th, uh)
        hc_attn = history_attention(c_enc, h_enc, h_lens)

        # Return the weighted sum of history with attention applied
        h_context = torch.einsum("it,itj->ij", hc_attn, h_enc)

        return self.clf(torch.cat([c_enc, h_context], dim=-1))
