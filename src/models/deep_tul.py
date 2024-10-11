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
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_locs = n_locs
        self.n_times = n_times
        self.n_layers = n_layers
        self.embedding_dim_loc = embedding_dim_loc
        self.embedding_dim_time = embedding_dim_time
        self.embedding_dim = embedding_dim_time + embedding_dim_loc

        self.lstm = nn.LSTM(
            self.embedding_dim, n_hidden, bidirectional=True, batch_first=True
        )
        self.loc_embed = nn.Embedding(n_locs, embedding_dim_loc)
        self.time_embed = nn.Embedding(n_times, embedding_dim_time)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        traj_lens = torch.tensor([len(xi) for xi in x])
        x_pad = pad_sequence(x, batch_first=True)
        t_pad = pad_sequence(t, batch_first=True)
        t_embed = self.time_embed(t_pad)
        x_embed = self.loc_embed(x_pad)
        traj_embed = torch.cat([x_embed, t_embed], dim=-1)

        # Encode trajectory
        traj_enc = self.fc_enc(traj_embed)
        x_packed = pack_padded_sequence(
            traj_enc, traj_lens, batch_first=True, enforce_sorted=False
        )
        out, (h, c) = self.encoder(x_packed)  # out.shape = (batch, seq, 2*n_hidden)
        out_unpacked, lens_unpacked = pad_packed_sequence(out, batch_first=True)
        idcs = (lens_unpacked - 1).view(1, -1, 1)
        return out_unpacked.gather(
            dim=1, index=idcs.expand(-1, 1, out_unpacked.shape[-1])
        ).squeeze(
            1
        )  # return shape = (batch, 2*n_hidden)


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        n_hidden,
        embedding_dim_loc,
        embedding_dim_time,
        embedding_dim_user,
        n_locs,
        n_times,
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

        self.loc_embed = nn.Embedding(n_locs, embedding_dim_loc)
        self.time_embed = nn.Embedding(n_times, embedding_dim_time)
        self.user_embed = nn.Embedding(n_times, embedding_dim_user)
        self.fc_xtu = nn.Linear(self.embedding_dim, 2 * n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, t, u):
        # x.shape = [(seq_len, 1) for b in batch_size for n in n_traces]

        for xis, tis, uis in zip(x, t, u):
            xi = torch.cat(xis)  # shape = (sum(seq_len), 1)
            ti = torch.cat(tis)  # shape = (sum(seq_len), 1)
            ui = torch.cat(uis)  # shape = (sum(seq_len), 1)

            # Step 1: Get unique combinations of location and time IDs
            xti = torch.stack([xi, ti], dim=-1)  # shape = (sum(seq_len), 2)
            xt_unique, inv_idcs, counts = torch.unique(
                xti, dim=0, return_inverse=True, counts=True
            )
            x_unique, t_unique = xt_unique.unbind(dim=-1)
            # x_unique.shape = (n_unique, 1)

            # Step 2: Embed locations and times
            x_embed = self.loc_embed(x_unique)
            # x_embed.shape = (n_unique, loc_embedding_dim)
            t_embed = self.time_embed(t_unique)
            # x_embed.shape = (n_unique, time_embedding_dim)

            # Step 3: Aggregate user embeddings for each unique (loc, time) pair
            u_emb_sum = torch.zeros(len(xt_unique), self.embedding_dim_user)
            ui_embed = self.user_embed(ui)

            # Use scatter_add_ to efficiently add u_embeds to emb_sum
            u_emb_sum = u_emb_sum.scatter_add_(
                0, inv_idcs.unsqueeze(-1).expand(-1, self.embedding_dim_user), ui_embed
            )
            u_embed = u_emb_sum / counts[:, None]

            # Step 4: Combine embeddings
            xtu_embed = torch.cat([x_embed, t_embed, u_embed], dim=-1)

            return self.fc_xtu(xtu_embed)
