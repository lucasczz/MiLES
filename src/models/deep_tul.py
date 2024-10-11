from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence


class CurrentEncoder(nn.Module):
    def __init__(self, n_hidden, embedding_dim, n_locs, n_times, dropout):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_locs = n_locs
        self.n_times = n_times
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            embedding_dim, n_hidden, bidirectional=True, batch_first=True
        )
        self.loc_embed = nn.Embedding(n_locs, embedding_dim)
        self.time_embed = nn.Embedding(n_times, embedding_dim)
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
        out, (h, c) = self.encoder(x_packed)  # out.shape = (batch, seq, n_hidden)
        out_unpacked, lens_unpacked = pad_packed_sequence(out, batch_first=True)
        return out_unpacked.gather(dim=1, index=lens_unpacked) # return shape = (batch, )
