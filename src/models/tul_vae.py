import torch
from torch import nn
import torch.nn.functional as F


class HierarchicEncoder(nn.Model):
    def __init__(
        self,
        n_hidden,
        embedding_dim_loc,
        embedding_dim_time,
        n_locs,
        n_times,
        dropout,
        n_layers,
        timesteps_split,
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
        self.timesteps_split = timesteps_split
        self.embedding_dim = embedding_dim_time + embedding_dim_loc

        self.micro_encoder = nn.LSTM(
            self.embedding_dim,
            n_hidden,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.macro_encoder = nn.LSTM(
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
        # x.shape = (batch_size, seq_len[i])
        # t.shape = (batch_size, seq_len[i])
        t0 = t[:, 0, None]
        delta_t = t - t0
