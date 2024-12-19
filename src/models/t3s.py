from typing import List
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch

from src.embedding import EMBEDDING_TYPES


class T3S(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_users: int,
        n_times: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 128,
        time_embedding_dim: int = 128,
        n_layers: int = 1,
        n_heads: int = 16,
        loc_level: int = None,
        device: torch.device = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.embedding = EMBEDDING_TYPES[embedding_type](
            num_embeddings_loc=n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=n_times,
            embedding_dim_time=time_embedding_dim,
            weight_factor=embedding_weight_factor,
            loc_level=loc_level
        )
        self.pos_embedding = PositionalEmbedding(
            d_model=self.embedding.dim, max_len=500
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding.dim, nhead=n_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.lstm_encoder = nn.LSTM(
            input_size=2, hidden_size=self.embedding.dim, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(self.embedding.dim, n_users)
        self.gamma = torch.nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x_padded: torch.Tensor,
        t_padded: torch.Tensor,
        ll_padded: torch.Tensor,
        seq_lengths: torch.Tensor,
    ):
        # Get transformer encoding
        padding_mask = x_padded[..., 0] == 0
        xt_embedded = self.embedding(x_padded, t_padded)
        xt_embedded = self.pos_embedding(xt_embedded)
        # x_embedded.shape = (batch_size, max_seq_length, embedding_dim)
        x_encoded = self.transformer_encoder(
            xt_embedded, src_key_padding_mask=padding_mask
        )
        x_encoded = x_encoded.mean(dim=1)

        # Get LSTM encoding
        ll_packed = pack_padded_sequence(
            ll_padded, seq_lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm_encoder(ll_packed)
        ll_encoded = h[-1]
        xll_encoded = x_encoded * self.gamma + ll_encoded * (1 - self.gamma)
        return self.fc(xll_encoded)

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        llc: List[torch.Tensor],
        uc: torch.Tensor,
        **kwargs,
    ):
        seq_lengths = torch.tensor([len(xci) for xci in xc])
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        llc_padded = pad_sequence(llc, batch_first=True).to(self.device)
        preds = self(xc_padded, tc_padded, llc_padded, seq_lengths)
        loss = F.cross_entropy(preds, uc.to(self.device))
        return loss

    def pred_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        llc: List[torch.Tensor],
        **kwargs,
    ):
        seq_lengths = torch.tensor([len(xci) for xci in xc])
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        llc_padded = pad_sequence(llc, batch_first=True).to(self.device)
        logits = self(xc_padded, tc_padded, llc_padded, seq_lengths)
        return logits


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
