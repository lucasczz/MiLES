from torch.nn.utils.rnn import pad_sequence
from typing import List
import math
from torch import nn
from src.embedding import EMBEDDING_TYPES
import torch.nn.functional as F
import torch


class TULHOR(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_users: int,
        n_times: int,
        embedding_type: str = "lookup_sum",
        loc_embedding_dim: int = 128,
        time_embedding_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 16,
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
        self.fc = nn.Linear(self.embedding.dim, n_users)

    def forward(
        self,
        x_padded: torch.Tensor,
        t_padded: torch.Tensor,
    ):
        # Get transformer encoding
        # x_padded = F.pad(x_padded, (0, 0, 1, 0, 0, 0), value=0)
        # t_padded = F.pad(t_padded, (0, 0, 1, 0, 0, 0), value=0)
        padding_mask = x_padded[..., 0] == -1
        xt_embedded = self.embedding(x_padded, t_padded)
        xt_embedded = self.pos_embedding(xt_embedded)
        # x_embedded.shape = (batch_size, max_seq_length, embedding_dim)
        x_encoded = self.transformer_encoder(
            xt_embedded, src_key_padding_mask=padding_mask
        )
        x_encoded = x_encoded.mean(dim=1)

        return self.fc(x_encoded)

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        llc: List[torch.Tensor],
        uc: torch.Tensor,
        **kwargs,
    ):
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        preds = self(xc_padded, tc_padded)
        loss = F.cross_entropy(preds, uc.to(self.device))
        return loss

    def pred_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        llc: List[torch.Tensor],
        **kwargs,
    ):
        xc_padded = pad_sequence(xc, batch_first=True, padding_value=-1).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True, padding_value=-1).to(self.device)
        logits = self(xc_padded, tc_padded)
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
