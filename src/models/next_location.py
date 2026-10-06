import math
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from src.embedding import EMBEDDING_TYPES

LOG_2PI = math.log(2 * math.pi)


def gaussian_nll(x, y, mu_x, mu_y, logvar_x, logvar_y, rho):
    """Manual implementation of bivariate normal log likelihood using log-variances."""
    import math

    # Standardize
    z_x = (x - mu_x) / torch.exp(0.5 * logvar_x)
    z_y = (y - mu_y) / torch.exp(0.5 * logvar_y)

    # Calculate log determinant using log-variances
    log_det = logvar_x + logvar_y + math.log(1 - rho**2)

    exponent = -0.5 / (1 - rho**2) * (z_x**2 + z_y**2 - 2 * rho * z_x * z_y)

    log_lik = -LOG_2PI - 0.5 * log_det + exponent
    return log_lik


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


class Transformer(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_users: int,
        n_times: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 128,
        time_embedding_dim: int = 128,
        user_embedding_dim: int = 64,
        day_embedding_dim: int = 64,
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
            loc_level=loc_level,
        )
        self.pos_embedding = PositionalEmbedding(
            d_model=self.embedding.dim, max_len=500
        )
        if user_embedding_dim > 0:
            self.u_embedding = nn.Embedding(n_users, user_embedding_dim)
        else:
            self.u_embedding = None
        if day_embedding_dim > 0:
            self.day_embedding = nn.Embedding(8, day_embedding_dim)
        else:
            self.day_embedding = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding.dim, nhead=n_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.fc = nn.Linear(
            self.embedding.dim + user_embedding_dim + day_embedding_dim,
            2,
        )

    def forward(
        self,
        x_padded: torch.Tensor,
        t_padded: torch.Tensor,
        ll_padded: torch.Tensor,
        u: torch.Tensor,
    ):
        # Get transformer encoding
        # x_padded = F.pad(x_padded, (0, 0, 1, 0, 0, 0), value=0)
        # t_padded = F.pad(t_padded, (0, 0, 1, 0, 0, 0), value=0)
        padding_mask = x_padded[..., 0] == -1
        xt_embedded = self.embedding(x_padded, t_padded, ll_padded)
        xt_embedded = self.pos_embedding(xt_embedded)
        # x_embedded.shape = (batch_size, max_seq_length, embedding_dim)
        x_encoded = self.transformer_encoder(
            xt_embedded, src_key_padding_mask=padding_mask
        )
        x_encoded = x_encoded.mean(dim=1)

        meta_embeddings = []
        if self.u_embedding is not None:
            meta_embeddings.append(self.u_embedding(u))
        if self.day_embedding is not None:
            meta_embeddings.append(self.day_embedding(t_padded[:, 0, 3]))

        xu = torch.cat([x_encoded, *meta_embeddings], dim=-1)

        return torch.sigmoid(self.fc(xu))

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        llc: List[torch.Tensor],
        uc: torch.Tensor,
        ll_target: torch.Tensor,
        **kwargs,
    ):
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        llc_padded = pad_sequence(llc, batch_first=True).to(self.device)
        uc = uc.to(self.device)
        preds = self(xc_padded, tc_padded, llc_padded, uc)
        loss = F.mse_loss(preds, ll_target.to(self.device))
        return loss

    def pred_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        llc: List[torch.Tensor],
        uc: torch.Tensor,
        **kwargs,
    ):
        xc_padded = pad_sequence(xc, batch_first=True, padding_value=-1).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True, padding_value=-1).to(self.device)
        llc_padded = pad_sequence(llc, batch_first=True).to(self.device)
        uc = uc.to(self.device)
        logits = self(xc_padded, tc_padded, llc_padded, uc)
        return logits


class MLP(nn.Module):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 1024,
        time_embedding_dim: int = 64,
        user_embedding_dim: int = 64,
        day_embedding_dim: int = 64,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout: float = 0.2,
        loc_level: int = None,
        device: torch.device = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        self.n_locs = n_locs
        self.n_users = n_users
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Same embedding as RNN
        self.embedding = EMBEDDING_TYPES[embedding_type](
            num_embeddings_loc=n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=n_times,
            embedding_dim_time=time_embedding_dim,
            weight_factor=embedding_weight_factor,
            loc_level=loc_level,
        )

        if user_embedding_dim > 0:
            self.u_embedding = nn.Embedding(n_users, user_embedding_dim)
        else:
            self.u_embedding = None

        if day_embedding_dim > 0:
            self.day_embedding = nn.Embedding(8, day_embedding_dim)
        else:
            self.day_embedding = None

        # Final classifier with hidden layer
        classifier_input_dim = self.embedding.dim
        layers = []
        input_dim = classifier_input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(input_dim, n_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = n_hidden
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_clf = nn.Linear(input_dim + user_embedding_dim + day_embedding_dim, 2)

    def forward(
        self,
        x_pad: torch.Tensor,
        t_pad: torch.Tensor,
        ll_pad: torch.Tensor,
        u: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        # Embedding
        x_embedded = self.embedding(x_pad, t_pad, ll_pad)  # [batch, seq_len, embed_dim]

        # Meta embeddings
        meta_embeddings = []
        if self.u_embedding is not None:
            meta_embeddings.append(self.u_embedding(u))
        if self.day_embedding is not None:
            meta_embeddings.append(self.day_embedding(t_pad[:, 0, 3]))

        # Global average pooling
        pooled = x_embedded.mean(dim=1)

        # Hidden layer
        hidden = self.hidden_layers(pooled)
        hidden = self.dropout(hidden)

        # Concatenate with meta embeddings
        final_features = torch.cat([hidden, *meta_embeddings], dim=-1)

        # Final prediction
        y_pred = torch.sigmoid(self.fc_clf(final_features))

        return y_pred

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        llc: List[torch.Tensor],
        ll_target: torch.Tensor,
        uc: torch.Tensor,
        **kwargs,
    ):
        seq_lengths = torch.tensor([len(xci) for xci in xc])
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        llc_padded = pad_sequence(llc, batch_first=True).to(self.device)
        uc = uc.to(self.device)
        logits = self(xc_padded, tc_padded, llc_padded, uc, seq_lengths)
        loss = F.mse_loss(logits, ll_target.to(self.device))
        return loss

    def pred_step(
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
        uc = uc.to(self.device)
        logits = self(xc_padded, tc_padded, llc_padded, uc, seq_lengths)
        return logits


class CNN(nn.Module):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 1024,
        time_embedding_dim: int = 64,
        user_embedding_dim: int = 64,
        day_embedding_dim: int = 64,
        n_filters: int = 256,
        kernel_size: int = 3,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout: float = 0.2,
        loc_level: int = None,
        device: torch.device = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        self.n_locs = n_locs
        self.n_users = n_users
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Same embedding as RNN
        self.embedding = EMBEDDING_TYPES[embedding_type](
            num_embeddings_loc=n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=n_times,
            embedding_dim_time=time_embedding_dim,
            weight_factor=embedding_weight_factor,
            loc_level=loc_level,
        )

        if user_embedding_dim > 0:
            self.u_embedding = nn.Embedding(n_users, user_embedding_dim)
        else:
            self.u_embedding = None

        if day_embedding_dim > 0:
            self.day_embedding = nn.Embedding(8, day_embedding_dim)
        else:
            self.day_embedding = None

        # Multiple 1D CNN layers
        self.conv_layers = nn.ModuleList()

        # First conv layer
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=self.embedding.dim,
                out_channels=n_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        # Additional conv layers
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )

        # Global max pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final classifier with hidden layer
        classifier_input_dim = n_filters + user_embedding_dim + day_embedding_dim
        self.fc_hidden = nn.Linear(classifier_input_dim, n_hidden)
        self.fc_clf = nn.Linear(n_hidden, 2)

    def forward(
        self,
        x_pad: torch.Tensor,
        t_pad: torch.Tensor,
        ll_pad: torch.Tensor,
        u: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        # Embedding
        x_embedded = self.embedding(x_pad, t_pad, ll_pad)  # [batch, seq_len, embed_dim]

        # Meta embeddings
        meta_embeddings = []
        if self.u_embedding is not None:
            meta_embeddings.append(self.u_embedding(u))
        if self.day_embedding is not None:
            meta_embeddings.append(self.day_embedding(t_pad[:, 0, 3]))

        # Conv1d expects [batch, channels, seq_len]
        x_conv = x_embedded.transpose(1, 2)

        # Apply conv layers
        for conv in self.conv_layers:
            x_conv = F.relu(conv(x_conv))
            x_conv = self.dropout(x_conv)

        # Global average pooling
        pooled = self.global_pool(x_conv).squeeze(-1)  # [batch, n_filters]

        # Concatenate with meta embeddings
        final_features = torch.cat([pooled, *meta_embeddings], dim=-1)

        # Hidden layer
        hidden = F.relu(self.fc_hidden(final_features))
        hidden = self.dropout(hidden)

        # Final prediction
        y_pred = torch.sigmoid(self.fc_clf(hidden))

        return y_pred

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        llc: List[torch.Tensor],
        ll_target: torch.Tensor,
        uc: torch.Tensor,
        **kwargs,
    ):
        seq_lengths = torch.tensor([len(xci) for xci in xc])
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        llc_padded = pad_sequence(llc, batch_first=True).to(self.device)
        uc = uc.to(self.device)
        logits = self(xc_padded, tc_padded, llc_padded, uc, seq_lengths)
        loss = F.mse_loss(logits, ll_target.to(self.device))
        return loss

    def pred_step(
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
        uc = uc.to(self.device)
        logits = self(xc_padded, tc_padded, llc_padded, uc, seq_lengths)
        return logits


class RNN(nn.Module):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 1024,
        time_embedding_dim: int = 64,
        user_embedding_dim: int = 64,
        day_embedding_dim: int = 64,
        n_hidden: int = 1024,
        n_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0,
        loc_level: int = None,
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

        self.embedding = EMBEDDING_TYPES[embedding_type](
            num_embeddings_loc=n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=n_times,
            embedding_dim_time=time_embedding_dim,
            weight_factor=embedding_weight_factor,
            loc_level=loc_level,
        )
        if user_embedding_dim > 0:
            self.u_embedding = nn.Embedding(n_users, user_embedding_dim)
        else:
            self.u_embedding = None
        if day_embedding_dim > 0:
            self.day_embedding = nn.Embedding(8, day_embedding_dim)
        else:
            self.day_embedding = None

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
        self.fc_clf = nn.Linear(
            self.n_dirs * n_hidden + user_embedding_dim + day_embedding_dim,
            2,
        )

    def forward(
        self,
        x_pad: torch.Tensor,
        t_pad: torch.Tensor,
        ll_pad: torch.Tensor,
        u: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        x_embedded = self.embedding(x_pad, t_pad, ll_pad)
        meta_embeddings = []
        if self.u_embedding is not None:
            meta_embeddings.append(self.u_embedding(u))
        if self.day_embedding is not None:
            meta_embeddings.append(self.day_embedding(t_pad[:, 0, 3]))

        # Encode trajectory
        x_packed = pack_padded_sequence(
            x_embedded, seq_lens, batch_first=True, enforce_sorted=False
        )
        if self.rnn_type == "GRU":
            _, h = self.encoder(x_packed)
        else:
            _, (h, _) = self.encoder(x_packed)
        h = rearrange(h[-self.n_dirs :], "dirs batch hidden -> batch (dirs hidden)")
        hu = torch.cat([self.dropout(h), *meta_embeddings], dim=-1)
        # hu = self.dropout(h)
        y_pred = torch.sigmoid(self.fc_clf(hu))

        return y_pred

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        llc: List[torch.Tensor],
        ll_target: torch.Tensor,
        uc: torch.Tensor,
        **kwargs,
    ):
        seq_lengths = torch.tensor([len(xci) for xci in xc])
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        llc_padded = pad_sequence(llc, batch_first=True).to(self.device)
        uc = uc.to(self.device)
        logits = self(xc_padded, tc_padded, llc_padded, uc, seq_lengths)
        loss = F.mse_loss(logits, ll_target.to(self.device))
        return loss

    def pred_step(
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
        uc = uc.to(self.device)
        logits = self(xc_padded, tc_padded, llc_padded, uc, seq_lengths)
        return logits


class GRU(RNN):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 1024,
        time_embedding_dim: int = 64,
        user_embedding_dim: int = 64,
        n_hidden: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        loc_level: int = None,
        device: torch.device = "cuda:0",
        **kwargs,
    ):
        super().__init__(
            n_locs=n_locs,
            n_users=n_users,
            n_times=n_times,
            embedding_type=embedding_type,
            embedding_weight_factor=embedding_weight_factor,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
            rnn_type="GRU",
            device=device,
            bidirectional=False,
            loc_level=loc_level,
        )


class BiLSTM(RNN):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 1024,
        time_embedding_dim: int = 64,
        user_embedding_dim: int = 64,
        day_embedding_dim: int = 64,
        n_hidden: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        loc_level: int = None,
        device: torch.device = "cuda:0",
        **kwargs,
    ):
        super().__init__(
            n_locs=n_locs,
            n_users=n_users,
            n_times=n_times,
            embedding_type=embedding_type,
            embedding_weight_factor=embedding_weight_factor,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            day_embedding_dim=day_embedding_dim,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
            rnn_type="LSTM",
            device=device,
            bidirectional=True,
            loc_level=loc_level,
        )


class LSTM(RNN):
    def __init__(
        self,
        n_locs: List[int],
        n_times: List[int],
        n_users: int,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 1024,
        time_embedding_dim: int = 64,
        user_embedding_dim: int = 64,
        n_hidden: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        loc_level: int = None,
        device: torch.device = "cuda:0",
        **kwargs,
    ):
        super().__init__(
            n_locs=n_locs,
            n_users=n_users,
            n_times=n_times,
            embedding_type=embedding_type,
            embedding_weight_factor=embedding_weight_factor,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
            rnn_type="LSTM",
            device=device,
            bidirectional=False,
            loc_level=loc_level,
        )
