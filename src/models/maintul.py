import math
from typing import List
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def masked_max_pool(
    encoder_output: torch.Tensor, seq_lengths: torch.Tensor
) -> torch.Tensor:
    """
    Compute maximum over sequence length dimension while respecting sequence lengths.

    Args:
        encoder_output: Tensor of shape (batch_size, max_seq_len, model_dim)
        seq_lengths: Tensor of shape (batch_size,) containing actual sequence lengths

    Returns:
        Tensor of shape (batch_size, model_dim) containing the maximum values
    """
    batch_size, max_seq_len, model_dim = encoder_output.shape

    # Create mask: (batch_size, max_seq_len)
    mask = torch.arange(max_seq_len, device=encoder_output.device).unsqueeze(
        0
    ) >= seq_lengths.unsqueeze(-1).to(encoder_output.device)

    # Set padding positions to negative infinity
    masked_output = encoder_output.masked_fill(mask.unsqueeze(-1), float("-inf"))

    # Take maximum over sequence length dimension (dim=1)
    pooled = torch.max(masked_output, dim=1)[0]

    return pooled


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_times: int,
        n_users: int,
        n_hidden: int,
        n_layers: int,
        embedding_dim: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
        device: torch.device = "cuda:0",
    ):
        super().__init__()
        self.loc_embedding_dim = embedding_dim
        self.time_embedding_dim = embedding_dim // 4
        self.n_locs = n_locs
        self.n_times = n_times
        self.n_dirs = 1 + bidirectional
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=n_hidden,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.loc_embedding = nn.Embedding(n_locs + 1, self.loc_embedding_dim)
        self.time_embedding = nn.Embedding(n_locs + 1, self.time_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_embedding = nn.Linear(
            in_features=self.time_embedding_dim + self.loc_embedding_dim,
            out_features=embedding_dim,
        )
        self.fc_out = nn.Linear(n_hidden, n_users)
        self.device = device

    def forward(
        self, x_padded: torch.Tensor, t_padded: torch.Tensor, traj_lengths: torch.Tensor
    ):
        t_embed = self.time_embedding(t_padded)
        x_embed = self.loc_embedding(x_padded)
        xt_embed = torch.cat([x_embed, t_embed], dim=-1)
        xt_enc = F.tanh(self.fc_embedding(xt_embed))
        xt_enc = self.dropout(xt_enc)

        xt_packed = pack_padded_sequence(
            xt_enc, traj_lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(xt_packed)  # out.shape = (batch, seq, 2*n_hidden)
        h = rearrange(h[-self.n_dirs :], "dirs batch hidden -> batch (dirs hidden)")
        return self.fc_out(h)


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_times: int,
        n_users: int,
        n_hidden: int,
        n_layers: int,
        n_heads: int,
        embedding_dim: int,
        dropout: float = 0.1,
        device: torch.device = torch.device,
    ):
        super().__init__()
        self.embedding = SpatioTemporalEmbedding(
            n_locs=n_locs,
            n_times=n_times,
            embedding_dim=embedding_dim,
            dropout=dropout,
            device=device,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=n_hidden,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embedding_dim, eps=1e-6),
        )
        self.fc_out = nn.Linear(embedding_dim, n_users)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ts: torch.Tensor,
        seq_lengths: torch.Tensor,
    ):
        padding_mask = x == 0
        xtts_embedded = self.embedding(x, t, ts)
        xtts_encoded = self.encoder(xtts_embedded, src_key_padding_mask=padding_mask)
        # xtts_enc.shape = (batch_size, max_seq_length, embedding_dim)
        xtts_pooled = masked_max_pool(xtts_encoded, seq_lengths)
        return self.fc_out(xtts_pooled)


class SpatioTemporalEmbedding(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_times: int,
        embedding_dim: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ts_embedding = nn.Linear(in_features=1, out_features=embedding_dim)
        self.fc = nn.Linear
        with torch.no_grad():
            self.ts_embedding.weight.copy_(
                torch.logspace(0, -9, embedding_dim).unsqueeze(-1)
            )
            self.ts_embedding.bias.fill_(0)
        self.device = device
        self.loc_embedding = nn.Embedding(n_locs + 1, embedding_dim)
        self.time_embedding = nn.Embedding(n_times + 1, embedding_dim // 4)
        self.fc = nn.Linear(embedding_dim + embedding_dim // 4, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.div = math.sqrt(1 / embedding_dim)

    def forward(
        self, x_padded: torch.Tensor, t_padded: torch.Tensor, ts_padded: torch.Tensor
    ):
        """Embeds GPS trajectory information

        Parameters
        ----------
        x : torch.Tensor
            Sequence of location IDs.
        t : torch.Tensor
            Sequence of hours.
        ts : torch.Tensor
            Sequence of timestamps in seconds.

        Returns
        -------
        _type_
            _description_
        """
        ts_embedded = torch.cos(self.ts_embedding(ts_padded.unsqueeze(-1))) * self.div
        time_embedded = self.time_embedding(t_padded)
        loc_embedded = self.loc_embedding(x_padded)
        xt_embedded = torch.cat([loc_embedded, time_embedded], dim=-1)
        xt_embedded = self.fc(xt_embedded)
        return self.dropout(F.tanh(xt_embedded) + ts_embedded)


class MainTUL(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_users: int,
        n_times: int = 24,
        n_hidden: int = 1024,
        embedding_dim: int = 512,
        beta: float = 10,
        dis_temp: float = 10,
        dropout: float = 0.1,
        n_heads: int = 8,
        n_layers_teacher: int = 3,
        n_layers_student: int = 3,
        bidirectional: bool = False,
        device: torch.device = "cuda:0",
    ):
        super().__init__()
        self.n_locs = n_locs
        self.n_times = n_times
        self.n_users = n_users
        self.dis_temp = dis_temp
        self.embedding_dim = embedding_dim
        self.device = device
        self.beta = beta

        self.student = LSTMEncoder(
            n_locs=n_locs,
            n_times=n_times,
            n_users=n_users,
            n_layers=n_layers_student,
            n_hidden=n_hidden,
            embedding_dim=embedding_dim,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.teacher = TemporalTransformerEncoder(
            n_locs=n_locs,
            n_times=n_times,
            n_users=n_users,
            n_hidden=n_hidden,
            n_layers=n_layers_teacher,
            n_heads=n_heads,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(
        self,
        xc_padded: torch.Tensor,
        tc_padded: torch.Tensor,
        lengths_c: torch.Tensor,
        xh_padded: torch.Tensor,
        th_padded: torch.Tensor,
        ths_padded: torch.Tensor,
        lengths_h: torch.Tensor,
    ):
        out_student1 = self.student(xc_padded, tc_padded, lengths_c)
        out_teacher1 = self.teacher(xh_padded, th_padded, ths_padded, lengths_h)
        return out_student1, out_teacher1

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        uc: torch.Tensor,
        tcs: List[torch.Tensor],
        xh: List[torch.Tensor],
        th: List[torch.Tensor],
        ths: List[torch.Tensor],
        **kwargs
    ):
        lengths_c = torch.tensor([len(xci) for xci in xc])
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        tcs_padded = pad_sequence(tcs, batch_first=True).to(self.device)

        lengths_h = torch.tensor([len(xhi) for xhi in xh])
        xh_padded = pad_sequence(xh, batch_first=True).to(self.device)
        th_padded = pad_sequence(th, batch_first=True).to(self.device)
        ths_padded = pad_sequence(ths, batch_first=True).to(self.device)
        uc = uc.to(self.device)

        out_student1, out_teacher1 = self(
            xc_padded, tc_padded, lengths_c, xh_padded, th_padded, ths_padded, lengths_h
        )
        out_student2, out_teacher2 = self(
            xh_padded, th_padded, lengths_h, xc_padded, tc_padded, tcs_padded, lengths_c
        )
        ce_loss_student1 = F.cross_entropy(out_student1, uc)
        ce_loss_student2 = F.cross_entropy(out_student2, uc)

        ce_loss_teacher1 = F.cross_entropy(out_teacher1, uc)
        ce_loss_teacher2 = F.cross_entropy(out_teacher2, uc)

        loss_dis1 = compute_loss_dis(out_student1, out_teacher1, self.dis_temp)
        loss_dis2 = compute_loss_dis(out_student2, out_teacher2, self.dis_temp)
        loss = (
            ce_loss_student1
            + ce_loss_student2
            + ce_loss_teacher1
            + ce_loss_teacher2
            + self.beta * (loss_dis1 + loss_dis2)
        )
        return loss


def compute_loss_dis(student_output, teacher_output, temperature):
    prob_student = F.log_softmax(student_output / temperature, dim=1)
    prob_teacher = F.softmax(teacher_output / temperature, dim=1)
    return F.kl_div(prob_student, prob_teacher, reduction="batchmean") * (
        temperature**2
    )
