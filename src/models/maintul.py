from bisect import bisect
from typing import List, Literal
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from src.embedding import EMBEDDING_TYPES, CosineEmbedding


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
        embedding_type: str = "lookup_sum",
        loc_embedding_dim: int = 512,
        time_embedding_dim: int = 0,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.loc_embedding_dim = loc_embedding_dim
        self.n_locs = n_locs
        self.n_times = n_times
        self.n_dirs = 1 + bidirectional
        self.n_hidden = n_hidden
        self.embedding = EMBEDDING_TYPES[embedding_type](
            num_embeddings_loc=n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=n_times,
            embedding_dim_time=time_embedding_dim,
        )
        self.fc_embedding = nn.Linear(
            in_features=self.embedding.dim,
            out_features=loc_embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=loc_embedding_dim,
            hidden_size=n_hidden,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(n_hidden, n_users)

    def forward(
        self, x_padded: torch.Tensor, t_padded: torch.Tensor, traj_lengths: torch.Tensor
    ):
        xt_embed = self.embedding(x_padded, t_padded)
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
        loc_embedding_dim: int = 512,
        time_embedding_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = CosineEmbedding(
            num_embeddings_loc=n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=n_times,
            embedding_dim_time=time_embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding.dim,
            nhead=n_heads,
            dim_feedforward=n_hidden,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(self.embedding.dim, eps=1e-6),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(self.embedding.dim, n_users)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        seq_lengths: torch.Tensor,
    ):
        padding_mask = x[..., 0] == 0
        xtts_embedded = self.dropout(self.embedding(x, t))
        xtts_encoded = self.encoder(xtts_embedded, src_key_padding_mask=padding_mask)
        # xtts_enc.shape = (batch_size, max_seq_length, embedding_dim)
        xtts_pooled = masked_max_pool(xtts_encoded, seq_lengths)
        return self.fc_out(xtts_pooled)


class MainTUL(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_users: int,
        n_times: int = 24,
        n_hidden: int = 1024,
        embedding_type: str = "lookup_sum",
        loc_embedding_dim: int = 512,
        time_embedding_dim: int = 128,
        lambduh: float = 1,
        distill_temp: float = 10,
        dropout: float = 0.1,
        n_heads: int = 8,
        n_layers: int = 3,
        bidirectional: bool = False,
        augment_strategy: Literal["recent", "random"] = "random",
        n_augment: int = 16,
        device: torch.device = "cuda:0",
    ):
        super().__init__()
        self.n_locs = n_locs
        self.n_times = n_times
        self.n_users = n_users
        self.distill_temp = distill_temp
        self.embedding_dim = loc_embedding_dim
        self.device = device
        self.lambduh = lambduh
        self.n_augment = n_augment
        self.augment_strategy = augment_strategy

        self.student = LSTMEncoder(
            n_locs=n_locs,
            n_times=n_times,
            n_users=n_users,
            n_layers=n_layers,
            n_hidden=n_hidden,
            embedding_type=embedding_type,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.teacher = TemporalTransformerEncoder(
            n_locs=n_locs,
            n_times=n_times,
            n_users=n_users,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
        )

    def forward(
        self,
        xc_padded: torch.Tensor,
        tc_padded: torch.Tensor,
        lengths_c: torch.Tensor,
        xh_padded: torch.Tensor,
        th_padded: torch.Tensor,
        lengths_h: torch.Tensor,
    ):
        out_student1 = self.student(xc_padded, tc_padded, lengths_c)
        out_teacher1 = self.teacher(xh_padded, th_padded, lengths_h)
        return out_student1, out_teacher1

    def augment_trajectory(self, xc, tc, uc, xh, th, uh):
        # Determine indices where uh matches uc
        uc = uc.to(self.device)
        uh = uh.to(self.device)
        mask = uc[:, None] == uh[None, :]
        idcs = [mask_i.nonzero().view(-1) for mask_i in mask]

        # Shuffle only if augment strategy is "random"
        if self.augment_strategy == "random":
            for idx, idcs_i in enumerate(idcs):
                sample_idcs = torch.randperm(len(idcs_i))[: self.n_augment].sort()[0]
                idcs[idx] = idcs_i[sample_idcs]

        # Slice the last n_augment indices in case of "last" strategy
        elif len(idcs) > self.n_augment:
            idcs = [idcs_i[-self.n_augment :] for idcs_i in idcs]

        xhs = [[xh[idx] for idx in idcs_i] for idcs_i in idcs]
        ths = [[th[idx] for idx in idcs_i] for idcs_i in idcs]

        for xci, tci, xhi, thi, idcs_i in zip(xc, tc, xhs, ths, idcs):
            idx = bisect(thi, tci[-1, -1], key=lambda r: r[-1, -1])
            xhi.insert(idx, xci)
            thi.insert(idx, tci)

        return [torch.cat(xhi) for xhi in xhs], [torch.cat(thi) for thi in ths]

    def train_step(
        self,
        xc: List[torch.Tensor],
        tc: List[torch.Tensor],
        uc: torch.Tensor,
        xh: List[torch.Tensor] = [],
        th: List[torch.Tensor] = [],
        uh: torch.Tensor = torch.empty(0),
        **kwargs,
    ):
        # uh.shape = [(n_sequences, ) for _ in range(batch_size)]
        lengths_c = torch.tensor([len(xci) for xci in xc])
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)

        if len(uh) < 1:
            xh_user, th_user = self.augment_trajectory(xc, tc, uc, xh, th, uh)
        else:
            xh_user, th_user = xc, tc
        lengths_h = torch.tensor([len(xhi) for xhi in xh_user])
        # Pad the subsampled trajectories
        xh_padded = pad_sequence([x for x in xh_user], batch_first=True).to(self.device)
        th_padded = pad_sequence([t for t in th_user], batch_first=True).to(self.device)
        uc = uc.to(self.device)

        out_student1, out_teacher1 = self(
            xc_padded, tc_padded, lengths_c, xh_padded, th_padded, lengths_h
        )
        out_student2, out_teacher2 = self(
            xh_padded, th_padded, lengths_h, xc_padded, tc_padded, lengths_c
        )
        ce_loss_student1 = F.cross_entropy(out_student1, uc)
        ce_loss_student2 = F.cross_entropy(out_student2, uc)

        ce_loss_teacher1 = F.cross_entropy(out_teacher1, uc)
        ce_loss_teacher2 = F.cross_entropy(out_teacher2, uc)

        loss_dis1 = compute_loss_dis(out_student1, out_teacher1, self.distill_temp)
        loss_dis2 = compute_loss_dis(out_student2, out_teacher2, self.distill_temp)
        loss = (
            ce_loss_student1
            + ce_loss_student2
            + ce_loss_teacher1
            + ce_loss_teacher2
            + self.lambduh * (loss_dis1 + loss_dis2)
        )
        return loss

    def pred_step(self, xc: List[torch.Tensor], tc: List[torch.Tensor], **kwargs):
        lengths_c = torch.tensor([len(xci) for xci in xc])
        xc_padded = pad_sequence(xc, batch_first=True).to(self.device)
        tc_padded = pad_sequence(tc, batch_first=True).to(self.device)
        logits = self.student(xc_padded, tc_padded, lengths_c)
        return logits


def compute_loss_dis(student_output, teacher_output, temperature):
    prob_student = F.log_softmax(student_output / temperature, dim=1)
    prob_teacher = F.softmax(teacher_output / temperature, dim=1)
    return F.kl_div(prob_student, prob_teacher, reduction="batchmean") * (
        temperature**2
    )
