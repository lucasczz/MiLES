import math
import numpy as np
from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F


def _init_level_proportional(embedding_list):
    counts = np.array([7 ** (i + 1) for i in range(len(embedding_list))])
    weights = counts / counts.sum()
    with torch.no_grad():
        for weight, embedding in zip(weights, embedding_list):
            embedding.weight.data *= weight * 2


def _init_unit_variance(embedding_list):
    with torch.no_grad():
        for embedding in embedding_list:
            embedding.weight.data *= 1 / np.sqrt(len(embedding_list))


def _init_only_top(embedding_list):
    with torch.no_grad():
        for idx, embedding in embedding_list[:-1]:
            embedding.weight.data *= 0


class LookupSumEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings_loc: List[int],
        embedding_dim_loc: int,
        num_embeddings_time: List[int] = [],
        embedding_dim_time: int = None,
        weight_factor: int = 2,
        loc_level: Optional[int] = None,
    ):
        super().__init__()
        self.embedding_dim_loc = embedding_dim_loc

        if loc_level:
            self.loc_embedding = nn.Embedding(
                num_embeddings=num_embeddings_loc[loc_level],
                embedding_dim=embedding_dim_loc,
            )
        else:
            self.loc_embedding = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim_loc)
                    for n in num_embeddings_loc
                ]
            )
            loc_level_weights = np.logspace(
                0,
                len(num_embeddings_loc) - 1,
                len(num_embeddings_loc),
                base=weight_factor,
            )
            loc_level_weights /= loc_level_weights.sum()
            with torch.no_grad():
                for loc_level_mult, embedding in zip(
                    loc_level_weights, self.loc_embedding
                ):
                    embedding.weight.data *= loc_level_mult

        self.time_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim_time)
                for n in num_embeddings_time
            ]
        )
        self.dim = (
            embedding_dim_loc + embedding_dim_time
            if len(num_embeddings_time) > 0
            else embedding_dim_loc
        )
        self.loc_level = loc_level

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x.shape = (batch_size, max_seq_length, n_loc_features)
        # loc_features = [POI, cell0, cell1, ...]
        # t.shape = (batch_size, max_seq_length, n_time_features)
        # loc_features = [hour, 6h, day, weekend, timestamp]
        if self.loc_level:
            x_embedded = self.loc_embedding(x[..., self.loc_level])
        else:
            x_embedded = torch.stack(
                [
                    embedding(x[..., level])
                    for level, embedding in enumerate(self.loc_embedding)
                ]
            ).sum(0)
        if self.time_embedding:
            t_embedded = torch.stack(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ]
            ).sum(0)
            # t_embedded = rearrange(t_embedded, "l b s d -> b s (l d)")
            return torch.cat([x_embedded, t_embedded], dim=-1)
        else:
            return x_embedded


class LookupConcatEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings_loc: List[int],
        embedding_dim_loc: int,
        num_embeddings_time: List[int] = [],
        embedding_dim_time: int = None,
        weight_factor: int = 2,
        loc_level: Optional[int] = None,
    ):
        super().__init__()
        self.embedding_dim_loc = embedding_dim_loc
        self.loc_level = loc_level

        # Determine active location embeddings
        if loc_level is not None and loc_level >= 0:
            self.active_levels = [loc_level]
        elif loc_level is not None and loc_level < 0:
            self.active_levels = [
                i for i in range(len(num_embeddings_loc)) if i != -loc_level - 1
            ]
        else:
            self.active_levels = list(range(len(num_embeddings_loc)))

        # Compute embedding dimensions for location levels
        loc_level_weights = np.array(
            [weight_factor**-i for i in self.active_levels], dtype=float
        )
        loc_level_weights /= loc_level_weights.sum()
        loc_level_dims = (loc_level_weights * embedding_dim_loc).astype(int)
        loc_level_dims[0] = embedding_dim_loc - loc_level_dims[1:].sum()

        self.loc_embedding = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=num_embeddings_loc[level], embedding_dim=l_dim
                )
                for level, l_dim in zip(self.active_levels, loc_level_dims)
            ]
        )

        # Compute embedding dimensions for time levels
        time_level_weights = np.logspace(
            0,
            1 - len(num_embeddings_time),
            len(num_embeddings_time),
            base=weight_factor,
        )
        time_level_weights /= time_level_weights.sum()
        time_level_dims = (time_level_weights * embedding_dim_time).astype(int)
        time_level_dims[0] = embedding_dim_time - time_level_dims[1:].sum()

        self.time_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n, embedding_dim=t_dim)
                for n, t_dim in zip(num_embeddings_time, time_level_dims)
            ]
        )

        # Update final dimension
        self.dim = int(sum(loc_level_dims) + embedding_dim_time)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x.shape = (batch_size, max_seq_length, n_loc_features)
        # loc_features = [POI, cell0, cell1, ...]
        # t.shape = (batch_size, max_seq_length, n_time_features)
        # time_features = [hour, 6h, day, weekend, timestamp]

        x_embedded = torch.concat(
            [
                embedding(x[..., level])
                for embedding, level in zip(self.loc_embedding, self.active_levels)
            ],
            dim=-1,
        )

        if self.time_embedding:
            t_embedded = torch.concat(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ],
                dim=-1,
            )
            return torch.cat([x_embedded, t_embedded], dim=-1)
        else:
            return x_embedded


class LookupWeightedConcatEmbedding(LookupConcatEmbedding):
    def __init__(
        self,
        num_embeddings_loc: List[int],
        embedding_dim_loc: int,
        num_embeddings_time: List[int] = [],
        embedding_dim_time: int = None,
        weight_factor: int = 2,
        loc_level: Optional[int] = None,
    ):
        super().__init__(
            num_embeddings_loc,
            embedding_dim_loc,
            num_embeddings_time,
            embedding_dim_time,
            weight_factor,
            loc_level,
        )
        self.weights = torch.nn.Parameter(
            torch.ones(len(self.active_levels), dtype=torch.float32), requires_grad=True
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x.shape = (batch_size, max_seq_length, n_loc_features)
        # loc_features = [POI, cell0, cell1, ...]
        # t.shape = (batch_size, max_seq_length, n_time_features)
        # time_features = [hour, 6h, day, weekend, timestamp]

        x_embedded = torch.concat(
            [
                embedding(x[..., level]) * weight
                for weight, embedding, level in zip(
                    self.weights, self.loc_embedding, self.active_levels
                )
            ],
            dim=-1,
        )

        if self.time_embedding:
            t_embedded = torch.concat(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ],
                dim=-1,
            )
            return torch.cat([x_embedded, t_embedded], dim=-1)
        else:
            return x_embedded


class LookupWeightedSumEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings_loc: List[int],
        embedding_dim_loc: int,
        num_embeddings_time: List[int] = [],
        embedding_dim_time: int = None,
        weight_factor: int = 2,
    ):
        super().__init__()
        self.embedding_dim_loc = embedding_dim_loc

        self.loc_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim_loc)
                for n in num_embeddings_loc
            ]
        )

        self.time_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim_time)
                for n in num_embeddings_time
            ]
        )
        counts = torch.ones(len(num_embeddings_loc), dtype=torch.float32)
        self.x_weights = nn.Parameter(counts, requires_grad=True)
        self.t_weights = nn.Parameter(
            torch.ones(len(num_embeddings_time), dtype=torch.float32),
            requires_grad=True,
        )
        self.dim = (
            embedding_dim_loc + embedding_dim_time
            if self.time_embedding
            else embedding_dim_loc
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x.shape = (batch_size, max_seq_length, n_loc_features)
        # t.shape = (batch_size, max_seq_length, n_time_features)
        # loc_features = [POI, cell0, cell1, ...]
        # loc_features = [hour, 3h, 6h, day, weekend, timestamp]
        x_embedded = torch.stack(
            [
                embedding(x[..., level])
                for level, embedding in enumerate(self.loc_embedding)
            ],
        )
        x_embedded = torch.einsum("lbsj,l->bsj", x_embedded, self.x_weights)

        if self.time_embedding:
            t_embedded = torch.stack(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ]
            )
            t_embedded = torch.einsum("lbsj,l->bsj", t_embedded, self.t_weights)
            return torch.cat([x_embedded, t_embedded], dim=-1)
        else:
            return x_embedded


class CosineEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings_loc: List[int],
        embedding_dim_loc: int,
        num_embeddings_time: List[int],
        embedding_dim_time: int,
    ):
        super().__init__()
        self.embedding_dim_loc = embedding_dim_loc
        self.embedding_dim_time = (
            embedding_dim_time if embedding_dim_time else embedding_dim_loc // 4
        )
        self.ts_embedding = nn.Linear(in_features=1, out_features=embedding_dim_loc)
        self.fc = nn.Linear
        with torch.no_grad():
            self.ts_embedding.weight.copy_(
                torch.logspace(0, -9, embedding_dim_loc).unsqueeze(-1)
            )
            self.ts_embedding.bias.fill_(0)
        self.loc_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim_loc)
                for n in num_embeddings_loc
            ]
        )
        self.time_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n, embedding_dim=self.embedding_dim_time)
                for n in num_embeddings_time
            ]
        )
        self.fc = nn.Linear(
            embedding_dim_loc + self.embedding_dim_time, embedding_dim_loc
        )
        self.div = math.sqrt(1 / embedding_dim_loc)
        self.dim = embedding_dim_loc

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        ts = self.ts_embedding(t[..., -1, None].to(torch.float))
        ts_embedded = torch.cos(ts) * self.div
        x_embedded = torch.stack(
            [
                embedding(x[..., level])
                for level, embedding in enumerate(self.loc_embedding)
            ]
        ).sum(0)
        t_embedded = torch.stack(
            [
                embedding(t[..., level])
                for level, embedding in enumerate(self.time_embedding)
            ]
        ).sum(0)
        xt_embedded = torch.cat([x_embedded, t_embedded], dim=-1)
        xt_embedded = self.fc(xt_embedded)
        return F.tanh(xt_embedded) + ts_embedded


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: torch.Tensor):
        freqs = torch.einsum("ij,k->ijk", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(t.device)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings_loc: List[int],
        embedding_dim_loc: int,
        num_embeddings_time: List[int] = [],
        embedding_dim_time: int = None,
        rotary_base: int = 1e4,
    ):
        super().__init__()
        self.loc_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim_loc)
                for n in num_embeddings_loc
            ]
        )
        self.time_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim_time)
                for n in num_embeddings_time
            ]
        )
        self.ts_embedding = Rotary(embedding_dim_loc, rotary_base)
        self.dim = (
            embedding_dim_time * (len(num_embeddings_time) > 0) + embedding_dim_loc
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x_embedded = torch.stack(
            [
                embedding(x[..., level])
                for level, embedding in enumerate(self.loc_embedding)
            ]
        ).sum(0)
        cos, sin = self.ts_embedding(t[..., -1])
        x_embedded = apply_rotary_pos_emb(x_embedded, cos, sin)
        if self.time_embedding:
            t_embedded = torch.stack(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ]
            ).sum(0)
            return torch.cat([x_embedded, t_embedded], dim=-1)
        else:
            return x_embedded


EMBEDDING_TYPES = {
    "lookup_sum": LookupSumEmbedding,
    "lookup_concat": LookupConcatEmbedding,
    "lookup_weighted_concat": LookupWeightedConcatEmbedding,
    "lookup_weighted_sum": LookupWeightedSumEmbedding,
    "cosine": CosineEmbedding,
}
