import math
from typing import List
import torch
from torch import nn
import torch.nn.functional as F


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: torch.Tensor):
        freqs = torch.einsum("ij,k->ijk", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(t.device)
        return emb.cos()[None, ...], emb.sin()[None, ...]


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
                nn.Embedding(num_embeddings=n + 1, embedding_dim=embedding_dim_loc)
                for n in num_embeddings_loc
            ]
        )
        self.time_embedding = Rotary(embedding_dim_loc, rotary_base)
        self.dim = embedding_dim_loc

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x_embedded = torch.stack(
            [
                embedding(x[..., level])
                for level, embedding in enumerate(self.loc_embedding)
            ]
        ).sum(0)
        cos, sin = self.time_embedding(t[..., -1])
        x_embedded = apply_rotary_pos_emb(x_embedded, cos, sin)
        return x_embedded


class LookupEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings_loc: List[int],
        embedding_dim_loc: int,
        num_embeddings_time: List[int] = [],
        embedding_dim_time: int = None,
    ):
        super().__init__()
        self.embedding_dim_loc = embedding_dim_loc
        self.embedding_dim_time = (
            embedding_dim_time if embedding_dim_time else embedding_dim_loc // 4
        )
        self.loc_embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n + 1, embedding_dim=embedding_dim_loc)
                for n in num_embeddings_loc
            ]
        )
        self.time_embedding = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=n + 1, embedding_dim=self.embedding_dim_time
                )
                for n in num_embeddings_time
            ]
        )
        self.dim = self.embedding_dim_time + embedding_dim_loc

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x.shape = (batch_size, max_seq_length, n_loc_features)
        # loc_features = [POI, cell0, cell1, ...]
        # t.shape = (batch_size, max_seq_length, n_time_features)
        # loc_features = [hour, 6h, day, weekend, timestamp]
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
                nn.Embedding(num_embeddings=n + 1, embedding_dim=embedding_dim_loc)
                for n in num_embeddings_loc
            ]
        )
        self.time_embedding = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=n + 1, embedding_dim=self.embedding_dim_time
                )
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


EMBEDDING_TYPES = {
    "lookup": LookupEmbedding,
    "rotary": RotaryEmbedding,
    "cosine": CosineEmbedding,
}
