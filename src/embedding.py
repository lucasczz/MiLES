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


class BaselineEmbedding(nn.Module):
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

        self.loc_embedding = nn.Embedding(
            num_embeddings=num_embeddings_loc[0], embedding_dim=embedding_dim_loc
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
        self.dim = int(embedding_dim_loc + embedding_dim_time)

    def forward(self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor):
        # x.shape = (batch_size, max_seq_length, n_loc_features)
        # loc_features = [POI, cell0, cell1, ...]
        # t.shape = (batch_size, max_seq_length, n_time_features)
        # time_features = [hour, 6h, day, weekend, timestamp]

        # Apply embeddings and weights in parallel
        x_embedded = self.loc_embedding(x[..., 0])

        # Repeat without weighting for time features. Note that we generally only use a single time feature.
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
            return torch.cat(x_embedded, dim=-1)


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

    def forward(self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor):
        # x.shape = (batch_size, max_seq_length, n_loc_features)
        # loc_features = [POI, cell0, cell1, ...]
        # t.shape = (batch_size, max_seq_length, n_time_features)
        # time_features = [hour, 6h, day, weekend, timestamp]
        loc_indices = x[..., self.active_levels]  # Single tensor operation

        # Apply embeddings and weights in parallel
        x_embedded = [
            emb(loc_indices[..., i]) for i, emb in enumerate(self.loc_embedding)
        ]

        # Repeat without weighting for time features. Note that we generally only use a single time feature.
        if self.time_embedding:
            t_embedded = torch.concat(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ],
                dim=-1,
            )
            return torch.cat([*x_embedded, t_embedded], dim=-1)
        else:
            return torch.cat(x_embedded, dim=-1)


class MilesEmbedding(nn.Module):
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
        self.weights = torch.nn.Parameter(
            torch.ones(len(self.active_levels), dtype=torch.float32), requires_grad=True
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor):
        """Computes MiLES embeddings, the input tensors must adhere to the following scheme:


        location_features = [POI, cell0, cell1, ...]
        x.shape = (batch_size, max_seq_length, len(location_features))
        t.shape = (batch_size, max_seq_length, n_time_features)

        For ease of implementation, we generate the features `cell0, cell1, ...` during data loading using `src.data.discretization.discretize_coordinates` instead of on the fly.
        """
        # Select embedding vectors, multiply them with learnable attention weights and concatenate them
        # Get all indices at once
        loc_indices = x[..., self.active_levels]  # Single tensor operation

        # Apply embeddings and weights in parallel
        x_embedded = [
            emb(loc_indices[..., i]) * w
            for i, (emb, w) in enumerate(zip(self.loc_embedding, self.weights))
        ]

        # Repeat without weighting for time features. Note that we generally only use a single time feature.
        if self.time_embedding:
            t_embedded = torch.concat(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ],
                dim=-1,
            )
            return torch.cat([*x_embedded, t_embedded], dim=-1)
        else:
            return torch.cat(x_embedded, dim=-1)


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

    def forward(self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor):
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

    def forward(self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor):
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

    def forward(self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor):
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


class SinusoidalLocationEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, scale: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 4
        self.inv_freq = (
            2
            * torch.pi
            * scale ** (torch.arange(0.0, self.half_dim, 1.0) / self.half_dim)
        )
        # self.inv_freq = torch.exp(
        #     -math.log(10000.0) * torch.arange(0.0, self.half_dim, 1.0) / self.half_dim
        # )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (batch, seq_len, 2) where the last dim is (lat, lon)
        """
        lat, lon = coords[..., 0], coords[..., 1]
        sin_lat = torch.sin(lat.unsqueeze(-1) * self.inv_freq.to(coords.device))
        cos_lat = torch.cos(lat.unsqueeze(-1) * self.inv_freq.to(coords.device))
        sin_lon = torch.sin(lon.unsqueeze(-1) * self.inv_freq.to(coords.device))
        cos_lon = torch.cos(lon.unsqueeze(-1) * self.inv_freq.to(coords.device))
        return torch.cat([sin_lat, cos_lat, sin_lon, cos_lon], dim=-1)  # (B, T, D)


class FourierFeatureEmbedding(nn.Module):
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
        self.active_levels = (
            list(range(len(num_embeddings_loc)))
            if loc_level is None
            else list(range(loc_level + 1))
        )

        self.poi_embedding = nn.Embedding(num_embeddings_loc[0], embedding_dim_loc // 2)

        self.time_embedding = (
            nn.ModuleList(
                [nn.Embedding(num, embedding_dim_time) for num in num_embeddings_time]
            )
            if embedding_dim_time
            else None
        )

        self.coord_embedding = SinusoidalLocationEmbedding(
            embedding_dim_loc // 2, weight_factor
        )
        self.dim = (
            embedding_dim_loc + embedding_dim_time
            if self.time_embedding
            else embedding_dim_loc
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, loc_features)
        t: (batch, seq_len, time_features)
        coords: (batch, seq_len, 2) — lat, lon in radians or normalized
        """

        # POI & level-based embeddings
        x_embedded = self.poi_embedding(x[..., 0])

        # Optional time embedding
        if self.time_embedding:
            t_embedded = torch.cat(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ],
                dim=-1,
            )
        else:
            t_embedded = None

        # Sinusoidal coordinate embedding
        coord_embedded = self.coord_embedding(ll)

        # Combine
        parts = [x_embedded, coord_embedded]
        if t_embedded is not None:
            parts.append(t_embedded)

        return torch.cat(parts, dim=-1)


class LinearProjectionEmbedding(nn.Module):
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
        self.active_levels = (
            list(range(len(num_embeddings_loc)))
            if loc_level is None
            else list(range(loc_level + 1))
        )

        self.poi_embedding = nn.Embedding(num_embeddings_loc[0], embedding_dim_loc // 2)

        self.time_embedding = (
            nn.ModuleList(
                [nn.Embedding(num, embedding_dim_time) for num in num_embeddings_time]
            )
            if embedding_dim_time
            else None
        )

        self.coord_embedding = nn.Linear(2, embedding_dim_loc // 2)
        self.dim = (
            embedding_dim_loc + embedding_dim_time
            if self.time_embedding
            else embedding_dim_loc
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, loc_features)
        t: (batch, seq_len, time_features)
        coords: (batch, seq_len, 2) — lat, lon in radians or normalized
        """

        # POI & level-based embeddings
        x_embedded = self.poi_embedding(x[..., 0])

        # Optional time embedding
        if self.time_embedding:
            t_embedded = torch.cat(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ],
                dim=-1,
            )
        else:
            t_embedded = None

        # Sinusoidal coordinate embedding
        coord_embedded = self.coord_embedding(ll)

        # Combine
        parts = [x_embedded, coord_embedded]
        if t_embedded is not None:
            parts.append(t_embedded)

        return torch.cat(parts, dim=-1)


class MilesFourierEmbedding(nn.Module):
    """MiLES multi-level discrete lookups concatenated with a Fourier coordinate
    encoding. This replaces the *POI half* of ``FourierFeatureEmbedding`` with the
    full weighted multi-level MiLES stack, keeping the sinusoidal coordinate half.

    ``weight_factor`` drives the MiLES per-level dimension allocation (as in
    ``MilesEmbedding``); the Fourier frequency scale is fixed to ``FOURIER_SCALE``
    (the tuned Space2Vec grid scale used for the standalone Fourier baseline),
    since a single ``weight_factor`` cannot serve both roles.
    """

    FOURIER_SCALE = 4000

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

        # Active location levels (identical logic to MilesEmbedding).
        if loc_level is not None and loc_level >= 0:
            self.active_levels = [loc_level]
        elif loc_level is not None and loc_level < 0:
            self.active_levels = [
                i for i in range(len(num_embeddings_loc)) if i != -loc_level - 1
            ]
        else:
            self.active_levels = list(range(len(num_embeddings_loc)))

        # Split the location budget: MiLES gets half, Fourier coords the other half.
        miles_dim = embedding_dim_loc // 2
        coord_dim = embedding_dim_loc - miles_dim

        # MiLES per-level dimensions within the MiLES half.
        loc_level_weights = np.array(
            [weight_factor**-i for i in self.active_levels], dtype=float
        )
        loc_level_weights /= loc_level_weights.sum()
        loc_level_dims = (loc_level_weights * miles_dim).astype(int)
        loc_level_dims[0] = miles_dim - loc_level_dims[1:].sum()

        self.loc_embedding = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=num_embeddings_loc[level], embedding_dim=l_dim
                )
                for level, l_dim in zip(self.active_levels, loc_level_dims)
            ]
        )
        self.weights = torch.nn.Parameter(
            torch.ones(len(self.active_levels), dtype=torch.float32), requires_grad=True
        )

        # Fourier coordinate half.
        self.coord_embedding = SinusoidalLocationEmbedding(coord_dim, self.FOURIER_SCALE)
        coord_out = 4 * (coord_dim // 4)  # actual sinusoidal output width

        # Time embedding (same scheme as MilesEmbedding/LookupConcatEmbedding).
        if num_embeddings_time and embedding_dim_time:
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
            time_out = embedding_dim_time
        else:
            self.time_embedding = None
            time_out = 0

        self.dim = int(sum(loc_level_dims) + coord_out + time_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor):
        loc_indices = x[..., self.active_levels]
        x_embedded = [
            emb(loc_indices[..., i]) * w
            for i, (emb, w) in enumerate(zip(self.loc_embedding, self.weights))
        ]
        coord_embedded = self.coord_embedding(ll)
        parts = [*x_embedded, coord_embedded]
        if self.time_embedding:
            t_embedded = torch.cat(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ],
                dim=-1,
            )
            parts.append(t_embedded)
        return torch.cat(parts, dim=-1)


class HexagonalLocationEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, scale: float, num_directions: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_directions = num_directions

        # Each (direction, frequency) pair produces sin + cos -> 2 outputs.
        # So half_dim = embedding_dim / (2 * num_directions) frequencies per direction.
        self.num_freqs = embedding_dim // (2 * num_directions)

        # Geometric frequency sweep, same construction as the axis-aligned version.
        inv_freq = (
            2
            * torch.pi
            * scale ** (torch.arange(0.0, self.num_freqs, 1.0) / self.num_freqs)
        )
        self.register_buffer("inv_freq", inv_freq)  # (num_freqs,)

        # Fixed unit vectors evenly spaced over a half-circle (Space2Vec uses 2π/3
        # apart for the hexagonal/grid-cell basis; generalized here to num_directions).
        angles = torch.arange(0.0, num_directions, 1.0) * (torch.pi / num_directions)
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer("directions", directions)  # (num_directions, 2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (batch, seq_len, 2) where the last dim is (lat, lon)
        returns: (batch, seq_len, embedding_dim)
        """
        # Project coords onto each direction: (B, T, num_directions)
        proj = coords @ self.directions.to(coords.device).T

        # Scale each projection by every frequency: (B, T, num_directions, num_freqs)
        scaled = proj.unsqueeze(-1) * self.inv_freq.to(coords.device)

        sin = torch.sin(scaled)
        cos = torch.cos(scaled)

        # Interleave/concatenate -> (B, T, num_directions * num_freqs * 2)
        out = torch.cat([sin, cos], dim=-1)
        return out.flatten(start_dim=-2)  # (B, T, embedding_dim)


class Space2VecEmbedding(nn.Module):
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

        poi_dim = embedding_dim_loc // 2
        coord_dim = embedding_dim_loc - poi_dim
        self.poi_embedding = nn.Embedding(num_embeddings_loc[0], poi_dim)

        self.coord_embedding = HexagonalLocationEmbedding(coord_dim, weight_factor)
        coord_out_dim = self.coord_embedding.num_freqs * self.coord_embedding.num_directions * 2
        self.coord_projection = (
            nn.Linear(coord_out_dim, coord_dim)
            if coord_out_dim != coord_dim
            else nn.Identity()
        )

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
        self.dim = int(embedding_dim_loc + embedding_dim_time)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, ll: torch.Tensor
    ) -> torch.Tensor:
        x_embedded = self.poi_embedding(x[..., 0])
        coord_embedded = self.coord_projection(self.coord_embedding(ll))

        if self.time_embedding:
            t_embedded = torch.cat(
                [
                    embedding(t[..., level])
                    for level, embedding in enumerate(self.time_embedding)
                ],
                dim=-1,
            )
            return torch.cat([x_embedded, coord_embedded, t_embedded], dim=-1)
        return torch.cat([x_embedded, coord_embedded], dim=-1)

EMBEDDING_TYPES = {
    "lookup_sum": LookupSumEmbedding,
    "lookup_concat": LookupConcatEmbedding,
    "lookup_weighted_concat": MilesEmbedding,
    "lookup_weighted_sum": LookupWeightedSumEmbedding,
    "fourier": FourierFeatureEmbedding,
    "miles_fourier": MilesFourierEmbedding,
    "linear": LinearProjectionEmbedding,
    "cosine": CosineEmbedding,
    "baseline": BaselineEmbedding,
    "space2vec": Space2VecEmbedding,
}
