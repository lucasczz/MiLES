import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from typing import List, Tuple

from einops import rearrange

from src.embedding import EMBEDDING_TYPES


class TULVAE(nn.Module):
    """
    A model that combines trajectory classification with VAE reconstruction.
    For each trajectory, it:
    1. Predicts user probabilities using a bidirectional LSTM classifier
    2. Reconstructs trajectories for all possible users using TULVAE
    3. Computes loss by weighting reconstruction losses with classification probabilities

    Args:
        n_locs (int): Number of possible locations
        n_times (int): Number of time steps
        n_users (int): Number of users
        n_hidden (int): Number of hidden units in the LSTM layers
        embedding_dim (int): Dimension of the embedding layer
        latent_dim (int): Dimension of the VAE latent space
        dropout (float): Dropout rate
        n_layers (int): Number of LSTM layers
        subseq_steps (int): Steps for trajectory subsequences
        device (torch.device): Device to run the model on
    """

    def __init__(
        self,
        n_locs: int,
        n_times: int,
        n_users: int,
        n_hidden: int = 512,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 300,
        time_embedding_dim: int = 50,
        latent_dim: int = 50,
        bidirectional: bool = True,
        dropout: float = 0.5,
        n_layers: int = 1,
        alpha: float = 0.5,
        beta: float = 0.5,
        subseq_steps: int = 6,
        monte_carlo_samples: int = 3,
        loc_level: int = None,
        device: torch.device = "cuda:0",
    ):
        super().__init__()
        self.n_users = n_users
        self.n_times = n_times
        self.n_locs = n_locs
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.k = monte_carlo_samples

        # VAE component
        self.vae = HierarchicalVAE(
            n_locs=n_locs,
            n_times=n_times,
            n_users=n_users,
            n_hidden=n_hidden,
            embedding_type=embedding_type,
            embedding_weight_factor=embedding_weight_factor,
            loc_embedding_dim=loc_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            latent_dim=latent_dim,
            bidirectional=bidirectional,
            dropout=dropout,
            n_layers=n_layers,
            subseq_steps=subseq_steps,
            loc_level=loc_level,
            device=device,
        )
        # Shared embedding layer for locations
        self.embedding = self.vae.embedding

        self.n_dirs = 1 + bidirectional
        # Classifier components
        self.clf_lstm = nn.LSTM(
            self.embedding.dim,
            n_hidden,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.clf_out = nn.Linear(self.n_dirs * n_hidden, n_users)

    def forward(
        self,
        loc_embedded: torch.Tensor,
        time_padded: torch.Tensor,
        seq_lengths: torch.Tensor,
        u: torch.Tensor,
    ):
        """
        Forward pass combining classification and reconstruction.

        Args:
            loc_embedded: Padded tensors of location embeddings with shape (batch_size, max_seq_length, embedding_dim).
            time_padded: Padded tensors of timestamps for each visit.
            seq_lengths: Lengths of sequences in the batch.
            u: User labels for each trajectory.

        Returns:
            Classification logits, reconstruction logits, and latent variables mu + logvar for sampled user IDs.
        """
        # Get classification logits
        loc_seq_packed = pack_padded_sequence(
            loc_embedded, seq_lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.clf_lstm(loc_seq_packed)
        h = self.dropout(h)
        h = rearrange(h[-self.n_dirs :], "dir batch hidden -> batch (dir hidden)")
        clf_logits = self.clf_out(h)

        # Get reconstruction logits and latent variables
        batch_size, max_seq_len, _ = loc_embedded.shape

        # Sample self.k user IDs randomly for reconstruction
        u_sample = torch.randint(
            low=0,
            high=self.n_users,
            size=(
                batch_size,
                self.k - 1,
            ),
            device=u.device,
        )
        u_sample = torch.cat([u_sample, u[:, None]], -1).flatten()
        u_sample = u_sample.repeat(batch_size)  # Expand for batch

        # Expand sequences for sampled users
        expanded_loc_padded = loc_embedded.repeat_interleave(self.k, dim=0)
        expanded_time_padded = time_padded.repeat_interleave(self.k, dim=0)
        expanded_lengths = seq_lengths.repeat_interleave(self.k)

        # Get reconstruction logits from VAE for sampled users
        rec_logits, mu, logvar = self.vae(
            expanded_loc_padded,
            expanded_time_padded,
            expanded_lengths,
            u_sample,
        )
        mu = mu.reshape(batch_size, self.k, -1)
        logvar = logvar.reshape(batch_size, self.k, -1)
        rec_logits = rec_logits.reshape(batch_size, -1, self.k, max_seq_len)
        # rec_logits.shape = (batch_size, n_locs, self.k, max_seq_len)

        clf_probas = torch.softmax(clf_logits, dim=-1)
        k_probas = clf_probas[
            torch.arange(batch_size).repeat_interleave(self.k), u_sample
        ]
        k_probas = k_probas.reshape(batch_size, self.k)

        return clf_logits, rec_logits, mu, logvar, k_probas

    def train_step(
        self, xc: List[torch.Tensor], tc: List[torch.Tensor], uc: torch.Tensor, **kwargs
    ):
        # Get sequence lengths and pad the sequences
        seq_lengths = torch.tensor([len(seq) for seq in xc])
        x_padded = pad_sequence(xc, batch_first=True).to(self.device)
        time_padded = pad_sequence(
            tc, batch_first=True, padding_value=2 * self.n_times[0]
        ).to(self.device)
        uc = uc.to(self.device)

        # Embed locations
        xt_embedded = self.embedding(
            x_padded, pad_sequence(tc, batch_first=True).to(self.device)
        )

        clf_logits, rec_logits, mu, logvar, k_probas = self(
            xt_embedded, time_padded, seq_lengths, uc
        )
        # clf_logits.shape = (batch_size, n_users)
        # rec_logits.shape = (batch_size, n_locs, n_users, max_seq_len)
        # loc_padded.shape = (batch_size, max_seq_len)
        # loc.shape = (batch_size, max_seq_len)

        rec_targets = x_padded[..., 0].unsqueeze(1).expand(-1, self.k, -1).long()
        rec_losses = F.cross_entropy(
            rec_logits, rec_targets, ignore_index=0, reduction="none"
        )
        # rec_losses.shape = (batch_size, n_users, max_seq_len)
        rec_losses = rec_losses.mean(dim=-1)
        latent_losses = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        vae_losses = rec_losses + self.beta * latent_losses
        vae_loss_diffs = vae_losses - vae_losses.mean(1)

        unlab_loss = torch.einsum("ik,ik->i", k_probas, vae_loss_diffs).mean()
        lab_rec_loss = vae_losses[torch.arange(len(vae_losses)), -1].mean()
        lab_clf_loss = F.cross_entropy(clf_logits, uc)
        loss = lab_rec_loss + lab_clf_loss + self.beta * unlab_loss
        return loss

    def pred_step(self, xc: List[torch.Tensor], tc: List[torch.Tensor], **kwargs):
        # Get sequence lengths and pad the sequences
        seq_lengths = torch.tensor([len(seq) for seq in xc])
        x_padded = pad_sequence(xc, batch_first=True).to(self.device)
        t_padded = pad_sequence(tc, batch_first=True).to(self.device)
        # Embed locations
        loc_embedded = self.embedding(x_padded, t_padded)
        loc_seq_packed = pack_padded_sequence(
            loc_embedded, seq_lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.clf_lstm(loc_seq_packed)
        h = self.dropout(h)
        h = rearrange(h[-self.n_dirs :], "dir batch hidden -> batch (dir hidden)")
        clf_logits = self.clf_out(h)
        return clf_logits


class HierarchicalVAE(nn.Module):
    def __init__(
        self,
        n_locs: int,
        n_times: int,
        n_users: int,
        n_hidden: int = 300,
        embedding_type: str = "lookup_sum",
        embedding_weight_factor: float = 2,
        loc_embedding_dim: int = 250,
        time_embedding_dim: int = 50,
        latent_dim: int = 50,
        bidirectional: bool = True,
        dropout: float = 0.5,
        n_layers: int = 1,
        subseq_steps: int = 6,
        loc_level: int = None,
        device: torch.device = "cuda:0",
    ):
        super().__init__()
        self.device = device
        self.n_hidden = n_hidden
        self.latent_dim = latent_dim
        self.n_dirs = 1 + bidirectional
        self.start_loc = torch.tensor(n_locs)[None, :]
        self.start_time = torch.tensor(n_times)[None, :]
        self.n_locs = [n_loc + 1 for n_loc in n_locs]
        self.n_times = [n_time + 1 for n_time in n_times]
        self.n_layers = n_layers
        self.n_users = n_users
        self.subseq_steps = subseq_steps
        self.dropout = nn.Dropout(dropout)

        # Embedding
        self.embedding = EMBEDDING_TYPES[embedding_type](
            num_embeddings_loc=self.n_locs,
            embedding_dim_loc=loc_embedding_dim,
            num_embeddings_time=self.n_times,
            embedding_dim_time=time_embedding_dim,
            weight_factor=embedding_weight_factor,
            loc_level=loc_level,
        )

        # LSTM Encoders and Decoder without dropout in constructor
        self.poi_encoder_lstm = nn.LSTM(
            self.embedding.dim, n_hidden, bidirectional=bidirectional, batch_first=True
        )
        self.subseq_encoder_lstm = nn.LSTM(
            self.n_dirs * n_hidden,
            n_hidden,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            self.embedding.dim + n_users,
            n_hidden,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Layers for latent variable distribution
        self.fc_mu_poi = nn.Linear(self.n_dirs * n_hidden, latent_dim)
        self.fc_logvar_poi = nn.Linear(self.n_dirs * n_hidden, latent_dim)
        self.fc_mu_subseq = nn.Linear(self.n_dirs * n_hidden, latent_dim)
        self.fc_logvar_subseq = nn.Linear(self.n_dirs * n_hidden, latent_dim)

        # Decoder input projection
        self.fc_decoder_input = nn.Linear(2 * latent_dim, self.n_dirs * n_hidden)
        self.fc_output = nn.Linear(self.n_dirs * n_hidden, self.n_locs[0])

    def forward(
        self,
        loc_embedded: torch.Tensor,
        time_padded: torch.Tensor,
        seq_lengths: torch.Tensor,
        user_id: torch.Tensor,
    ) -> torch.Tensor:
        subseq_input, poi_hidden, poi_cell = self.encode_poi(
            loc_embedded, time_padded, seq_lengths
        )
        if subseq_input is not None:
            subseq_hidden = self.encode_subseq(subseq_input)
        else:
            subseq_hidden = torch.zeros_like(poi_hidden)

        # Get latent variable distribution
        poi_mu, poi_logvar = self.fc_mu_poi(poi_hidden), self.fc_logvar_poi(poi_hidden)
        subseq_mu, subseq_logvar = (
            self.fc_mu_subseq(subseq_hidden),
            self.fc_logvar_subseq(subseq_hidden),
        )
        mu = torch.cat([poi_mu, subseq_mu], dim=-1)
        logvar = torch.cat([poi_logvar, subseq_logvar], dim=-1)

        # Reparametrize latent variable
        z = self.reparameterize(mu, logvar)

        # Decode the latent variable to reconstruct the trajectory
        decoded_logits = self.decode(z, poi_cell, loc_embedded, user_id)
        return decoded_logits, mu, logvar

    def decode(
        self,
        z: torch.Tensor,
        poi_cell: torch.Tensor,
        loc_embedded: torch.Tensor,
        user_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_seq_len, _ = loc_embedded.shape
        user_one_hot = (
            F.one_hot(user_id, self.n_users).unsqueeze(1).expand(-1, max_seq_len, -1)
        ).to(self.device)

        loc_embed_shifted = torch.roll(loc_embedded, shifts=1, dims=1)
        loc_embed_shifted[:, 0] = self.embedding(
            self.start_loc.to(self.device), self.start_time.to(self.device)
        )

        loc_user_embed = torch.cat([loc_embed_shifted, user_one_hot], dim=-1)

        z_projected = F.softplus(self.fc_decoder_input(z)).view(
            self.n_dirs, batch_size, self.n_hidden
        )

        # Decode the sequence with dropout applied to decoder outputs
        decoder_output, _ = self.decoder_lstm(loc_user_embed, (z_projected, poi_cell))
        decoder_output = self.dropout(decoder_output)
        return self.fc_output(decoder_output)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_subseq(self, subseq_input: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.subseq_encoder_lstm(subseq_input)
        h = rearrange(h[-self.n_dirs :], "dir batch hidden -> batch (dir hidden)")
        return self.dropout(h)

    def encode_poi(
        self,
        loc_embedded: torch.Tensor,
        time_padded: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loc_packed = pack_padded_sequence(
            loc_embedded, seq_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (h, c) = self.poi_encoder_lstm(loc_packed)
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        batch_size, max_subseq_len, n_hidden = unpacked_output.shape
        subseq_idcs, subseq_lengths = self.get_subseq_indices(time_padded[..., 0])
        subseq_input = unpacked_output.gather(
            1, subseq_idcs.unsqueeze(-1).expand(-1, -1, n_hidden)
        )
        if subseq_input.nelement() > 0:
            subseq_input_packed = pack_padded_sequence(
                subseq_input, subseq_lengths, batch_first=True, enforce_sorted=False
            )
        else:
            subseq_input_packed = None
        h = rearrange(h[-self.n_dirs :], "dir batch hidden -> batch (dir hidden)")
        return subseq_input_packed, self.dropout(h), c

    def get_subseq_indices(
        self, time_padded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_deltas = time_padded - time_padded[:, 0, None]
        time_intervals = (
            torch.arange(0, self.n_times[0], self.subseq_steps, device=self.device)
            .unsqueeze(0)
            .expand(time_deltas.shape[0], -1)
            .contiguous()
        )
        subseq_idcs = torch.searchsorted(time_deltas, time_intervals, side="right") - 1

        mask = subseq_idcs[:, 1:] == subseq_idcs[:, :-1]
        subseq_idcs = subseq_idcs[:, 1:]
        batch_size, max_idx = time_padded.shape
        subseq_idcs[mask] = max_idx
        subseq_idcs, _ = torch.sort(subseq_idcs)
        subseq_lengths = (subseq_idcs < max_idx).sum(-1)
        subseq_idcs = [
            subseq_idx[:subseq_len]
            for subseq_idx, subseq_len in zip(subseq_idcs, subseq_lengths)
        ]
        return pad_sequence(subseq_idcs, batch_first=True), subseq_lengths.to("cpu")
