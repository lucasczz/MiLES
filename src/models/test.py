import pytest
import torch
from src.models.tulvae import TULVAE
from torch.optim import Adam


@pytest.mark.parametrize(
    "batch_size, n_locs, n_times, n_users, n_steps",
    [(4, 10, 5, 3, 10)],  # Small batch and short sequence for faster testing
)
def test_loss_decreases(batch_size, n_locs, n_times, max_seq_len, n_users, n_steps):
    # Model parameters
    n_hidden_ae = 300
    n_hidden_clf = 512
    embedding_dim = 300
    latent_dim = 50
    dropout = 0.5
    n_layers = 1
    device = torch.device("cpu")

    # Initialize model
    model = TULVAE(
        n_locs=n_locs,
        n_times=n_times,
        n_users=n_users,
        n_hidden_ae=n_hidden_ae,
        n_hidden_clf=n_hidden_clf,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        n_layers=n_layers,
        device=device,
    ).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Generate random data for a batch
    seq_lengths = torch.randint(3, max_seq_len, (batch_size,))
    loc = [torch.randint(0, n_locs, (seq_length,)) for seq_length in seq_lengths]
    time = [
        torch.randint(0, n_times, (seq_length,)).sort().values
        for seq_length in seq_lengths
    ]
    user_ids = torch.randint(0, n_users, (batch_size,))

    # Track loss over training steps
    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = model.train_step(loc, time, user_ids)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Check if loss is decreasing on average
    assert losses[-1] < losses[0], "Loss did not decrease over time"


test_loss_decreases(8, 1000, 24, 20, 100, 10)
