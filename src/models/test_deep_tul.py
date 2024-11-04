import pytest
import torch

from src.models.deeptul import DeepTUL, HistoryEncoder


@pytest.fixture
def history_model():
    # Set up the parameters for the HistoryEncoder
    n_hidden = 4
    embedding_dim_loc = 2
    embedding_dim_time = 2
    embedding_dim_user = 2
    n_locs = 4
    n_times = 4
    n_users = 6
    dropout = 0.0
    n_layers = 2
    device = torch.device("cpu")

    # Create the model
    return HistoryEncoder(
        n_hidden=n_hidden,
        loc_embedding_dim=embedding_dim_loc,
        time_embedding_dim=embedding_dim_time,
        user_embedding_dim=embedding_dim_user,
        n_locs=n_locs,
        n_times=n_times,
        n_users=n_users,
        dropout=dropout,
        n_layers=n_layers,
        device=device,
    )


@pytest.fixture
def full_model():
    # Set up the parameters for the HistoryEncoder
    n_hidden = 4
    embedding_dim_loc = 2
    embedding_dim_time = 2
    embedding_dim_user = 2
    n_locs = 4
    n_times = 4
    n_users = 6
    dropout = 0.0
    n_layers = 2
    device = torch.device("cpu")

    # Create the model
    return DeepTUL(
        n_hidden=n_hidden,
        loc_embedding_dim=embedding_dim_loc,
        time_embedding_dim=embedding_dim_time,
        user_embedding_dim=embedding_dim_user,
        n_locs=n_locs,
        n_times=n_times,
        n_users=n_users,
        dropout=dropout,
        n_layers=n_layers,
        device=device,
    )


def random_data(n_users=3, n_locs=2, n_times=2, lens=[[5, 3], [4, 6]], seed=42):
    torch.manual_seed(seed)
    n_users += 1
    n_locs += 1
    n_times += 1
    # Create random input data (batch size = 2, variable sequence lengths)
    x = [
        [torch.randint(1, n_locs, (seq_len,)) for seq_len in seq_lens]
        for seq_lens in lens
    ]
    t = [
        [torch.randint(1, n_times, (seq_len,)) for seq_len in seq_lens]
        for seq_lens in lens
    ]
    u = [
        [torch.randint(1, n_users, (seq_len,)) for seq_len in seq_lens]
        for seq_lens in lens
    ]
    return x, t, u


@pytest.fixture
def default_data():
    return random_data()


def test_history_forward_shape(history_model, default_data):
    x, t, u = default_data
    output, n_unique = history_model(x, t, u)

    # Test the output dimensions
    assert output.shape[0] == len(x)  # Batch size
    assert n_unique.shape[0] == len(x)  # Same batch size


def test_full_forward_shape(default_data):
    n_users = 6
    n_locs = 2
    n_times = 2
    xh, th, uh = random_data(
        n_locs=n_locs, n_times=n_times, n_users=n_users, lens=[[3, 5], [6, 4]]
    )
    xc, tc, _ = random_data(n_users=n_users, lens=[[5], [7]])
    xc = [xc[0][0], xc[1][0]]
    tc = [tc[0][0], tc[1][0]]

    model = DeepTUL(2, 2, 2, 2, n_locs, n_times, n_users, 0, 2, "cpu")
    output = model(xc, tc, xh, th, uh)
    assert output.shape == (2, n_users)


def test_history_padding_effect(history_model):
    batch1 = random_data(lens=[[5], [5]])
    batch2 = random_data(lens=[[5], [8]])
    for data1, data2 in zip(batch1, batch2):
        for i in range(len(data1)):
            data2[i][: len(data1[i])] = data1[i]

    # Forward pass for both batches
    output1, _ = history_model(*batch1)
    output2, _ = history_model(*batch2)

    # Check that the results for the first sequence in both batches are the same
    assert torch.allclose(
        output1[0], output2[0]
    ), "Padding affects the result of the first sequence"
