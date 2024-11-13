import numpy as np
import pytest
import torch
from src.models.deeptul import DeepTUL
from src.models.maintul import MainTUL
from src.models.t3s import T3S
from src.models.tuler import TULERG, TULERL, BiTULER
from src.models.tulvae import TULVAE
from torch.optim import Adam

from src.models.word2vec import Word2Vec

N_USERS = 10
N_LOCS = 1000
BATCH_SIZE = 4
LOC_LEVELS = 4
TIME_LEVELS = 4
N_EPOCHS = 10

models_to_test = [MainTUL, TULVAE, DeepTUL, BiTULER, TULERL, TULERG, Word2Vec, T3S]


def generate_time_features(size):
    hours = torch.randint(1, 25, (size,)).sort()[0]
    sixhours = hours // 6
    day = torch.randint(1, 8, (1,)).expand_as(hours)
    weekend = (day > 5).to(int)
    timestamps = torch.randint(1, 100000, (size,)).sort()[0]
    return torch.stack([hours, sixhours, day, weekend, timestamps], dim=-1)


@pytest.fixture
def synth_data():
    # tc [batch_size][seq_len, (hour, 6h, day, weekend, timestamp)]
    # hc [batch_size][seq_len, (POI, cell0, cell1, ...)]
    sizes_c = np.random.randint(1, 30, (BATCH_SIZE,))
    xc = [torch.randint(1, N_LOCS, (size, 5)) for size in sizes_c]
    tc = [generate_time_features(size) for size in sizes_c]
    llc = [torch.randn((size, 2)) for size in sizes_c]

    uc = torch.randint(1, N_USERS, (BATCH_SIZE,))

    sizes_h = [
        np.random.randint(1, 30, (n_trajectories,))
        for n_trajectories in np.random.randint(3, 12, (BATCH_SIZE,))
    ]

    xh = [
        [torch.randint(1, N_LOCS, (size, 5)) for size in sizes_t] for sizes_t in sizes_h
    ]
    th = [[generate_time_features(size) for size in sizes_t] for sizes_t in sizes_h]
    uh = [torch.randint(0, N_USERS, (len(sizes_t),)) for sizes_t in sizes_h]
    for uhi, uci in zip(uh, uc):
        idx = np.random.randint(0, len(uhi))
        uhi[idx] = uci
    return xc, tc, uc, llc, xh, th, uh


@pytest.mark.parametrize("model_fn", models_to_test)
def test_train_step(model_fn, synth_data):
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    xc, tc, uc, llc, xh, th, uh = synth_data
    model = model_fn(n_users=N_USERS, n_locs=[N_LOCS], n_times=[24], device=device, embedding_type='cosine')
    model = model.to(device)
    optim = Adam(model.parameters())
    losses = []
    for epoch in range(N_EPOCHS):
        loss = model.train_step(xc=xc, tc=tc, uc=uc, xh=xh, th=th, uh=uh, llc=llc)
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.detach().item())
    assert losses[0] > losses[-1]
