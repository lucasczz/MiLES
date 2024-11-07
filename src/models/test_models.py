import pytest
import torch
from src.models.deeptul import DeepTUL
from src.models.maintul import MainTUL
from src.models.tulvae import TULVAE
from torch.optim import Adam

N_USERS = 10
N_LOCS = 1000
BATCH_SIZE = 4
N_EPOCHS = 10

models_to_test = [MainTUL, TULVAE, DeepTUL]


@pytest.fixture
def synth_data():
    sizes_c = [4, 3, 8, 12]
    xc = [torch.randint(1, N_LOCS, (size,)) for size in sizes_c]
    tc = [torch.randint(0, 24, (size,)).sort()[0] for size in sizes_c]
    tcs = [
        torch.randint(0, 100000, (size,), dtype=torch.float32).sort()[0]
        for size in sizes_c
    ]
    uc = torch.randint(1, N_USERS, (BATCH_SIZE,))

    sizes_h = [12, 25, 17, 33]

    xh = [torch.randint(1, N_LOCS, (size,)) for size in sizes_h]
    th = [torch.randint(0, 24, (size,)).sort()[0] for size in sizes_h]
    ths = [
        torch.randint(0, 100000, (size,), dtype=torch.float32).sort()[0]
        for size in sizes_h
    ]
    uh = [torch.randint(0, N_USERS, (size,)) for size in sizes_h]
    return xc, tc, tcs, xh, th, ths, uc, uh


@pytest.mark.parametrize("model_fn", models_to_test)
def test_train_step(model_fn, synth_data):
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    xc, tc, tcs, xh, th, ths, uc, uh = synth_data
    model = model_fn(n_users=N_USERS, n_locs=N_LOCS, n_times=24, device=device)
    model = model.to(device)
    optim = Adam(model.parameters())
    losses = []
    for epoch in range(N_EPOCHS):
        loss = model.train_step(
            xc=xc, tc=tc, uc=uc, tcs=tcs, xh=xh, th=th, ths=ths, uh=uh
        )
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.detach().item())
    assert losses[0] > losses[-1]
