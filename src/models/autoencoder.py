from torch import nn
import torch
import lightning as L


class GRUAutoencoder(L.LightningModule):
    """GRU autoencoder module as proposed by [Sutskever et al., 2014](https://arxiv.org/pdf/1409.3215.pdf). Each observation within the input window is iteratively fed to an encoder-GRU. Once all observations have been ingested, the hidden state is passed to a decoder-GRU that reconstructs them in reverse order (so the last input is reconstructed first).

    Parameters
    ----------
    n_channels: int, optional
        Number of channels in the input data.
    hidden_size : int, optional
        Number of hidden units used by the GRU layers, by default 32.
    n_layers : int, optional
        Number of GRU layers, by default 1.
    sigmoid_output : bool, optional
        Whether to apply sigmoid to reconstructions, by default True.
    lr : float, optional
        Learning rate to use for training the network, by default 0.001.
    weight_decay : float, optional
        Weight decay of the optimizer used for training, by default 0.0001.
    window_size : int, optional
        Number of previous samples to reconstruct, by default 64.
    loss_fn : str, optional
        The name of the error function to use for training and predicting the anomalousness of samples. Choose from: 'mse_loss', 'l1_loss', 'smooth_l1_loss', 'binary_cross_entropy', 'binary_cross_entropy_with_logits', 'kl_div', 'margin_ranking_loss'.
    optim_fn : str, optional
        The name of the optimizer function used for training. Choose from: 'SGD', 'Adam', 'Adadelta', 'Adagrad', 'RMSprop', 'AdamW'.
    use_embedding: bool, optional
        Whether to embed the input data by adding positional encodings and applying a convolutional layer.
    anomaly_weight: float, optional
        Weight of anomalies relative to total data to be used when computing classification metrics.
    """

    def __init__(
        self,
        n_channels: int = 13,
        hidden_size: int = 32,
        n_layers: int = 1,
        sigmoid_output: bool = True,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        window_size: int = 64,
        loss_fn: str = "l1_loss",
        optim_fn: str = "AdamW",
        anomaly_weight: float = 0.05,
        bidirectional: bool = False,
        dropout: float = 0,
        pred_target: str = "last",
        offset: int = 0,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            window_size=window_size,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            sigmoid_output=sigmoid_output,
            anomaly_weight=anomaly_weight,
            dropout=dropout,
            pred_target=pred_target,
            offset=offset,
        )
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.fc_enc = nn.Linear(n_channels, hidden_size)
        self.fc_dec = nn.Linear(hidden_size, n_channels)
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.decoder = nn.ModuleList(
            [nn.GRUCell(hidden_size, hidden_size) for _ in range(n_layers)]
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor):
        # x.shape = (batch_size, n_timesteps, n_channels)
        x = x.flip(1)
        if self.embedding:
            x = self.embedding(x)
        x = self.fc_enc(x)
        _, h_enc = self.encoder(x)
        o = h_enc[-1]
        h_dec = [None] * self.n_layers
        x_pred = []
        for _ in range(x.shape[1]):
            for idx, layer in enumerate(self.decoder):
                o = layer(o, h_dec[idx])
                h_dec[idx] = o
            x_pred.append(o)
        x_pred = torch.stack(x_pred, dim=1)
        x_pred = self.fc_dec(x_pred)
        return x_pred
