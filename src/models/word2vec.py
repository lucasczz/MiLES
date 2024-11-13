from typing import List
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence


class Word2Vec(nn.Module):
    def __init__(
        self,
        n_locs: int,
        embedding_dim: int = 200,
        context_length=3,
        finetune: bool = False,
        max_norm: float = 1.0,
        device: torch.device = "cuda:0",
        **kwargs
    ):
        super(Word2Vec, self).__init__()
        self.finetune = finetune
        self.context_length = context_length
        self.window_size = 2 * context_length + 1
        self.embeddings = nn.Embedding(
            num_embeddings=n_locs + 1,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=n_locs,
        )
        self.device = device

    def forward(self, xc_padded: torch.Tensor):
        if self.finetune:
            return self.embeddings(xc_padded)
        else:
            with torch.no_grad():
                return self.embeddings(xc_padded)

    def train_step(self, xc: List[torch.Tensor], **kwargs):
        words, targets = self.get_training_pairs(xc)
        words_embedded = self.embeddings(words)
        preds = self.linear(words_embedded)
        loss = F.cross_entropy(preds, targets, reduction="sum")
        return loss

    def get_training_pairs(self, xc):
        xc = pad_sequence(xc, batch_first=True, padding_value=-1).to(self.device)
        xc_padded = F.pad(xc, (self.context_length, self.context_length), value=-1)
        xc_windows = xc_padded.unfold(1, self.window_size, 1)
        idcs = list(range(self.window_size))
        idcs.pop(self.context_length)
        xc_windows = xc_windows[..., idcs]

        # Expand target word dimension and stack with context
        xc_windows = torch.stack(
            [xc_windows, xc.unsqueeze(-1).expand_as(xc_windows)], -1
        )

        # Filter out rows that contain padding (-1) in the context window
        xc_pairs = xc_windows.reshape(-1, 2)

        mask = (xc_pairs != -1).all(-1)
        xc_pairs = xc_pairs[mask]
        words, targets = xc_pairs[..., 0], xc_pairs[..., 1]
        return words, targets
