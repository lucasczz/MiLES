import json
from pathlib import Path
import time
import torch
import torch.nn as nn
from typing import Dict, List
from collections import defaultdict
import numpy as np

BASEPATH = Path(__file__).parent.parent.parent.joinpath("reports")


def top_n_tp(logits, label, n=5):
    n_largest_idcs = np.argpartition(logits[0], -n)[-n:]
    return (n_largest_idcs == label).sum().item()


def handle_queue(q, file_path):
    """listens for messages on the q, writes to file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        while True:
            m = q.get()
            if m == "kill":
                break
            json.dump(m, f)
            f.write("\n")
            f.flush()


class EmbeddingWeightTracker:
    def __init__(self):
        self.embedding_weights = []
        self.embedding_norms = [[] for _ in range(4)]  # Fixed initialization

    def weight_hook_fn(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Hook function to store embedding weights during forward pass"""
        # Store a copy of the current embedding weights
        self.embedding_weights.append(module.weights.numpy(force=True))

    def get_norm_hook_fn(self, level):
        def norm_hook_fn(
            module: nn.Module, input: torch.Tensor, output: torch.Tensor
        ) -> None:
            """Hook function to store embedding weights during forward pass"""
            # Store a copy of the current embedding weights
            self.embedding_norms[level].append(torch.norm(output).item())

        return norm_hook_fn

    def register_hooks(
        self, model: nn.Module
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """Register hooks on all embedding layers in the model"""
        weight_handle = None
        if hasattr(model.embedding, "weights"):
            weight_handle = model.embedding.register_forward_hook(self.weight_hook_fn)
        norm_handles = [
            level.register_forward_hook(self.get_norm_hook_fn(idx))
            for idx, level in enumerate(model.embedding.loc_embedding)
        ]
        return weight_handle, norm_handles


class JSONTracker:
    def __init__(
        self,
        save_path: str,
        parameters: Dict = {},
        write_queue=None,
        module=None,
        emb_log_interval: int = 1,
        log_emb_norms: bool = False,
    ):
        self.labels, self.preds, self.top_5_tps = [], [], []
        self.embedding_weights = []
        self.embedding_norms = []
        self.start = time.time()
        self.parameters = parameters
        self.save_path = BASEPATH.joinpath(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_queue = write_queue
        self.module = module
        self.mod_is_weighted = hasattr(module.embedding, "weights")
        self.counter = 0
        self.emb_log_interval = emb_log_interval
        self.log_emb_norms = log_emb_norms

    def update(self, logits, label):
        _logits = logits.numpy(force=True)
        _label = label.to(torch.int16).item()
        self.labels.append(_label)
        self.preds.append(_logits.argmax(-1).item())
        self.top_5_tps.append(top_n_tp(_logits, _label))
        if self.module is not None and self.counter % self.emb_log_interval == 0:
            if self.mod_is_weighted:
                self.embedding_weights.append(
                    self.module.embedding.weights.numpy(force=True)
                )
            if self.log_emb_norms:
                self.embedding_norms.append(
                    [
                        torch.norm(level.weight, dim=1).mean().item()
                        for level in self.module.embedding.loc_embedding
                    ]
                )
        self.counter += 1

    def save(self):
        entry = self.parameters | {
            "labels": self.labels,
            "preds": self.preds,
            "top_5_tps": self.top_5_tps,
            "runtime": time.time() - self.start,
        }
        if self.module is not None:
            if self.mod_is_weighted:
                emb_weights_t = np.stack(self.embedding_weights).T
                for i, weights in enumerate(emb_weights_t):
                    entry[f"embedding_weight_{i}"] = weights.tolist()
            if self.log_emb_norms:
                norms_t = np.stack(self.embedding_norms).T
                for i, norms in enumerate(norms_t):
                    entry[f"embedding_norm_{i}"] = norms.tolist()

        if self.write_queue is not None:
            self.write_queue.put(entry)
        else:
            with open(self.save_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")
