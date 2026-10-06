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
        log_input_weight_norms: bool = False,
        ll_scale: np.ndarray = None,
    ):
        self.labels, self.preds, self.top_5_tps = [], [], []
        self.embedding_weights = []
        self.embedding_norms = []
        self.input_weight_norms = []
        self.start = time.time()
        self.parameters = parameters
        self.save_path = BASEPATH.joinpath(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_queue = write_queue
        self.module = module
        self.mod_is_weighted = module and hasattr(module.embedding, "weights")
        self.counter = 0
        self.emb_log_interval = emb_log_interval
        self.log_emb_norms = log_emb_norms
        self.log_input_weight_norms = log_input_weight_norms
        # Per-level column slices into the RNN input-weight matrix, in the order the
        # location embeddings are concatenated. Used to read off, per level, how
        # strongly the first RNN layer attends to that level's embedding block.
        self._loc_blocks = None
        if module is not None and log_input_weight_norms:
            dims = [e.embedding_dim for e in module.embedding.loc_embedding]
            slices, start = [], 0
            for d in dims:
                slices.append((start, start + d))
                start += d
            self._loc_blocks = slices
        self.pred_dists = []
        self.ll_scale = ll_scale
        self.custom_metrics = {}

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
            if self.log_input_weight_norms:
                self.input_weight_norms.append(self._input_block_col_norms())
        self.counter += 1

    def _input_block_col_norms(self):
        """Mean L2 column-norm of the first RNN layer's input weights, per location
        level. For a bidirectional RNN the forward and reverse blocks are averaged.
        This is the (identifiable) read-out scale W_l: combined with w_l and the
        embedding row norm it gives each level's forward contribution magnitude."""
        enc = self.module.encoder
        mats = [enc.weight_ih_l0]
        rev = getattr(enc, "weight_ih_l0_reverse", None)
        if rev is not None:
            mats.append(rev)
        col_norms = sum(m.detach().norm(dim=0) for m in mats) / len(mats)
        return [col_norms[s:e].mean().item() for s, e in self._loc_blocks]

    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.custom_metrics:
                self.custom_metrics[key].append(value)
            else:
                self.custom_metrics["key"] = [value]

    def update_next_loc(self, preds, y):
        _dist = np.sqrt(
            (
                ((preds.numpy(force=True) - y.numpy(force=True)) * self.ll_scale) ** 2
            ).sum()
        )
        self.pred_dists.append(_dist)
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
            if self.log_input_weight_norms:
                wnorms_t = np.stack(self.input_weight_norms).T
                for i, norms in enumerate(wnorms_t):
                    entry[f"input_weight_norm_{i}"] = norms.tolist()
        if len(self.pred_dists) > 0:
            entry["pred_dists"] = self.pred_dists
        for key, values in self.custom_metrics:
            entry[key] = values

        if self.write_queue is not None:
            self.write_queue.put(entry)
        else:
            with open(self.save_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")
