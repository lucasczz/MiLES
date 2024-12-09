import json
from pathlib import Path
import time
from typing import Dict

import numpy as np
import torch

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


class JSONTracker:
    def __init__(self, save_path: str, parameters: Dict = {}, write_queue=None):
        self.labels, self.preds, self.top_5_tps = [], [], []
        self.start = time.time()
        self.parameters = parameters
        self.save_path = BASEPATH.joinpath(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_queue = write_queue

    def update(self, logits, label):
        _logits = logits.numpy(force=True)
        _label = label.to(torch.int16).item()
        self.labels.append(_label)
        self.preds.append(_logits.argmax(-1).item())
        self.top_5_tps.append(top_n_tp(_logits, _label))

    def save(self):
        entry = self.parameters | {
            "labels": self.labels,
            "preds": self.preds,
            "top_5_tps": self.top_5_tps,
            "runtime": time.time() - self.start,
        }
        if self.write_queue is not None:
            self.write_queue.put(entry)
        else:
            with open(self.save_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")
