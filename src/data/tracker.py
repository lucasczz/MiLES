import csv
from pathlib import Path
import time
from typing import Callable, Dict, List

import numpy as np
import torch


class ExperimentTracker:
    def __init__(
        self,
        metric_fns: List[Callable],
        parameters: Dict,
        save_path: str,
        logging_interval: int = 100,
    ) -> None:
        self.metric_fns = metric_fns
        self.parameters = parameters
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.logging_interval = logging_interval
        self.tmp_variables = {"loss": []}
        self.history = []

        self.step = 0
        self.labels = []
        self.preds = []
        self.last_time = time.time()

    def update(self, y, y_pred, loss):
        # Log loss
        self.tmp_variables["loss"].append(loss.item())

        # Log predictions and labels
        self.labels.append(y.detach().numpy())
        self.preds.append(y_pred.detach().numpy())

        self.step += 1
        # Calculate metrics and save average of values tracked since last logging step
        if self.step % self.logging_interval == 0:
            self.log_step()

    def log_step(self):
        results = {k: v for k, v in self.parameters.items()}
        results["step"] = self.step
        results["runtime"] = time.time() - self.last_time
        self.preds = np.concatenate(self.preds)
        self.labels = np.concatenate(self.labels)

        # Calculate metrics based on predictions and labels captured since last loggin step
        for metric in self.metric_fns:
            results[metric.__name__] = metric(self.labels, self.preds)
        self.labels = []
        self.preds = []

        # Calculate mean of variables
        for var_name, values in self.tmp_variables.items():
            results[var_name] = np.mean(values)
            self.tmp_variables[var_name] = []

        self._write_step(results)
        self.last_time = time.time()

    def _write_step(self, results):
        # Check if file to write to has a .csv header already
        has_header = False
        if self.save_path.exists():
            with open(self.save_path, "r") as f:
                has_header = f.read(1024) != ""

        # Write results captured since last logging step
        with open(self.save_path, "a") as f:
            writer = csv.DictWriter(f, results.keys())
            if not has_header:
                writer.writeheader()
            writer.writerow(results)
