import csv
from pathlib import Path
import time
from typing import Callable, Dict, List

from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    top_k_accuracy_score,
)
import numpy as np


def top_1_accuracy(label, logits, classes):
    return top_k_accuracy_score(label, logits, k=1, labels=classes)


def top_5_accuracy(label, logits, classes):
    return top_k_accuracy_score(label, logits, k=5, labels=classes)


def pred(labels, logits, classes):
    return 


METRICS = [macro_f1, macro_precision, macro_recall, top_1_accuracy, top_5_accuracy]


class ExperimentTracker:
    def __init__(
        self,
        save_path: str,
        metric_fns: List[Callable] = METRICS,
        parameters: Dict = {},
        logging_interval: int = 128,
        n_classes: int = 400,
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
        self.classes = np.arange(n_classes)

    def update(self, y, y_pred, loss):
        # Log loss
        self.tmp_variables["loss"].append(loss.item())

        # Log predictions and labels
        self.labels.append(y.numpy(force=True))
        self.preds.append(y_pred.numpy(force=True))

        self.step += len(y)
        # Calculate metrics and save average of values tracked since last logging step
        if self.step % self.logging_interval == 0:
            self.log_step()

    def log_step(self):
        if len(self.preds) > 0:
            results = {k: v for k, v in self.parameters.items()}
            results["step"] = self.step
            results["runtime"] = time.time() - self.last_time
            self.preds = np.concatenate(self.preds)
            self.labels = np.concatenate(self.labels)

            # Calculate metrics based on predictions and labels captured since last logging step
            for metric in self.metric_fns:
                results[metric.__name__] = metric(self.labels, self.preds, self.classes)
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
