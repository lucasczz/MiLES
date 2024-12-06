import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List
import re
from functools import partial


def get_metric_fn(metric: str):
    # Pattern for top_k_accuracy_n%_low
    match = re.match(r"top_(\d+)_accuracy_(\d+)%_low", metric)
    if match:
        k = int(match.group(1))  # Extract the value of k
        percent = int(match.group(2))  # Extract the percentage
        # Return a partial function of get_top_k_accuracy_percent_low
        return partial(get_top_k_accuracy_percent_low, k=k, percent=percent)

    # Map metric names to their corresponding functions
    metric_map = {
        "macro_precision": get_macro_precision,
        "macro_recall": get_macro_recall,
        "top_1_accuracy": get_top_1_accuracy,
        "top_5_accuracy": get_top_5_accuracy,
        "macro_f1": get_macro_f1,
        "tumbling_top_5_accuracy": get_tumbling_top_5_accuracy,
        "tumbling_top_1_accuracy": get_tumbling_top_1_accuracy,
    }

    # Return the appropriate function or raise an error for invalid metric
    if metric in metric_map:
        return metric_map[metric]
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def get_macro_precision(row):
    return precision_score(
        row["labels"], row["preds"], average="macro", zero_division=0
    )


def get_macro_recall(row):
    return recall_score(row["labels"], row["preds"], average="macro", zero_division=0)


def get_top_1_accuracy(row):
    return accuracy_score(row["labels"], row["preds"])


def get_top_5_accuracy(row):
    return np.mean(row["top_5_tps"])


def get_macro_f1(row):
    return f1_score(row["labels"], row["preds"], average="macro", zero_division=0)


def get_top_k_accuracy_percent_low(row, k=1, percent=10):
    all_scores = row[f"tumbling_top_{k}_accuracy"]
    return np.sort(all_scores)[: round(len(all_scores) * percent / 100)].mean()


def get_tumbling_top_5_accuracy(row, window_size=100):
    top_5_tps = np.array(row["top_5_tps"])
    top_5_tps = top_5_tps[: -(len(top_5_tps) % window_size)].reshape(-1, window_size)
    return top_5_tps.mean(axis=-1)


def get_tumbling_top_1_accuracy(row, window_size=100):
    preds = np.array(row["preds"])
    labels = np.array(row["labels"])
    preds = preds[: -(len(preds) % window_size)].reshape(-1, window_size)
    labels = labels[: -(len(labels) % window_size)].reshape(-1, window_size)
    return (labels == preds).mean(axis=-1)


def get_metrics(df, metrics: List[str]):
    for metric in metrics:
        metric_fn = get_metric_fn(metric)
        df[metric] = df.apply(metric_fn, axis=1)
