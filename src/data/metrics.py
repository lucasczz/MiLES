import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(df):
    df["macro_precision"] = df.apply(
        lambda row: precision_score(
            row["labels"], row["preds"], average="macro", zero_division=0
        ),
        axis=1,
    )
    df["macro_recall"] = df.apply(
        lambda row: recall_score(
            row["labels"], row["preds"], average="macro", zero_division=0
        ),
        axis=1,
    )
    df["macro_f1"] = df.apply(
        lambda row: f1_score(
            row["labels"], row["preds"], average="macro", zero_division=0
        ),
        axis=1,
    )
    df["top_1_accuracy"] = df.apply(
        lambda row: accuracy_score(row["labels"], row["preds"]), axis=1
    )
    df["top_5_accuracy"] = df.apply(lambda row: np.mean(row["top_5_tps"]), axis=1)


