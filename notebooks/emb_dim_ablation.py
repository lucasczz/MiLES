import pandas as pd
import numpy as np
from src.data.metrics import get_metrics

df = pd.read_json("../reports/lookup_level0_embedding_dim_ablation.jsonl", lines=True)
print("rows", len(df))
print(df.groupby("loc_embedding_factor").size())
metrics = ["top_1_accuracy", "top_5_accuracy", "macro_f1"]
get_metrics(df, metrics)

# dim = loc_embedding_factor * n_hidden
df["dim"] = (df["loc_embedding_factor"] * df["n_hidden"]).astype(int)

g_mean = df.groupby("dim")[metrics].mean() * 100
g_std = df.groupby("dim")[metrics].std() * 100

# Format each cell as "mean ± std"
formatted = pd.DataFrame(index=g_mean.index.sort_values(ascending=False))
col_labels = {"top_1_accuracy": "Acc@1", "top_5_accuracy": "Acc@5", "macro_f1": "F1"}
for m, label in col_labels.items():
    formatted[label] = [
        f"{g_mean.loc[d, m]:.2f} $\\pm$ {g_std.loc[d, m]:.2f}"
        for d in formatted.index
    ]

formatted.index.name = "Embedding dim. $d$"

print("\n" + formatted.to_latex(escape=False))
