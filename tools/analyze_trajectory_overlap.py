import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.utils import get_dataloader


BASEPATH = Path(__file__).parent.parent

Token = Tuple[int, int]


def lcs_length(a: Sequence[Token], b: Sequence[Token]) -> int:
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a

    previous = [0] * (len(short) + 1)
    for token_long in long:
        current = [0]
        prev_diag = 0
        for idx, token_short in enumerate(short, start=1):
            up = previous[idx]
            left = current[-1]
            value = prev_diag + 1 if token_long == token_short else max(up, left)
            current.append(value)
            prev_diag = up
        previous = current
    return previous[-1]


def load_report(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, orient="records")


def stream_key(row: Dict) -> Tuple:
    return (
        row["dataset"],
        optional_value(row, "n_users"),
        optional_value(row, "discretization_rows", 100),
        optional_value(row, "discretization_shape", "hex"),
        optional_value(row, "aggregation_mode", "grow"),
        optional_value(row, "grow_factor", 2),
        optional_value(row, "subsample"),
        optional_value(row, "subsample_mode", "first"),
        optional_value(row, "subsample_start", 0),
        optional_value(row, "drift_location"),
        optional_value(row, "drift_users"),
        optional_value(row, "pre_drift_lon_limit"),
    )


def optional_value(row: Dict, key: str, default=None):
    value = row.get(key, default)
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    return value


def iter_signatures(row: Dict, time_feature: str) -> Iterable[Tuple[List[Token], int]]:
    time_feature_idx = {"hour": 0, "3hour": 1, "6hour": 2}
    if time_feature not in time_feature_idx:
        raise ValueError(f"Unsupported time feature: {time_feature}")

    dataloader, _, _ = get_dataloader(
        dataset=row["dataset"],
        n_users=optional_value(row, "n_users"),
        discretization_rows=optional_value(row, "discretization_rows", 100),
        discretization_shape=optional_value(row, "discretization_shape", "hex"),
        aggregation_mode=optional_value(row, "aggregation_mode", "grow"),
        grow_factor=optional_value(row, "grow_factor", 2),
        batch_size=1,
        subsample=optional_value(row, "subsample"),
        subsample_mode=optional_value(row, "subsample_mode", "first"),
        subsample_start=optional_value(row, "subsample_start", 0),
        drift_location=optional_value(row, "drift_location"),
        drift_users=optional_value(row, "drift_users"),
        pre_drift_lon_limit=optional_value(row, "pre_drift_lon_limit"),
    )

    for x, t, _, u in dataloader:
        x_arr = x[0].numpy()
        t_arr = t[0].numpy()
        time_idx = time_feature_idx[time_feature]
        if x_arr.shape[1] < 1 or t_arr.shape[1] <= time_idx:
            raise ValueError(f"Expected x[:, 0] as POI and t[:, {time_idx}] as {time_feature}.")
        signature = list(zip(x_arr[:, 0].astype(int), t_arr[:, time_idx].astype(int)))
        yield signature, int(u.item())


def compute_prior_overlap(row: Dict, time_feature: str) -> pd.DataFrame:
    trajectories: List[List[Token]] = []
    users: List[int] = []
    token_index = defaultdict(list)
    records = []

    for idx, (signature, user) in enumerate(iter_signatures(row, time_feature)):
        candidate_ids = set()
        for token in set(signature):
            candidate_ids.update(token_index[token])

        best_lcs = 0
        best_user = None
        best_candidate_length = 0
        for candidate_idx in candidate_ids:
            if users[candidate_idx] == user:
                continue
            candidate_lcs = lcs_length(signature, trajectories[candidate_idx])
            if candidate_lcs > best_lcs:
                best_lcs = candidate_lcs
                best_user = users[candidate_idx]
                best_candidate_length = len(trajectories[candidate_idx])

        length = len(signature)
        records.append(
            {
                "trajectory_idx": idx,
                "label": user,
                "length": length,
                "best_prior_lcs": best_lcs,
                "best_prior_user": best_user,
                "overlap_current": best_lcs / length if length else 0.0,
                "overlap_min": (
                    best_lcs / min(length, best_candidate_length)
                    if best_lcs > 0
                    else 0.0
                ),
            }
        )

        trajectories.append(signature)
        users.append(user)
        for token in set(signature):
            token_index[token].append(idx)

    return pd.DataFrame.from_records(records)


def attach_performance(overlap_df: pd.DataFrame, row: Dict) -> pd.DataFrame:
    labels = np.array(row["labels"])
    preds = np.array(row["preds"])
    top5 = np.array(row["top_5_tps"])
    if len(overlap_df) != len(labels):
        raise ValueError(
            f"Report row has {len(labels)} predictions, but stream has {len(overlap_df)} trajectories."
        )

    result = overlap_df.copy()
    result["correct"] = labels == preds
    result["top5_correct"] = top5.astype(bool)
    result["pred"] = preds
    return result


def summarize_slice(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for key, group in df.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        labels = group["label"].to_numpy()
        preds = group["pred"].to_numpy()
        rows.append(
            dict(zip(group_cols, key))
            | {
                "n": len(group),
                "top1": float(group["correct"].mean()),
                "top5": float(group["top5_correct"].mean()),
                "macro_f1": float(
                    f1_score(labels, preds, average="macro", zero_division=0)
                ),
                "mean_length": float(group["length"].mean()),
                "mean_best_prior_lcs": float(group["best_prior_lcs"].mean()),
                "mean_overlap_current": float(group["overlap_current"].mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def add_overlap_bins(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["overlap_bin"] = np.select(
        [
            result["overlap_current"] == 0,
            result["overlap_current"] <= 0.5,
            result["overlap_current"] > 0.5,
        ],
        ["none", "partial_le_50", "high_gt_50"],
        default="unknown",
    )
    result["has_prior_overlap"] = result["best_prior_lcs"] > 0
    result["high_prior_overlap"] = result["overlap_current"] > 0.5
    return result


def add_overlap_percentages(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    denominator_cols = [
        col
        for col in [
            "dataset",
            "n_users",
            "model_cls",
            "embedding_type",
            "loc_levels",
            "loc_level",
            "n_hidden",
            "loc_embedding_factor",
        ]
        if col in result.columns
    ]
    total = result.groupby(denominator_cols, dropna=False, observed=True)["n"].transform(
        "sum"
    )
    result["percentage"] = 100 * result["n"] / total
    return result


def average_run_summaries(summary_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "n_users",
        "model_cls",
        "embedding_type",
        "loc_levels",
        "loc_level",
        "n_hidden",
        "loc_embedding_factor",
        "overlap_bin",
    ]
    value_cols = [
        "n",
        "top1",
        "top5",
        "macro_f1",
        "mean_length",
        "mean_best_prior_lcs",
        "mean_overlap_current",
    ]
    return (
        summary_df.groupby(group_cols, dropna=False, observed=True)[value_cols]
        .mean()
        .reset_index()
    )


def format_latex_table(df: pd.DataFrame) -> str:
    display = df.copy()
    display["Embedding"] = display["embedding_type"].map(
        {
            "lookup_concat": "POI-only",
            "lookup_weighted_concat": "MiLES",
        }
    ).fillna(display["embedding_type"])
    display["embedding_order"] = display["Embedding"].map({"POI-only": 0, "MiLES": 1}).fillna(99)
    display["Overlap bin"] = display["overlap_bin"].map(
        {
            "none": "0\\%",
            "partial_le_50": "$\\leq 50\\%$",
            "high_gt_50": "$>50\\%$",
        }
    ).fillna(display["overlap_bin"])
    display = display.sort_values(
        ["percentage", "embedding_order"], ascending=[False, True]
    )

    display["top1"] = 100 * display["top1"]
    display["top5"] = 100 * display["top5"]
    display["macro_f1"] = 100 * display["macro_f1"]
    display["n"] = display["n"].round().astype(int)

    columns = [
        "Overlap bin",
        "n",
        "percentage",
        "Embedding",
        "top1",
        "top5",
        "macro_f1",
    ]
    display = display[[col for col in columns if col in display.columns]]
    display = display.rename(
        columns={
            "dataset": "Dataset",
            "model_cls": "Model",
            "n": "N",
            "percentage": "Percent",
            "top1": "Acc@1",
            "top5": "Acc@5",
            "macro_f1": "Macro F1",
        }
    )
    return display.to_latex(index=False, float_format="%.2f")


def row_metadata(row: Dict) -> Dict:
    keys = [
        "dataset",
        "n_users",
        "model_cls",
        "embedding_type",
        "loc_levels",
        "loc_level",
        "n_hidden",
        "loc_embedding_factor",
        "seed",
    ]
    return {key: row.get(key) for key in keys}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quantify trajectory ambiguity from prior different-user LCS overlap of "
            "(POI, time) tuples and summarize TUL performance by overlap bin."
        )
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=BASEPATH / "reports" / "lookup_level0_embedding_dim_ablation.jsonl",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=BASEPATH / "reports" / "trajectory_overlap_lookup_level0_embedding_dim",
    )
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model-cls", action="append")
    parser.add_argument("--embedding-type", action="append")
    parser.add_argument("--n-users", type=int, action="append")
    parser.add_argument(
        "--time-feature",
        choices=["hour", "3hour", "6hour"],
        default="hour",
        help="Time component used in the (POI, time) LCS signature.",
    )
    args = parser.parse_args()

    report_df = load_report(args.report)
    if args.dataset:
        report_df = report_df[report_df["dataset"].isin(args.dataset)]
    if args.model_cls:
        report_df = report_df[report_df["model_cls"].isin(args.model_cls)]
    if args.embedding_type:
        report_df = report_df[report_df["embedding_type"].isin(args.embedding_type)]
    if args.n_users:
        report_df = report_df[report_df["n_users"].isin(args.n_users)]
    rows = report_df.to_dict(orient="records")
    if not rows:
        raise ValueError("No report rows matched the supplied filters.")

    overlap_cache = {}
    all_per_trajectory = []
    all_summaries = []

    for row_id, row in enumerate(rows):
        key = stream_key(row) + (args.time_feature,)
        if key not in overlap_cache:
            overlap_cache[key] = compute_prior_overlap(row, args.time_feature)

        per_traj = attach_performance(overlap_cache[key], row)
        per_traj = add_overlap_bins(per_traj)
        meta = row_metadata(row) | {"row_id": row_id}
        for meta_key, meta_value in meta.items():
            per_traj[meta_key] = meta_value
        per_traj["overlap_time_feature"] = args.time_feature

        summary = summarize_slice(
            per_traj,
            [
                "dataset",
                "n_users",
                "model_cls",
                "embedding_type",
                "loc_levels",
                "loc_level",
                "n_hidden",
                "loc_embedding_factor",
                "seed",
                "overlap_bin",
            ],
        )
        summary["row_id"] = row_id
        all_summaries.append(summary)
        all_per_trajectory.append(per_traj)

    per_trajectory_df = pd.concat(all_per_trajectory, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    aggregate_df = average_run_summaries(summary_df)
    aggregate_df = add_overlap_percentages(aggregate_df)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_prefix.with_suffix(".summary.csv")
    aggregate_path = args.out_prefix.with_suffix(".aggregate.csv")
    per_trajectory_path = args.out_prefix.with_suffix(".per_trajectory.csv.gz")

    summary_df.to_csv(summary_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    per_trajectory_df.to_csv(per_trajectory_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {per_trajectory_path}")
    print(format_latex_table(aggregate_df))


if __name__ == "__main__":
    main()
