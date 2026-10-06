import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.utils import get_discretization, get_trajectories, load_data


BASEPATH = Path(__file__).parent.parent
SHARED_CELL_LEVELS = ("cell1", "cell2", "cell3")


def optional_value(row: Dict, key: str, default=None):
    value = row.get(key, default)
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    return value


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
    )


def normalized_entropy(values: pd.Series) -> float:
    counts = values.value_counts(dropna=False)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts.to_numpy(dtype=float) / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(counts)))


def cell_entropy_maps(
    df: pd.DataFrame,
    levels: Tuple[str, ...] = SHARED_CELL_LEVELS,
) -> Dict[str, Dict[str, Dict[int, float]]]:
    maps: Dict[str, Dict[str, Dict[int, float]]] = {"category": {}, "user": {}}
    for level in levels:
        maps["category"][level] = (
            df.groupby(level, sort=False)["venueCategoryId"]
            .apply(normalized_entropy)
            .to_dict()
        )
        maps["user"][level] = (
            df.groupby(level, sort=False)["user"].apply(normalized_entropy).to_dict()
        )
    return maps


def reconstruct_quantized_stream(row: Dict) -> Tuple[pd.DataFrame, List[Tuple]]:
    if optional_value(row, "drift_location") is not None:
        raise ValueError("Cell heterogeneity analysis does not support drifted streams.")

    df, x_features, t_features, _ = load_data(
        dataset=row["dataset"],
        n_users=optional_value(row, "n_users"),
    )
    df_quantized = get_discretization(
        df,
        n_rows=optional_value(row, "discretization_rows", 100),
        shape=optional_value(row, "discretization_shape", "hex"),
        aggregation_mode=optional_value(row, "aggregation_mode", "grow"),
        grow_factor=optional_value(row, "grow_factor", 2),
    )
    trajectories = get_trajectories(
        df=df_quantized,
        x_features=x_features,
        t_features=t_features,
        subsample=optional_value(row, "subsample"),
        subsample_mode=optional_value(row, "subsample_mode", "first"),
        subsample_start=optional_value(row, "subsample_start", 0),
    )
    return df_quantized, trajectories


def compute_trajectory_heterogeneity(row: Dict) -> pd.DataFrame:
    df_quantized, trajectories = reconstruct_quantized_stream(row)
    entropy_maps = cell_entropy_maps(df_quantized)
    groups = list(df_quantized.groupby("t_idx", sort=False))

    subsample = optional_value(row, "subsample")
    if subsample:
        start = optional_value(row, "subsample_start", 0)
        if optional_value(row, "subsample_mode", "first") == "first":
            groups = groups[start : start + subsample]
        else:
            groups = groups[start + subsample :]

    if len(groups) != len(trajectories):
        raise ValueError(
            f"Reconstructed {len(trajectories)} trajectories, but found {len(groups)} grouped trajectories."
        )

    records = []
    for trajectory_idx, ((t_idx, group), trajectory) in enumerate(zip(groups, trajectories)):
        xs, _, _, user = trajectory
        category_scores = []
        user_scores = []
        for level in SHARED_CELL_LEVELS:
            cells = group[level].to_numpy()
            category_scores.extend(
                entropy_maps["category"][level][int(cell)] for cell in cells
            )
            user_scores.extend(entropy_maps["user"][level][int(cell)] for cell in cells)

        records.append(
            {
                "trajectory_idx": trajectory_idx,
                "t_idx": t_idx,
                "label": int(user),
                "length": int(len(xs)),
                "category_heterogeneity": float(np.mean(category_scores)),
                "user_heterogeneity": float(np.mean(user_scores)),
                "heterogeneity": float(
                    np.mean([np.mean(category_scores), np.mean(user_scores)])
                ),
                "mean_cell1": float(group["cell1"].mean()),
                "mean_cell2": float(group["cell2"].mean()),
                "mean_cell3": float(group["cell3"].mean()),
            }
        )

    return pd.DataFrame.from_records(records)


def add_heterogeneity_bins(
    df: pd.DataFrame,
    score_col: str = "heterogeneity",
) -> pd.DataFrame:
    result = df.copy()
    quantiles = result[score_col].quantile([1 / 3, 2 / 3]).to_numpy()
    if np.isclose(quantiles[0], quantiles[1]):
        result["heterogeneity_bin"] = pd.qcut(
            result[score_col].rank(method="first"),
            q=3,
            labels=["low", "medium", "high"],
        )
    else:
        result["heterogeneity_bin"] = pd.cut(
            result[score_col],
            bins=[-np.inf, quantiles[0], quantiles[1], np.inf],
            labels=["low", "medium", "high"],
            include_lowest=True,
        )
    return result


def attach_performance(heterogeneity_df: pd.DataFrame, row: Dict) -> pd.DataFrame:
    labels = np.array(row["labels"])
    preds = np.array(row["preds"])
    top5 = np.array(row["top_5_tps"])
    if len(heterogeneity_df) != len(labels):
        raise ValueError(
            f"Report row has {len(labels)} predictions, but stream has {len(heterogeneity_df)} trajectories."
        )

    result = heterogeneity_df.copy()
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
                "mean_heterogeneity": float(group["heterogeneity"].mean()),
                "mean_category_heterogeneity": float(
                    group["category_heterogeneity"].mean()
                ),
                "mean_user_heterogeneity": float(group["user_heterogeneity"].mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def add_percentages(df: pd.DataFrame) -> pd.DataFrame:
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


def format_latex_table(
    df: pd.DataFrame,
    score_col: str = "user_heterogeneity",
) -> str:
    display = df.copy()
    score_labels = {
        "heterogeneity": "Heterogeneity",
        "category_heterogeneity": "Category diversity",
        "user_heterogeneity": "User diversity",
    }
    bin_label = score_labels[score_col]
    display["Embedding"] = display["embedding_type"].map(
        {
            "lookup_concat": "POI-only",
            "lookup_weighted_concat": "MiLES",
        }
    ).fillna(display["embedding_type"])
    display["embedding_order"] = display["Embedding"].map(
        {"POI-only": 0, "MiLES": 1}
    ).fillna(99)
    display[bin_label] = display["heterogeneity_bin"].map(
        {"low": "Low", "medium": "Medium", "high": "High"}
    ).fillna(display["heterogeneity_bin"])
    display["bin_order"] = display["heterogeneity_bin"].map(
        {"low": 0, "medium": 1, "high": 2}
    )
    display = display.sort_values(["bin_order", "embedding_order"])

    for col in ["top1", "top5", "macro_f1"]:
        display[col] = 100 * display[col]
    display["n"] = display["n"].astype(int)

    display = display[
        [
            bin_label,
            "Embedding",
            "top1",
            "top5",
            "macro_f1",
        ]
    ].rename(
        columns={
            "top1": "Acc@1",
            "top5": "Acc@5",
            "macro_f1": "Macro F1",
        }
    )
    return display.to_latex(index=False, float_format="%.2f")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize TUL performance by user-label diversity of MiLES shared "
            "grid cells."
        )
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=BASEPATH / "reports" / "TUL.jsonl",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=BASEPATH / "reports" / "user_diversity_TUL_BiTULER_NYC",
    )
    parser.add_argument("--dataset", default="foursquare_NYC")
    parser.add_argument("--n-users", type=int, default=800)
    parser.add_argument("--model-cls", default="BiTULER")
    parser.add_argument(
        "--embedding-type",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--score-col",
        choices=["heterogeneity", "category_heterogeneity", "user_heterogeneity"],
        default="user_heterogeneity",
        help="Trajectory heterogeneity score used for tertile bins.",
    )
    parser.add_argument(
        "--include-replay",
        action="store_true",
        help="Include replay-buffer variants instead of limiting to the base online runs.",
    )
    args = parser.parse_args()
    embedding_types = args.embedding_type or [
        "lookup_concat",
        "lookup_weighted_concat",
    ]

    report_df = load_report(args.report)
    report_df = report_df[
        (report_df["dataset"] == args.dataset)
        & (report_df["n_users"] == args.n_users)
        & (report_df["model_cls"] == args.model_cls)
        & (report_df["embedding_type"].isin(embedding_types))
    ]
    if not args.include_replay and "replay_buffer_type" in report_df.columns:
        report_df = report_df[
            report_df["replay_buffer_type"].isna()
            | (report_df["replay_buffer_type"] == "none")
        ]
    rows = report_df.to_dict(orient="records")
    if not rows:
        raise ValueError("No report rows matched the supplied filters.")

    heterogeneity_cache = {}
    all_per_trajectory = []
    all_summaries = []

    for row_id, row in enumerate(rows):
        key = stream_key(row)
        if key not in heterogeneity_cache:
            heterogeneity_cache[key] = add_heterogeneity_bins(
                compute_trajectory_heterogeneity(row),
                score_col=args.score_col,
            )

        per_traj = attach_performance(heterogeneity_cache[key], row)
        meta = row_metadata(row) | {"row_id": row_id}
        for meta_key, meta_value in meta.items():
            per_traj[meta_key] = meta_value
        per_traj["heterogeneity_score_col"] = args.score_col

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
                "heterogeneity_bin",
            ],
        )
        summary["row_id"] = row_id
        all_summaries.append(summary)
        all_per_trajectory.append(per_traj)

    per_trajectory_df = pd.concat(all_per_trajectory, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    aggregate_df = summarize_slice(
        per_trajectory_df,
        [
            "dataset",
            "n_users",
            "model_cls",
            "embedding_type",
            "loc_levels",
            "loc_level",
            "n_hidden",
            "loc_embedding_factor",
            "heterogeneity_bin",
        ],
    )
    aggregate_df = add_percentages(aggregate_df)

    bin_counts = aggregate_df.groupby("heterogeneity_bin", observed=True)["n"].sum()
    if (bin_counts == 0).any() or len(bin_counts) != 3:
        raise ValueError(f"Expected non-empty low/medium/high bins, got {bin_counts}.")

    percentage_sums = aggregate_df.groupby(
        ["embedding_type", "loc_levels", "n_hidden", "loc_embedding_factor"],
        dropna=False,
        observed=True,
    )["percentage"].sum()
    if not np.allclose(percentage_sums.to_numpy(), 100.0):
        raise ValueError(f"Percentages do not sum to 100: {percentage_sums}.")

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
    print(format_latex_table(aggregate_df, score_col=args.score_col))


if __name__ == "__main__":
    main()
