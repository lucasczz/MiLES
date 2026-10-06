"""Figure: per-level forward-contribution share over the online stream, MiLES vs -WL.

For each location level l we track the identifiable per-level forward contribution
magnitude g_l(t) = w_l(t) * mean||z_l||(t) * mean||W_l||_col(t), where w_l is the
level weight (1 for -WL), ||z_l|| the embedding row norm, and ||W_l||_col the mean
column norm of the first RNN layer's input block for that level. g_l is invariant
to the rescaling symmetry (w_l <-> z_l and w_l <-> W_l), so its share across levels
is a well-defined per-level weighting. MiLES re-weights the levels via the single
scalar w_l; -WL can only do so through the high-dimensional block W_l and stays
near its initialization profile.

Paper house style (viridis, twin-panel) to match fig:level_weights.
Reads reports/level_gain_{miles,nowl}.jsonl, writes pub/figs/level_gain.pdf.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
OUT = ROOT / "pub" / "figs" / "level_gain.pdf"


def load(path):
    rows = {}
    for line in Path(path).open():
        if line.strip():
            r = json.loads(line)
            rows[r["seed"]] = r
    return rows


def share_traj(rows, weighted):
    """Mean-over-seeds per-level share of g_l(t). Returns (T, 4)."""
    shares = []
    for s in sorted(rows):
        r = rows[s]
        W = np.stack([r[f"input_weight_norm_{i}"] for i in range(4)], 1)
        E = np.stack([r[f"embedding_norm_{i}"] for i in range(4)], 1)
        wl = (
            np.stack([r[f"embedding_weight_{i}"] for i in range(4)], 1)
            if weighted
            else np.ones_like(W)
        )
        g = wl * E * W
        shares.append(g / g.sum(1, keepdims=True))
    return np.stack(shares).mean(0)


def smooth(a, w=1500):
    """Rolling mean so the per-step share is legible."""
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="valid")


def main():
    miles = load(REPORTS / "level_gain_miles.jsonl")
    nowl = load(REPORTS / "level_gain_nowl.jsonl")
    sm = share_traj(miles, True)
    sn = share_traj(nowl, False)

    cmap = plt.get_cmap("viridis", 4)
    colors = cmap(range(4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.5), sharey=True)
    for ax, sh, title in [
        (ax1, sm, "MiLES"),
        (ax2, sn, r"$-$WL"),
    ]:
        for i in range(4):
            m = smooth(sh[:, i])
            x = np.arange(len(m))
            ax.plot(x, m, label=rf"$g_{i}$", color=colors[i])
        ax.set_xlim(0, x.max())
        ax.set_xlabel("# of Trajectories")
        ax.set_title(title)
        ax.set_xticks(ax.get_xticks()[:-1:2])
    ax1.set_ylim(top=0.46)
    ax1.set_ylabel("Level Contrib. Share")
    ax1.legend(
        loc="upper center",
        frameon=False,
        ncol=4,
        fontsize="small",
        borderaxespad=0.3,
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.2,
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
