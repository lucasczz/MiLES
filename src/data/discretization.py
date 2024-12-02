from matplotlib import patches
import numpy as np
import pandas as pd


def group_cells(qr, shape="hex", radius=1):
    if shape == "hex":
        qrs = np.concatenate([qr, -qr.sum(axis=-1)[..., None]], axis=-1)
        area = 3 * radius**2 + 3 * radius + 1
        shift = 3 * radius + 2

        temp_qrs = (np.roll(qrs, -1, axis=-1) + shift * qrs) // area
        qrs_big = (1 + temp_qrs - np.roll(temp_qrs, -1, axis=-1)) // 3
        return qrs_big[:, :-1]
    elif shape == "square" or shape == "diamond":
        qr_big = np.floor_divide(qr, (3**radius))
        return qr_big


def get_hex_size(limits, n_rows):
    return (limits[1, 1] - limits[0, 1]) / ((n_rows + 0.5) * np.sqrt(3))


def discretize_coordinates(df, n_rows, shape="hex", col_suffix=""):
    limits = np.array(
        [[df["lon"].min(), df["lat"].min()], [df["lon"].max(), df["lat"].max()]]
    )

    xy = df[["lon", "lat"]].values
    xy = xy - limits[0]
    df_disc = df.copy()

    if shape == "hex":
        hex_size = get_hex_size(limits=limits, n_rows=n_rows)
        qr = coords_to_hex(xy, hex_size)
    elif shape == "square":
        square_size = (limits[1, 1] - limits[0, 1]) / n_rows
        qr = coords_to_square(xy, square_size)
    elif shape == "diamond":
        square_size = (limits[1, 1] - limits[0, 1]) / n_rows
        qr = coords_to_diamond(xy, square_size)

    df_disc[f"q{col_suffix}"] = qr[:, 0]
    df_disc[f"r{col_suffix}"] = qr[:, 1]
    return df_disc


def coords_to_square(coords: np.ndarray, square_size: float):
    return np.floor_divide(coords, square_size)


def coords_to_diamond(coords: np.ndarray, square_size: float):
    rot_angle = np.pi / 4
    a = np.array(
        [
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)],
        ]
    )
    coords_rot = coords @ a.T
    return np.floor_divide(coords_rot, square_size)


def coords_to_hex(coords: np.ndarray, hex_size: float):
    """Converts geo-coordinates to axial hexgrid coordinates

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of shape (n, 2) to convert to match to hexgrid.
    hex_size : float
        Size of hexagons.

    Returns
    -------
    Axial coordinates of form [q, r]
    """
    a = np.array([[2, -1], [0, np.sqrt(3)]]) / 3
    fracs = coords @ a / hex_size
    return cell_round(fracs)


def hex_to_coords(hex, hex_size, coord_offset):
    a = np.array([[1.5, 0.5 * np.sqrt(3)], [0, np.sqrt(3)]])
    return hex @ a * hex_size + coord_offset


def cell_distance(cell1, cell2):
    diff = np.abs(cell1 - cell2).sum(axis=-1)
    if cell1.shape[-1] != 3:
        diff += np.abs(cell1.sum(-1) - cell2.sum(-1))
    return diff / 2


def cell_round(frac):
    """_summary_

    Parameters
    ----------
    frac : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    rounded = np.round(frac)
    round_diffs = frac - rounded
    abs_diffs = np.abs(round_diffs)
    mask = abs_diffs[..., 0] > abs_diffs[..., 1]
    rounded[mask, 0] += np.round(round_diffs[mask, 0] + 0.5 * round_diffs[mask, 1])
    rounded[~mask, 1] += np.round(round_diffs[~mask, 1] + 0.5 * round_diffs[~mask, 0])

    return rounded.astype(int)


def interpolate_cells(a, b):
    n = cell_distance(a, b)
    tmp = []
    for i, ni in enumerate(n):
        if np.isnan(ni) or ni <= 1:
            continue
        t = np.arange(1, ni) / ni
        tmp.append(a[i] + (b - a)[i] * t[:, None])
    return cell_round(np.concatenate(tmp))


def interpolate_cell_jumps(df):
    qr1 = df[["q", "r"]].shift(1).values
    qr2 = df[["q", "r"]].values

    df["cell_dist"] = cell_distance(qr1, qr2)
    df["cell_dist"] = df["cell_dist"].fillna(1)
    if not (df["cell_dist"] > 1).any():
        return df
    else:
        t1 = df[["datetime"]].shift(1).values
        t2 = df[["datetime"]].values

        qri = []
        ti = []

        for i, ni in enumerate(df["cell_dist"]):
            if not np.isnan(ni) and ni > 1:
                t = np.arange(1, ni) / ni
                ti.append(t1[i] + (t2[i] - t1[i]) * t[:, None])
                qri.append(qr1[i] + (qr2[i] - qr1[i]) * t[:, None])
        new_rows = pd.DataFrame(cell_round(np.concatenate(qri)), columns=["q", "r"])
        new_rows["datetime"] = np.concatenate(ti)
        new_rows["t_idx"] = df.iloc[0]["t_idx"]
        new_rows["user"] = df.iloc[0]["user"]
        df = df[df["cell_dist"] >= 1]
        return pd.concat([df, new_rows]).sort_values("datetime")


def get_hex_traj(traj, n_rows, limits):
    size = get_hex_size(limits, n_rows)
    xys = hex_to_coords(traj, size, limits[0])
    hexagons = [
        patches.RegularPolygon(
            xy,
            numVertices=6,
            radius=size,
            orientation=np.radians(30),
            edgecolor="k",
            facecolor=(0.1, 0.2, 0.8, 0.5),
            linewidth=0.5,
        )
        for xy in xys
    ]

    return hexagons


def get_hexgrid(
    limits,
    patch=None,
    n_rows=None,
    size=None,
):
    if size is None:
        size = get_hex_size(limits, n_rows)
    xstep = 3 / 2 * size
    ystep = np.sqrt(3) * size

    step = np.array([xstep, ystep])
    offset = (patch[0] - limits[0]) // step * step
    x = np.arange(limits[0, 0] + offset[0], patch[1, 0], xstep)  # (n_cols)
    y = np.arange(limits[0, 1] + offset[1], patch[1, 1], ystep)  # (n_rows)

    n_cols, n_rows = len(x), len(y)
    xs, ys = np.meshgrid(x, y)  # (n_rows, n_cols), (n_rows, n_cols)
    xy = np.stack([xs, ys], axis=-1)  # (n_rows, n_cols, 2)
    odd_cols = np.arange(1, n_cols, 2)
    xy[:, odd_cols, 1] += ystep / 2
    xy = xy.reshape(-1, 2)

    hexagons = [
        patches.RegularPolygon(
            xyi,
            numVertices=6,
            radius=size,
            orientation=np.radians(30),
            edgecolor="k",
            facecolor="none",
            linewidth=0.5,
        )
        for xyi in xy
    ]

    return hexagons
