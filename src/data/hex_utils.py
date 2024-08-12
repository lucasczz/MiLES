import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import Polygon
import folium

KEYS = {"axial": ["q", "r"], "oddq": ["col", "row"]}
STYLE_GRID = {"fillOpacity": "0.0", "weight": 1}
STYLE_VISITED = {
    "fillOpacity": "0.2",
    "weight": 1,
    "color": "orange",
    "fillColor": "orange",
}


def hexagonize(df, n_rows, limits):
    lonlat_offset = np.array([limits[0].x, limits[0].y])
    hex_size = get_hex_size(limits=limits, n_rows=n_rows)

    xy = df[["x", "y"]].values
    qr = coords_to_hex(xy, hex_size, lonlat_offset)
    hdf = pd.DataFrame(qr, columns=["q", "r"])
    hdf["datetime"] = df["datetime"]
    hdf["t_idx"] = df["t_idx"]
    hdf["user"] = df["user"]
    return hdf


def coords_to_hex(coords: np.ndarray, hex_size: float, coord_offset: np.ndarray):
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
    pixels = coords - coord_offset
    a = np.array([[2, -1], [0, np.sqrt(3)]]) / 3
    fracs = pixels @ a / hex_size
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
        df = df[df["cell_dist"] >= 1]
        return pd.concat([df, new_rows]).sort_values("datetime")


def get_hex_size(limits, n_rows):
    return (limits[1].y - limits[0].y) / ((n_rows + 0.5) * np.sqrt(3))


def make_hexgrid(limits, size):
    xstep = 3 / 2 * size
    ystep = np.sqrt(3) * size

    x = np.arange(limits[0].x, limits[1].x - xstep, xstep)  # (n_cols)
    y = np.arange(limits[0].y, limits[1].y - ystep, ystep)  # (n_rows)

    n_cols, n_rows = len(x), len(y)
    xs, ys = np.meshgrid(x, y)  # (n_rows, n_cols), (n_rows, n_cols)
    xy = np.stack([xs, ys], axis=-1)  # (n_rows, n_cols, 2)
    odd_cols = np.arange(1, n_cols, 2)
    xy[:, odd_cols, 1] += ystep / 2
    xy = xy.reshape(-1, 2)

    oddq = np.array(
        [[col, row] for row in range(n_rows) for col in range(n_cols)]
    )  # (n_cols * n_rows, 2)

    axial = oddq_to_axial(oddq)
    all_coords = np.concatenate([axial, oddq], axis=-1)

    stepsize = np.array([0.5, 0.5 * np.sqrt(3)]) * size  # (2)
    steps = np.array([[0, 1], [1, 2], [3, 2], [4, 1], [3, 0], [1, 0]])  # (6, 2)
    hexes = xy[..., None, :] + (steps * stepsize)
    polys = [Polygon(hex) for hex in hexes]

    return gpd.GeoDataFrame(
        data=all_coords,
        columns=["q", "r", "col", "row"],
        geometry=polys,
        crs=limits.crs,
    )


def oddq_to_axial(oddq):
    """Convert offset coordinates to axial coordinates.

    Parameters
    ----------
    oddq : _type_
        Coordinates of form [col, row]

    Returns
    -------
    axial
        Coordinates of form [q, r]
    """
    axial_offset = np.zeros_like(oddq)
    axial_offset[:, 1] = (oddq[:, 0] - oddq[:, 0] % 2) / 2
    return oddq - axial_offset


def axial_to_oddq(axial):
    oddq_offset = np.zeros_like(axial)
    oddq_offset[:, 1] = (axial[:, 0] - axial[:, 0] % 2) / 2
    return axial + oddq_offset


def plot_traj(grid: gpd.GeoDataFrame, axial=None, oddq=None):
    if axial is not None:
        keys = KEYS["axial"]
        idx = axial
    else:
        keys = KEYS["oddq"]
        idx = oddq

    map = folium.Map(location=(40.2, 116.383331), zoom_start=8.2)
    # grid = grid.set_index(keys)
    pgrid = grid.to_crs("epsg:4326")
    pgrid = pgrid.set_index(keys)
    pgrid["visited"] = 0
    pgrid.loc[idx.tolist(), "visited"] = 1
    pgrid = pgrid.reset_index(names=keys)

    grid_json = pgrid.to_json()
    tooltip = folium.GeoJsonTooltip(fields=keys)

    def get_style(feature):
        props = feature["properties"]
        if [props[key] for key in keys] in idx.tolist():
            return STYLE_VISITED
        else:
            return STYLE_GRID

    grid_layer = folium.GeoJson(
        grid_json,
        name="non-visited",
        style_function=get_style,
        tooltip=tooltip,
    )
    grid_layer.add_to(map)
    return map
