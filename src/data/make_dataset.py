import datetime
import pickle
import numpy as np
import pandas as pd
import glob
import os.path
from pyproj import Transformer
from tqdm import tqdm
import os
from pathlib import Path
import geopandas as gpd

from src.data.hex_utils import (
    cell_distance,
    hexagonize,
    interpolate_cell_jumps,
    small_to_big,
)

BASEPATH = Path(__file__).parent.parent.parent

ACTIONS = {
    (0, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
    (0, -1): 3,
    (1, 0): 4,
    (-1, 0): 5,
    (1, -1): 6,
    (-1, 1): 7,
}


def read_plt(plt_file, index):
    points = pd.read_csv(
        plt_file,
        skiprows=6,
        header=None,
        names=[
            "lat",
            "lon",
            "zero",
            "alt",
            "num-date",
            "date",
            "time",
        ],
        usecols=[
            "lat",
            "lon",
            "date",
            "time",
        ],
    )
    points["datetime"] = pd.to_datetime(
        points["date"] + " " + points["time"], format="%Y-%m-%d %H:%M:%S"
    )
    points = points.drop(columns=["date", "time"])
    points["trajectory"] = index
    return points


def read_user(user_folder):
    plt_files = glob.glob(os.path.join(user_folder, "Trajectory", "*.plt"))
    df = pd.concat([read_plt(f, idx) for idx, f in enumerate(plt_files)])
    return df


def read_all_users(folder):
    subfolders = os.listdir(folder)
    dfs = []
    for sf in tqdm(subfolders, desc="Processing user folders"):
        df = read_user(os.path.join(folder, sf))
        df["user"] = int(sf)
        dfs.append(df)
    return pd.concat(dfs)


def fix_datetime(df):
    mask = (df["datetime"] == df["datetime"].shift(1)).values
    if not mask.any():
        return df

    idx_last_non_na = df.index[~mask][-1]
    step = (df["datetime"].iloc[idx_last_non_na] - df["datetime"].iloc[0]) / len(df)

    for n, idx in enumerate(range(idx_last_non_na + 1, len(df))):
        df.loc[idx, "datetime"] += step * (n + 1)
    mask[idx_last_non_na:] = False

    df.loc[:, "datetime"] = df["datetime"].mask(mask)
    df.loc[:, "datetime"] = df["datetime"].interpolate()
    return df


def prep_geolife(limits, crs="epsg:2333"):
    gdf = read_all_users(
        BASEPATH.joinpath("data", "raw", "Geolife Trajectories 1.3", "Data")
    )
    gdf = gpd.GeoDataFrame(
        gdf, geometry=gpd.points_from_xy(gdf.lon, gdf.lat, crs="epsg:4326")
    )

    # project .to_crs("epsg:2333")
    print("Projecting geocoordinates...")
    gdf = gdf.to_crs(crs)
    print("Removing points outside of provided limits...")
    gdf = gdf.cx[limits[0].x : limits[1].x, limits[0].y : limits[1].y]
    gdf = gdf.reset_index(drop=True)

    xy = gdf["geometry"].get_coordinates()
    gdf = gdf.drop(columns=["geometry", "lon", "lat"])
    gdf = pd.concat([gdf, xy], axis=1)
    gdf["t_idx"] = gdf.groupby(["user", "trajectory"]).ngroup()

    # Split trajectories that span more than 3 hours
    tmp = []
    for t_idx, tdf in gdf.groupby("t_idx"):
        traj_start = tdf["datetime"].iloc[0]
        tdf["sub_traj"] = (tdf["datetime"] - traj_start) // datetime.timedelta(hours=3)
        tmp.append(tdf)
    gdf = pd.concat(tmp)
    gdf = gdf.sort_values(by=["user", "datetime"])
    gdf["t_idx"] = gdf.groupby(["t_idx", "sub_traj"]).ngroup()

    # Get time between records
    print("Computing time differences...")
    gdf["timediff"] = gdf.groupby("t_idx")["datetime"].transform(
        lambda x: x - x.shift(1)
    )
    gdf["timediff"] = gdf["timediff"].fillna(pd.Timedelta(seconds=2))

    # Remove rows with identical timestamps
    gdf = gdf[gdf["timediff"] > 0]

    # Get distance between records
    print("Calculating distances...")
    tmp = []
    for t_idx, tdf in gdf.groupby("t_idx"):
        xy = tdf[["x", "y"]]
        tdf["dist"] = np.linalg.norm(xy - xy.shift(1), axis=-1)
        tmp.append(tdf)
    gdf = pd.concat(tmp)
    gdf["dist"] = gdf["dist"].fillna(0)

    # Calculate speed
    gdf["speed"] = gdf["dist"] / gdf["timediff"].dt.total_seconds()

    # Filter out all trajectories where the average speed is above 30 m/s $\approx$ chinese driving speed limit
    speed_mask = gdf.groupby(["t_idx"])["speed"].mean() <= 30
    valid_traj_idx = speed_mask[speed_mask].index
    gdf = gdf[gdf["t_idx"].isin(valid_traj_idx)]

    return gdf


if __name__ == "__main__":

    crs1 = "epsg:4326"
    crs2 = "epsg:3857"
    limits = np.array([[116.1, 39.7], [116.7, 40.1]])
    n_rows = 100

    transformer = Transformer.from_crs(crs1, crs2)

    gdf_path = BASEPATH.joinpath("data", "processed", "geolife.pkl")
    if not gdf_path.is_file:
        gdf = prep_geolife(limits, crs2)
        gdf.to_pickle(BASEPATH.joinpath("data", "processed", "geolife.pkl"))
    else:
        with open(gdf_path, "rb") as f:
            gdf = pickle.load(f)

    print("Hexagonizing trajectories...")
    hdf = hexagonize(gdf, n_rows=n_rows, limits=limits)
    hdf["cell0"] = hdf.groupby(["q", "r"]).ngroup()

    hdf = hdf.rename(columns={"q": "q0", "r": "r0"})

    # Remove consecutive rows with identical cell coordinates 
    cells = hdf[['q0', 'r0']]
    hdf['cell_dist'] = cell_distance(cells.values, cells.shift().values)
    hdf['cell_dist'] = hdf['cell_dist'].fillna(1)
    hdf = hdf[hdf['cell_dist'] >= 1]

    for level in tqdm(list(range(1, 4)), desc="Computing high-level cells..."):
        q_new, r_new = small_to_big(
            qr=hdf[[f"q{level-1}", f"r{level-1}"]].values, radius=1
        ).T
        hdf[f"q{level}"] = q_new
        hdf[f"r{level}"] = r_new
        hdf[f"cell{level}"] = hdf.groupby([f"q{level}", f"r{level}"]).ngroup()

    print("Assigning time labels...")
    hdf['weekday'] = hdf['datetime'].dt.day_of_week
    hdf["is_workday"] = (hdf["weekday"] < 5).astype(int)
    hdf['timestamp'] = (hdf['datetime'] - hdf['datetime'].min()).astype('int64') // 1e9
    hour_thresholds = range(0, 25, 6)

    for idx in range(len(hour_thresholds) - 1):
        hdf[f"is_in_time_{idx}"] = hdf["datetime"].apply(
            lambda x: hour_thresholds[idx] < x.hour <= hour_thresholds[idx + 1]
        )

    time_features = [f"is_in_time_{idx}" for idx in range(4)] + ["is_workday"]
    hdf["time_label"] = hdf[time_features].astype(int).values @ np.arange(5)

    print("Saving data...")
    hdf.to_pickle(BASEPATH.joinpath("data", "processed", f"geolife_hex_{n_rows}.pkl"))
