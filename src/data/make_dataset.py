import numpy as np
import pandas as pd
import glob
import os.path
from tqdm import tqdm
import os
import geopandas as gpd

from src.data.hex_utils import (
    hexagonize,
    interpolate_cell_jumps,
)


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
    gdf = read_all_users("data/raw/Geolife Trajectories 1.3/Data")
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
    gdf = gdf.assign(t_idx=gdf.groupby(["user", "trajectory"]).ngroup() + 1)

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

    crs = "epsg:2333"
    lonlimits = [116.1, 116.7]
    latlimits = [39.7, 40.1]
    n_rows = 50

    limits = gpd.points_from_xy(x=lonlimits, y=latlimits, crs="epsg:4326")
    limits = limits.to_crs(crs)

    gdf = prep_geolife(limits, crs)
    gdf.to_pickle("data/processed/geolife.pkl")

    hdf = hexagonize(gdf, n_rows=n_rows, limits=limits)
    tmp = []
    for t_idx, dft in hdf.groupby("t_idx"):
        tmp.append(interpolate_cell_jumps(dft))
    hdf = pd.concat(tmp)
    hdf = hdf.groupby("t_idx").filter(lambda x: len(x) >= 3)

    hdf["is_workday"] = hdf["datetime"].apply(lambda x: x.weekday() < 5)
    hour_thresholds = range(0, 25, 6)

    for idx in range(len(hour_thresholds) - 1):
        hdf[f"is_in_time_{idx}"] = hdf["datetime"].apply(
            lambda x: hour_thresholds[idx] < x.hour <= hour_thresholds[idx + 1]
        )

    tmp = []
    q_max = hdf["q"].max()
    r_max = hdf["r"].max()
    for t_idx, dfi in hdf.groupby("t_idx"):
        dfnew = pd.concat([dfi.head(1), dfi, dfi.tail(1)])
        q_idx = dfnew.columns.get_loc("q")
        r_idx = dfnew.columns.get_loc("r")
        dfnew.iloc[0, q_idx] = q_max + 1
        dfnew.iloc[-1, q_idx] = q_max + 2
        dfnew.iloc[0, r_idx] = r_max + 1
        dfnew.iloc[-1, r_idx] = r_max + 2
        tmp.append(dfnew)
    hdf = pd.concat(tmp)

    hdf.to_pickle(f"data/processed/geolife_hex_{n_rows}.pkl")

    tmp = []
    for t_idx, tdf in hdf.groupby("t_idx"):
        tdf_shift = tdf.shift(-1)
        tdf["delta_q"] = tdf_shift["q"] - tdf["q"]
        tdf["delta_r"] = tdf_shift["r"] - tdf["r"]
        tmp.append(tdf)
    ndf = pd.concat(tmp)
    ndf["move"] = ndf.apply(
        lambda row: ACTIONS.get((row["delta_q"], row["delta_r"]), 1), axis=1
    )
    ndf.loc[ndf["q"] == 61, "move"] = 0
    ndf = ndf[ndf["q"] != 62]

    ndf.to_pickle("../data/processed/geolife_hex_50_moves.pkl")
