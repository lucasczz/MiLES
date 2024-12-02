import datetime
import numpy as np
import pandas as pd
import glob
import os.path
from tqdm import tqdm
import os
from pathlib import Path
import geopandas as gpd


def split_trajectories(df, max_hours, group_cols=["user"]):
    # Initialize columns for sub-trajectory index and t_idx
    df = df.sort_values(by=["user", "datetime"]).copy()
    max_duration = pd.Timedelta(hours=max_hours)
    sub_trajectories = []
    t_idx_values = []

    # Process each user individually with progress bar
    for user_id, user_data in tqdm(
        df.groupby(group_cols), desc="Splitting trajectories..."
    ):
        sub_trajectory_index = 0
        start_time = user_data["datetime"].iloc[0]

        # Store sub-trajectory indices for the user
        sub_trajectory_ids = []

        # Iterate over each row in the user's trajectory
        for current_time in user_data["datetime"]:
            # Start a new sub-trajectory if the duration exceeds max_dur
            if current_time - start_time > max_duration:
                sub_trajectory_index += 1
                start_time = current_time

            sub_trajectory_ids.append(sub_trajectory_index)
        # Append results to main list
        sub_trajectories.extend(sub_trajectory_ids)
        t_idx_values.extend([f"{user_id}_{sub_id}" for sub_id in sub_trajectory_ids])

    # Add new columns to the DataFrame at once
    df["sub_trajectory"] = sub_trajectories
    df["t_idx"] = pd.factorize(np.array(t_idx_values))[0]

    return df


def project_lon_lat(df):
    print("Converting to GeoPandas dataframe...")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat, crs="epsg:4326")
    )

    print("Projecting geocoordinates...")
    gdf = gdf.to_crs("epsg:3857")
    xy = gdf["geometry"].get_coordinates()
    gdf = gdf.drop(columns=["geometry", "lon", "lat"])
    xy = xy.rename(columns={"x": "lon", "y": "lat"})
    gdf = pd.concat([gdf, xy], axis=1)
    return gdf


def get_time_features(df):
    print("Assigning time labels...")
    df["weekday"] = df["datetime"].dt.day_of_week + 1
    df["is_workday"] = (df["weekday"] < 5).astype(int) + 1

    df["timestamp"] = (df["datetime"] - df["datetime"].min()).astype("int64") // 1e9
    df["hour"] = df["datetime"].dt.hour + 1
    return df


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


def resample_checkins(df, interval="1min", group_cols=["t_idx"]):
    return pd.concat(
        [
            dfg.set_index("datetime").resample(interval).first().dropna().reset_index()
            for _, dfg in df.groupby(group_cols)
        ]
    )


if __name__ == "__main__":
    # Preprocess Foursquare datasets
    BASEPATH = Path(__file__).parent.parent.parent.joinpath("data")
    max_hours = 24
    cities = ["NYC", "TKY"]
    tzones = ["US/Eastern", "Asia/Tokyo"]
    n_users = [400, 800]
    n_rows = 100

    for city, tz in zip(cities, tzones):
        print(f"Reading Foursquare {city} dataset...")
        df = pd.read_csv(
            f"../data/raw/dataset_TSMC2014_{city}.csv.zip",
            parse_dates=[-1],
            date_format="%a %b %d %H:%M:%S %z %Y",
        )
        df = df.rename(
            columns={
                "utcTimestamp": "datetime",
                "longitude": "lon",
                "latitude": "lat",
                "userId": "user",
            }
        )
        print("Converting datetimes...")
        df["datetime"] = df["datetime"].dt.tz_convert(tz)
        df = df.sort_values(by=["user", "datetime"])
        df = split_trajectories(df, max_hours)

        # Step 1: Calculate the minimum datetime per group
        min_datetime_per_group = df.groupby("t_idx")["datetime"].first()

        # Step 2: Sort the groups by their minimum datetime
        sorted_groups = min_datetime_per_group.sort_values().index

        # Step 3: Reorder the original dataframe
        df = df.set_index("t_idx").loc[sorted_groups].reset_index()
        for n_users_i in n_users:
            top_users = df.groupby("user").size().nlargest(n_users_i).index
            df_top_users = df[df["user"].isin(top_users)]
            df_top_users = get_time_features(df_top_users)
            print("Saving dataset...")
            df_top_users.to_csv(
                BASEPATH.joinpath("processed", f"foursquare_{city}_{n_users_i}.csv.gz"),
                index=False,
            )

    # Preprocess Geolife

    max_hours = 3
    n_users = [75, 150]
    limits = np.array([[116.1, 39.7], [116.7, 40.1]])

    df = read_all_users(BASEPATH.joinpath("raw", "Geolife Trajectories 1.3", "Data"))
    df[
        (df.lon >= limits[0, 0])
        & (df.lon < limits[1, 0])
        & (df.lat >= limits[0, 1])
        & (df.lat < limits[1, 1])
    ]
    df["datetime"] = df["datetime"] + datetime.timedelta(hours=8)
    df = df.sort_values(by=["user", "datetime"])
    df = split_trajectories(df, max_hours=max_hours)
    df = resample_checkins(df, interval="1min")
    for n_users_i in n_users:
        top_users = df.groupby("user").size().nlargest(n_users_i).index
        df_top_users = df[df["user"].isin(top_users)]
        df_top_users["user"] = df_top_users.groupby("user").ngroup()
        df_top_users = project_lon_lat(df_top_users)
        df_top_users = get_time_features(df_top_users)
        print("Saving dataset...")
        df_top_users.to_csv(
            BASEPATH.joinpath("processed", f"geolife_{n_users_i}.csv.gz"), index=False
        )
