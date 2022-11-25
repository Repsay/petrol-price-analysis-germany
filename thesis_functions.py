"""This file contains the functions to load the data"""
import datetime
import os
from typing import Any, Callable, Dict, List, Optional, Union

import gadm
import geopandas as gpd
import grip4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import statsmodels.api as sm
from haversine import haversine  # type: ignore
from linearmodels.panel import PanelOLS, PooledOLS
from linearmodels.panel.results import PanelResults
from shapely.geometry import Point
from tqdm import tqdm
from wp import WP

from tankerkoenig import load as tk

FUEL_TYPES = tk.FUEL_TYPES
PROCESSED_PATH = tk.PROCESSED_PATH
PRICE_CATEGORIES: Dict[str, Callable[[Union[float, int]], bool]] = {
    "negative": lambda x: x < 0,  # type: ignore
    "zero": lambda x: x == 0,  # type: ignore
    "1-2": lambda x: 1 <= x < 2,  # type: ignore
    "2-3": lambda x: 2 <= x < 3,  # type: ignore
    "3-4": lambda x: 3 <= x < 4,  # type: ignore
    "4+": lambda x: x >= 4,  # type: ignore
}

tqdm.pandas()


def load_prices_per_day(start: datetime.datetime, end: datetime.datetime) -> None:
    """
    load_prices_per_day loads the prices per day for each fuel type and stores them in a feather file

    Args:
        start (datetime.datetime): start date
        end (datetime.datetime): end date
    """
    current = start
    while current < end:
        tk.load_prices(current)
        print(f"{current.year}-{current.month:02}-{current.day:02}", end="\r")
        current += datetime.timedelta(days=1)


def load_stations_per_day(start: datetime.datetime, end: datetime.datetime) -> None:
    """
    load_stations_per_day loads the stations per day and stores them in a feather file

    Args:
        start (datetime.datetime): start date
        end (datetime.datetime): end date
    """
    current = start
    while current < end:
        tk.load_stations(current)
        print(f"{current.year}-{current.month:02}-{current.day:02}", end="\r")
        current += datetime.timedelta(days=1)


def load_unique_stations_uuid_in_range(
    start: datetime.datetime, end: datetime.datetime
) -> Dict[str, List[str]]:
    """
    load_unique_stations_uuid_in_range loads the unique stations uuids in a given range

    Args:
        start (datetime.datetime): start date
        end (datetime.datetime): end date

    Raises:
        MissingDataFile: if a file is missing

    Returns:
        Dict[str, List[str]]: dictionary with fuel type as key and list of uuids as value
    """
    stations: Dict[str, List[str]] = {key: [] for key in FUEL_TYPES}

    for fuel in FUEL_TYPES:
        current = start
        while current < end:
            filename = f"{PROCESSED_PATH}/prices/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather"
            if not os.path.isfile(filename):
                tk.load_stations(current)

            df: pd.DataFrame = feather.read_feather(filename)  # type: ignore
            stations[fuel].extend(df["station_uuid"].unique())  # type: ignore

            print(f"{current.year}-{current.month:02}-{current.day:02}", end="\r")
            current += datetime.timedelta(days=1)

        stations[fuel] = list(set(stations[fuel]))

    return stations


def load_unique_stations_in_range(
    uuids: Dict[str, List[str]], start: datetime.datetime, end: datetime.datetime
) -> Dict[str, str]:
    """
    load_unique_stations_in_range loads the unique stations in a given range

    Args:
        uuids (Dict[str, List[str]]): dictionary with fuel type as key and list of uuids as value
        start (datetime.datetime): start date
        end (datetime.datetime): end date

    Returns:
        Dict[str, str]: dictionary with fuel type as key and filename as value
    """
    new_filenames: Dict[str, str] = {}
    for fuel in FUEL_TYPES:
        new_df: Optional[pd.DataFrame] = None

        for i, uuid in enumerate(uuids[fuel]):
            print(f"{(round(i/len(uuids[fuel])*100, 2))}%", end="\r")
            df = load_station_data_in_range(uuid, start, end)
            if len(df) > 0:
                if new_df is None:
                    new_df = df
                else:
                    new_df = pd.concat([new_df, df], ignore_index=True)  # type: ignore

        new_filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}.feather"
        if not os.path.isdir(os.path.dirname(new_filename)):
            os.makedirs(os.path.dirname(new_filename))
        feather.write_feather(new_df, new_filename)  # type: ignore
        new_filenames[fuel] = new_filename

    return new_filenames


def load_station_data_in_range(
    uuid: str, start: datetime.datetime, end: datetime.datetime
) -> pd.DataFrame:
    """
    load_station_data_in_range loads the station data in a given range

    Args:
        uuid (str): station uuid
        start (datetime.datetime): start date
        end (datetime.datetime): end date

    Returns:
        pd.DataFrame: station data
    """
    current = end - datetime.timedelta(days=1)
    df: pd.DataFrame = pd.DataFrame()
    while current >= start:
        filename = (
            f"{PROCESSED_PATH}/stations/{current.year}-{current.month:02}-{current.day:02}.feather"
        )
        if not os.path.isfile(filename):
            tk.load_stations(current)

        if os.path.isfile(filename):
            df_temp: pd.DataFrame = feather.read_feather(filename)  # type: ignore
            df_temp = df_temp[df_temp["uuid"] == uuid]
            if len(df_temp) > 0:
                return df_temp
            else:
                current -= datetime.timedelta(days=1)

    return df


def mean_price_per_station_per_day(start: datetime.datetime, end: datetime.datetime) -> None:
    """
    mean_price_per_station_per_day calculates the mean price per station per day

    Args:
        start (datetime.datetime): start date
        end (datetime.datetime): end date
    """
    for fuel in FUEL_TYPES:
        current = start
        while current < end:
            filename = f"{PROCESSED_PATH}/prices/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather"

            if not os.path.isfile(filename):
                tk.load_prices(current)

            if os.path.isfile(filename):
                df: pd.DataFrame = feather.read_feather(filename)  # type: ignore
                df_mean = df.groupby(["station_uuid"])[fuel].mean()  # type: ignore
                df_mean = pd.DataFrame(df_mean, columns=[fuel])
                new_filename = f"{PROCESSED_PATH}/prices_daily_avg/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather"
                if not os.path.isdir(os.path.dirname(new_filename)):
                    os.makedirs(os.path.dirname(new_filename))
                feather.write_feather(df_mean, new_filename)  # type: ignore

            print(f"{fuel}: {current.year}-{current.month:02}-{current.day:02}", end="\r")
            current += datetime.timedelta(days=1)


def categorize_station_by_price_in_range(start: datetime.datetime, end: datetime.datetime) -> None:
    """
    categorize_station_by_price_in_range categorizes the stations by price in a given range

    Args:
        start (datetime.datetime): start date
        end (datetime.datetime): end date
    """
    stations: Dict[str, Dict[str, List[str]]] = {
        fuel: {cat: [] for cat in PRICE_CATEGORIES.keys()} for fuel in FUEL_TYPES
    }
    for fuel in FUEL_TYPES:
        current = start
        while current < end:
            filename = f"{PROCESSED_PATH}/prices/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather"

            if not os.path.isfile(filename):
                tk.load_prices(current)

            if os.path.isfile(filename):
                df: pd.DataFrame = feather.read_feather(filename)  # type: ignore

                for cat, func in PRICE_CATEGORIES.items():
                    stations[fuel][cat].extend(
                        df[df[fuel].apply(func)]["station_uuid"].unique()  # type: ignore
                    )

            print(f"{fuel}: {current.year}-{current.month:02}-{current.day:02}", end="\r")
            current += datetime.timedelta(days=1)

        for key in stations[fuel].keys():
            stations[fuel][key] = list(set(stations[fuel][key]))
            df = pd.DataFrame(stations[fuel][key], columns=["station_uuid"])
            filename = f"{PROCESSED_PATH}/stations_by_price/{fuel}/{key}.feather"
            if not os.path.isdir(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            feather.write_feather(df, filename)  # type: ignore


def count_stations_percentage_negative_zero(
    start: datetime.datetime, end: datetime.datetime
) -> Dict[str, str]:
    """
    count_stations_percentage_negative_zero counts the percentage of stations with negative and zero prices

    Returns:
        Dict[str, str]: filename of the stations with negative and zero prices per fuel type
    """
    stations: Dict[str, Dict[str, List[str]]] = {
        key: {"negative": [], "zero": []} for key in FUEL_TYPES
    }
    new_filenames: Dict[str, str] = {}

    for fuel in FUEL_TYPES:
        total_stations = []
        current = start
        while current < end:
            filename = f"{PROCESSED_PATH}/prices/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather"

            if not os.path.isfile(filename):
                tk.load_prices(current)

            if os.path.isfile(filename):
                df: pd.DataFrame = feather.read_feather(filename)  # type: ignore
                total_stations.extend(df["station_uuid"].unique())  # type: ignore
                stations[fuel]["negative"].extend(df[df[fuel] < 0]["station_uuid"].unique())  # type: ignore
                stations[fuel]["zero"].extend(df[df[fuel] == 0]["station_uuid"].unique())  # type: ignore

            print(f"{fuel}: {current.year}-{current.month:02}-{current.day:02}", end="\r")
            current += datetime.timedelta(days=1)

        negative = len(set(stations[fuel]["negative"]))
        zero = len(set(stations[fuel]["zero"]))
        total = len(set(total_stations))  # type: ignore

        new_filename = f"{PROCESSED_PATH}/stations_by_price/{fuel}/negative_zero.feather"
        new_filenames[fuel] = new_filename
        df = pd.DataFrame(stations[fuel])
        if not os.path.isdir(os.path.dirname(new_filename)):
            os.makedirs(os.path.dirname(new_filename))
        feather.write_feather(df, new_filename)  # type: ignore

        print(fuel)
        print(f"Negative: {negative}")
        print(f"Zero: {zero}")
        print(f"Total: {total}")
        print(f"Negative Percentage: {negative / total * 100}%")
        print(f"Zero Percentage: {zero / total * 100}%")

    return new_filenames


def load_population_data(required: Optional[int] = None) -> pd.DataFrame:
    """
    load_population_data loads the population data

    Args:
        required (Optional[int], optional): minimum required population. Defaults to None.
    Returns:
        DataFrame: population data
    """
    filename = f"{PROCESSED_PATH}/population/population_germany.feather"

    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    if not os.path.isfile(filename):
        df = WP().to_pandas("DEU", "ppp_2020")
        feather.write_feather(df, filename)  # type: ignore

    df: pd.DataFrame = feather.read_feather(filename)  # type: ignore

    if not required is None:
        df = df[df["population"] > required]

    return df


def create_bar_chart_for_stations_by_price() -> None:
    """
    create_bar_chart_for_stations_by_price creates a bar chart for the stations by price
    """
    stations: Dict[str, Dict[str, int]] = {
        fuel: {cat: 0 for cat in PRICE_CATEGORIES.keys()} for fuel in FUEL_TYPES
    }
    for fuel in FUEL_TYPES:
        for key in PRICE_CATEGORIES.keys():
            filename = f"{PROCESSED_PATH}/stations_by_price/{fuel}/{key}.feather"
            df: pd.DataFrame = feather.read_feather(filename)  # type: ignore
            stations[fuel][key] = len(df)
        ranges = list(stations[fuel].keys())
        values = list(stations[fuel].values())
        fig = plt.figure()  # type: ignore
        plt.bar(ranges, values, color="blue")  # type: ignore
        plt.xlabel("Price Range")  # type: ignore
        plt.ylabel("Number of Stations")  # type: ignore
        plt.title(f"Number of Stations by Price Range for {fuel.upper()}")  # type: ignore
        save_path = f"{PROCESSED_PATH}/stations_by_price/{fuel}/stations_by_price.png"
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)  # type: ignore
        plt.close(fig)  # type: ignore


def load_border_data(country: str) -> pd.DataFrame:
    """
    load_border_data loads the border data for a country

    Args:
        country (str): country code ISO3

    Returns:
        pd.DataFrame: border data
    """
    filename = f"{PROCESSED_PATH}/borders/{country}.feather"

    if not os.path.isfile(filename):
        coords = gadm.Country(country).get_coordinates()
        df = pd.DataFrame(coords, columns=["lon", "lat"])
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        feather.write_feather(df, filename)  # type: ignore

    df: pd.DataFrame = feather.read_feather(filename)  # type: ignore

    return df


def load_road_data(road_type: int = 1) -> pd.DataFrame:
    """
    load_road_data loads the road data

    Args:
        road_type (int, optional): road type not implemented at the moment. Defaults to 1.

    Returns:
        DataFrame: road data
    """
    filename = f"{PROCESSED_PATH}/road/germany-{road_type}.feather"

    if not os.path.isfile(filename):
        europe = grip4.Region("Europe")
        germany = europe.get_country("DEU")
        germany.set_highways_filter()
        coords = germany.get_coordinates()
        df = pd.DataFrame(coords, columns=["lat", "lon"])
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        feather.write_feather(df, filename)  # type: ignore

    df: pd.DataFrame = feather.read_feather(filename)  # type: ignore

    return df


def calculate_shortest_distance_to_border(
    latitude: float, longitude: float, country: str
) -> float:
    """
    calculate_shortest_distance_to_border calculates the shortest distance to the border of a country

    Args:
        latitude (float): latitude of the point
        longitude (float): longitude of the point
        country (str): country code ISO3

    Returns:
        float: shortest distance to the border
    """
    df = load_border_data(country)
    df = df[
        (df["lat"] <= latitude + 0.25)
        & (df["lat"] >= latitude - 0.25)
        & (df["lon"] <= longitude + 0.5)
        & (df["lon"] >= longitude - 0.5)
    ]
    df["distance"] = df.apply(
        lambda row: haversine((latitude, longitude), (row["lat"], row["lon"])), axis=1
    )
    return df["distance"].min()


def calculate_distances_to_border(fuel: str, country: str) -> pd.DataFrame:
    """
    calculate_distances_to_border calculates the distances to the border of a country

    Args:
        fuel (str): fuel type
        country (str): country code ISO3

    Raises:
        FileNotFoundError: if the file does not exist

    Returns:
        DataFrame: station info with distance to border
    """
    new_filename = (
        f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_distance_to_border_{country}.feather"
    )

    if not os.path.isfile(new_filename):
        filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}.feather"

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"{filename} not found")

        df: pd.DataFrame = feather.read_feather(filename)  # type: ignore

        df[f"distance_to_border_{country}"] = df.progress_apply(
            lambda row: calculate_shortest_distance_to_border(
                row["latitude"], row["longitude"], country
            ),
            axis=1,
        )

        if not os.path.isdir(os.path.dirname(new_filename)):
            os.makedirs(os.path.dirname(new_filename))

        feather.write_feather(df, new_filename)  # type: ignore
        return df

    return feather.read_feather(new_filename)  # type: ignore


def calculate_dummy_distance_to_border(fuel: str, country: str) -> pd.DataFrame:
    """
    calculate_dummy_distance_to_border calculates the dummy distance to the border of a country

    Args:
        fuel (str): fuel type
        country (str):  country code ISO3

    Returns:
        DataFrame: station info with dummy distance to border
    """
    stations_filename = (
        f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_distance_to_border_DEU.feather"
    )
    new_filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_distance_to_border_dummy_{country}.feather"

    if not os.path.isfile(new_filename):
        df: pd.DataFrame = feather.read_feather(stations_filename)  # type: ignore

        df = df[df["distance_to_border_DEU"] < 20]  # type: ignore

        df[f"distance_to_border_{country}"] = df.progress_apply(
            lambda row: calculate_shortest_distance_to_border(
                row["latitude"], row["longitude"], country
            ),
            axis=1,
        )

        df[f"distance_to_border_{country}_dummy"] = np.where(
            df[f"distance_to_border_{country}"] < 20, 1, 0
        )

        if not os.path.isdir(os.path.dirname(new_filename)):
            os.makedirs(os.path.dirname(new_filename))

        feather.write_feather(df, new_filename)  # type: ignore

        return df

    return feather.read_feather(new_filename)  # type: ignore


def calculate_shortest_distance_to_highway(latitude: float, longitude: float) -> float:
    """
    calculate_shortest_distance_to_highway calculates the shortest distance to the highway

    Args:
        latitude (float): latitude of the point
        longitude (float): longitude of the point

    Returns:
        float: shortest distance to the highway
    """
    df = load_road_data()
    df = df[
        (df["lat"] <= latitude + 0.5)
        & (df["lat"] >= latitude - 0.5)
        & (df["lon"] <= longitude + 1)
        & (df["lon"] >= longitude - 1)
    ]
    df["distance"] = df.apply(
        lambda row: haversine((latitude, longitude), (row["lat"], row["lon"])), axis=1
    )
    return df["distance"].min()


def calculate_distances_to_highway(fuel: str) -> pd.DataFrame:
    """
    calculate_distances_to_highway calculates the distances to the highway

    Args:
        fuel (str): fuel type

    Returns:
        DataFrame: station info with distance to highway
    """
    filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}.feather"
    df: pd.DataFrame = feather.read_feather(filename)  # type: ignore

    df["distance_to_highway"] = df.progress_apply(
        lambda row: calculate_shortest_distance_to_highway(row["latitude"], row["longitude"]),
        axis=1,
    )

    filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_distance_to_highway.feather"
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    feather.write_feather(df, filename)  # type: ignore

    return df


def calculate_competition_for_station(fuel: str, latitude: float, longitude: float) -> Any:
    """
    calculate_competition_for_station calculates the competition for a station

    Args:
        fuel (str): fuel type
        latitude (float): latitude of the station
        longitude (float): longitude of the station

    Raises:
        FileNotFoundError: if the file does not exist

    Returns:
        Series[int]: competition for the station
    """
    filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}.feather"
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} not found")
    df: pd.DataFrame = feather.read_feather(filename)  # type: ignore

    df = df[
        (df["latitude"] <= latitude + 0.1)
        & (df["latitude"] >= latitude - 0.1)
        & (df["longitude"] <= longitude + 0.2)
        & (df["longitude"] >= longitude - 0.2)
    ]

    df["distance"] = df.apply(
        lambda row: haversine((latitude, longitude), (row["latitude"], row["longitude"])), axis=1
    )

    count_05 = len(df[df["distance"] < 0.5]) - 1
    count_1 = len(df[df["distance"] < 1]) - 1
    count_2 = len(df[df["distance"] < 2]) - 1
    count_3 = len(df[df["distance"] < 3]) - 1
    count_5 = len(df[df["distance"] < 5]) - 1

    return pd.Series([count_05, count_1, count_2, count_3, count_5])  # type: ignore


def calculate_competition(fuel: str) -> pd.DataFrame:
    """
    calculate_competition calculates the competition for each station

    Args:
        fuel (str): fuel type

    Raises:
        FileNotFoundError: if the file is not found

    Returns:
        DataFrame: station info with competition
    """
    filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}.feather"
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} not found")
    df: pd.DataFrame = feather.read_feather(filename)  # type: ignore

    df[
        ["competition_05", "competition_1", "competition_2", "competition_3", "competition_5"]
    ] = df.progress_apply(
        lambda row: calculate_competition_for_station(fuel, row["latitude"], row["longitude"]),
        axis=1,
    )

    filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_competition.feather"
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    feather.write_feather(df, filename)  # type: ignore

    return df


def calculate_population_for_station(latitude: float, longitude: float) -> Any:
    """
    calculate_population_for_station calculates the population for a station

    Args:
        latitude (float): latitude of the station
        longitude (float): longitude of the station

    Returns:
        Series[float]: population for the station
    """
    df = load_population_data(0)

    df = df[
        (df["lat"] <= latitude + 0.1)
        & (df["lat"] >= latitude - 0.1)
        & (df["lon"] <= longitude + 0.2)
        & (df["lon"] >= longitude - 0.2)
    ]

    df["distance"] = df.apply(
        lambda row: haversine((latitude, longitude), (row["lat"], row["lon"])), axis=1
    )

    df_1 = df[df["distance"] < 1]
    df_3 = df[df["distance"] < 3]
    df_5 = df[df["distance"] < 5]

    return pd.Series(
        [df_1["population"].sum(), df_3["population"].sum(), df_5["population"].sum()]
    )


def calculate_population(fuel: str) -> pd.DataFrame:
    """
    calculate_population calculates the population for each station

    Args:
        fuel (str): fuel type

    Raises:
        FileNotFoundError: if the file is not found

    Returns:
        DataFrame: station info with population
    """
    filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}.feather"
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} not found")
    df: pd.DataFrame = feather.read_feather(filename)  # type: ignore

    df[["population_1", "population_3", "population_5"]] = df.progress_apply(
        lambda row: calculate_population_for_station(row["latitude"], row["longitude"]), axis=1
    )

    filename = f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_populations.feather"
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    feather.write_feather(df, filename)  # type: ignore

    return df


def generate_station_data(fuel: str) -> pd.DataFrame:
    """
    generate_station_data generates the station data

    Args:
        fuel (str): fuel type

    Returns:
        DataFrame: station data
    """
    df_unique_stations: pd.DataFrame = feather.read_feather(
        f"{PROCESSED_PATH}/stations/all_unique_{fuel}.feather"
    )
    df_unique_stations.set_index("uuid", inplace=True)

    df_distance_border: pd.DataFrame = feather.read_feather(
        f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_distance_to_border.feather"
    )
    df_distance_border.set_index("uuid", inplace=True)
    df_distance_border.drop(columns=df_unique_stations.columns, inplace=True)

    df_distance_highway: pd.DataFrame = feather.read_feather(
        f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_distance_to_highway.feather"
    )
    df_distance_highway.set_index("uuid", inplace=True)
    df_distance_highway.drop(columns=df_unique_stations.columns, inplace=True)

    df_population: pd.DataFrame = feather.read_feather(
        f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_populations.feather"
    )
    df_population.set_index("uuid", inplace=True)
    df_population.drop(columns=df_unique_stations.columns, inplace=True)

    df_competition: pd.DataFrame = feather.read_feather(
        f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_competition.feather"
    )
    df_competition.set_index("uuid", inplace=True)
    df_competition.drop(columns=df_unique_stations.columns, inplace=True)

    # Merge all dataframes
    df = pd.concat(
        [
            df_unique_stations,
            df_distance_border,
            df_distance_highway,
            df_population,
            df_competition,
        ],
        axis=1,
    )
    df = df[df["population_5"] != 0]
    feather.write_feather(
        df,
        f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_all_data.feather",
    )

    return df


def generate_avg_prices_per_day_per_station(
    fuel: str, start_date: datetime.datetime, end_date: datetime.datetime
) -> pd.DataFrame:
    """
    generate_avg_prices_per_day_per_station generates the average prices per day per station as dataframe

    Args:
        fuel (str): fuel type
        start_date (datetime.datetime): start date
        end_date (datetime.datetime): end date

    Returns:
        DataFrame: average prices per day per station
    """
    df: pd.DataFrame = feather.read_feather(
        f"{PROCESSED_PATH}/stations/all_unique_{fuel}_with_all_data.feather"
    )

    stations = df.index.unique()
    new_df = pd.DataFrame(columns=stations)
    stations_list = stations.to_list()
    new_df = new_df.T
    current = start_date

    while current < end_date:
        df_avg: pd.DataFrame = feather.read_feather(
            f"{PROCESSED_PATH}/prices_daily_avg/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather"
        )
        df_avg = df_avg.rename(columns={"e5": current.strftime("%Y-%m-%d")})
        new_df = pd.concat([new_df, df_avg], axis=1)
        current = current + datetime.timedelta(days=1)

    new_df: pd.DataFrame = new_df.T
    new_df.fillna(method="ffill", inplace=True)
    new_df["date"] = new_df.index
    new_df = pd.melt(
        new_df, id_vars=["date"], value_vars=stations_list, var_name="uuid", value_name="price"
    )

    new_df = new_df[["uuid", "date", "price"]]

    feather.write_feather(
        new_df,
        f"{PROCESSED_PATH}/stations/avg_prices_{fuel}_{start_date.year}-{start_date.month:02}-{start_date.day:02}_{end_date.year}-{end_date.month:02}-{end_date.day:02}.feather",
    )

    return new_df


def generate_avg_prices_for_range_per_station(
    fuel: str, start_date: datetime.datetime, end_date: datetime.datetime
) -> pd.DataFrame:
    """
    generate_avg_prices_for_range_per_station generates the average prices for a range per station

    Args:
        fuel (str): fuel type
        start_date (datetime.datetime): start date
        end_date (datetime.datetime): end date

    Returns:
        DataFrame: average prices for a range per station
    """
    df: pd.DataFrame = feather.read_feather(
        f"{PROCESSED_PATH}/stations/avg_prices_{fuel}_{start_date.year}-{start_date.month:02}-{start_date.day:02}_{end_date.year}-{end_date.month:02}-{end_date.day:02}.feather"
    )

    df = df.groupby(["uuid"])["price"].mean().reset_index()

    feather.write_feather(
        df,
        f"{PROCESSED_PATH}/stations/avg_prices_{fuel}_range_{start_date.year}-{start_date.month:02}-{start_date.day:02}_{end_date.year}-{end_date.month:02}-{end_date.day:02}.feather",
    )

    return df


def generate_shapefile(df: pd.DataFrame):
    df["geometry"] = df.apply(lambda x: Point((float(x["longitude"]), float(x["latitude"]))), axis=1)  # type: ignore
    gdf = gpd.GeoDataFrame(df, geometry="geometry")  # type: ignore
    gdf.to_file(f"{PROCESSED_PATH}/Stations.shp", driver="ESRI Shapefile")  # type: ignore


def pooled_ols_regression(panel_data: pd.DataFrame) -> PanelResults:
    """
    pooled_ols_regression performs a pooled OLS regression

    Args:
        panel_data (pd.DataFrame): panel data

    Returns:
        PanelResults: results of the regression
    """
    exog_var = [
        "tax",
        "distance_to_highway",
        "distance_to_border",
        "population_1",
        "population_3",
        "population_5",
        "competition_1",
        "competition_3",
        "competition_5",
        "day",
        "day_post",
    ]
    exog = sm.add_constant(panel_data[exog_var])
    mod = PooledOLS(panel_data["price"], exog)
    pooled_res = mod.fit()
    return pooled_res
    # cov_type='clustered', cluster_entity=True


def pooled_ols_regression_with_interaction(panel_data: pd.DataFrame) -> PanelResults:
    """
    pooled_ols_regression_with_interaction performs a pooled OLS regression with interaction

    Args:
        panel_data (pd.DataFrame): panel data

    Returns:
        PanelResults: results of the regression
    """
    exog_var = [
        "tax",
        "distance_to_highway",
        "distance_to_border",
        "population_1",
        "population_3",
        "population_5",
        "competition_1",
        "competition_3",
        "competition_5",
        "tax_distance_to_highway",
        "tax_distance_to_border",
        "tax_population_1",
        "tax_population_3",
        "tax_population_5",
        "tax_competition_1",
        "tax_competition_3",
        "tax_competition_5",
        "day",
        "day_post",
    ]
    exog = sm.add_constant(panel_data[exog_var])
    mod = PooledOLS(panel_data["price"], exog)
    pooled_res = mod.fit()
    return pooled_res


def pooled_ols_regression_fixed_effects(panel_data: pd.DataFrame) -> PanelResults:
    """
    pooled_ols_regression_fixed_effects performs a pooled OLS regression on fixed effects

    Args:
        panel_data (pd.DataFrame): panel data

    Returns:
        PanelResults: results of the regression
    """
    exog_var = ["tax", "day", "day_post"]
    exog = sm.add_constant(panel_data[exog_var])
    mod = PanelOLS(panel_data["price"], exog, entity_effects=True)
    fe_res = mod.fit()
    return fe_res


def pooled_ols_regression_fixed_effects_with_interaction(panel_data: pd.DataFrame) -> PanelResults:
    """
    pooled_ols_regression_fixed_effects_with_interaction performs a pooled OLS regression on fixed effects with interaction

    Args:
        panel_data (pd.DataFrame): panel data

    Returns:
        PanelResults: results of the regression
    """
    exog_var = [
        "tax",
        "tax_distance_to_highway",
        "tax_distance_to_border",
        "tax_population_1",
        "tax_population_3",
        "tax_population_5",
        "tax_competition_1",
        "tax_competition_3",
        "tax_competition_5",
        "day",
        "day_post",
    ]
    exog = sm.add_constant(panel_data[exog_var])
    mod = PanelOLS(panel_data["price"], exog, entity_effects=True)
    fe_res = mod.fit()
    return fe_res
