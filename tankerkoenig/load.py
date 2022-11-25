"""
    Load the data from tankerkoenig.de

Raises:
    DirectoryNotFound: If the data directory is not found

"""

import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pyarrow import feather


class DirectoryNotFound(Exception):
    """Exception raised for errors in the input."""


CUR_DIR = os.path.abspath(__file__)
PROCESSED_PATH = os.path.join(Path(CUR_DIR).parent.parent.absolute(), "processed-data")
DATA_PATH = os.path.join(CUR_DIR, "data")
FUEL_TYPES = ["e5", "e10", "diesel"]

if not os.path.isdir(DATA_PATH):
    raise DirectoryNotFound(
        "Data directory not found."
        + "Please clone the git repository first."
        + "https://tankerkoenig@dev.azure.com/tankerkoenig/tankerkoenig-data/_git/tankerkoenig-data"
    )

if not os.path.isdir(PROCESSED_PATH):
    os.makedirs(PROCESSED_PATH)


def find_recent_file(category: str) -> str:
    """
    find_recent_file finds the most recent file in the data directory

    Args:
        category (str): The category of the file to find. Either "prices" or "stations"

    Returns:
        str: The path to the most recent file
    """
    year = None
    years = os.listdir(f"{DATA_PATH}/{category}")

    for elem in years:
        if elem.isnumeric():
            if year is None:
                year = int(elem)
            elif int(elem) > year:
                year = int(elem)

    month = None
    months = os.listdir(f"{DATA_PATH}/{category}/{year}")

    for elem in months:
        if elem.isnumeric():
            if month is None:
                month = int(elem)
            elif int(elem) > month:
                month = int(elem)

    day = None
    files = os.listdir(f"{DATA_PATH}/{category}/{year}/{month:02}")

    for elem in files:
        file_day = int(elem.split("-")[2])
        if day is None:
            day = file_day
        elif file_day > day:
            day = file_day

    return f"{DATA_PATH}/{category}/{year}/{month:02}/{year}-{month:02}-{day:02}-{category}.csv"


def format_date(elem: str) -> datetime.datetime:
    """
    format_date formats the date to a datetime object

    Args:
        elem (str): The date as a string

    Returns:
        datetime.datetime: The date as a datetime object
    """
    return datetime.datetime.strptime(elem[:-3], "%Y-%m-%d %H:%M:%S")


def load_prices(date: Optional[datetime.datetime] = None) -> Dict[str, str]:
    """
    load_prices loads the prices from the most recent file in the data directory or from the given date

    Args:
        date (Optional[datetime.datetime], optional): The date to load the prices for. Defaults to None.

    Raises:
        FileExistsError: If the file for the given date does not exist

    Returns:
        dict[str,str]: The paths to the feather files
    """
    new_filenames: Dict[str, str] = {}
    change_columns = {"diesel": "dieselchange", "e5": "e5change", "e10": "e10change"}

    if date is None:
        filename = find_recent_file("prices")
    else:
        filename = f"{DATA_PATH}/prices/{date.year}/{date.month:02}/{date.year}-{date.month:02}-{date.day:02}-prices.csv"

    if os.path.isfile(filename):
        data = pd.read_csv(filename)  # type: ignore
    else:
        raise FileExistsError(f"No file found for {date}")

    data["date"] = data["date"].apply(format_date)  # type: ignore

    for fuel in FUEL_TYPES:
        if date is None:
            new_filename = f"{PROCESSED_PATH}/prices/{fuel}/recent.feather"
        else:
            new_filename = (
                f"{PROCESSED_PATH}/prices/{fuel}/{date.year}-{date.month:02}-{date.day:02}.feather"
            )
        change_drop = change_columns
        change_drop.pop(fuel)
        columns_drop: List[str] = [
            col
            for col in data.columns
            if (col in FUEL_TYPES and col != fuel) or (col in change_drop.values())
        ]

        data_new = data.drop(columns_drop, axis=1)

        data_new = data_new[(data_new[fuel] >= 0.5) & (data_new[fuel] <= 3.0)]

        feather.write_feather(data_new, new_filename)  # type: ignore

        new_filenames[fuel] = new_filename

    return new_filenames


def load_stations(date: Optional[datetime.datetime] = None) -> str:
    """
    load_stations loads the stations from the most recent file in the data directory or from the given date

    Args:
        date (Optional[datetime.datetime], optional): The date to load the stations for. Defaults to None.

    Raises:
        FileExistsError: If the file for the given date does not exist

    Returns:
        str: The path to the feather file
    """
    if date is None:
        filename = find_recent_file("stations")
        new_filename = f"{PROCESSED_PATH}/stations/recent.feather"
    else:
        filename = f"{DATA_PATH}/stations/{date.year}/{date.month:02}/{date.year}-{date.month:02}-{date.day:02}-stations.csv"
        new_filename = (
            f"{PROCESSED_PATH}/stations/{date.year}-{date.month:02}-{date.day:02}.feather"
        )

    if os.path.isfile(filename):
        data = pd.read_csv(filename)  # type: ignore
    else:
        raise FileExistsError(f"No file found for {date}")

    data.drop(
        ["street", "house_number", "post_code", "city", "first_active", "openingtimes_json"],
        inplace=True,
        axis=1,
    )

    feather.write_feather(data, new_filename)  # type: ignore

    return new_filename
