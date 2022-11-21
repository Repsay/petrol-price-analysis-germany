import os
import pandas as pd
import pyarrow.feather as feather
import datetime
from typing import Dict, Optional

def find_recent_file(category: str):
    year = None
    years = os.listdir(f'D:/Thesis/petrol-price-analysis-germany/tankerkoenig/data/{category}')

    for elem in years:
        if elem.isnumeric():
            if year is None:
                year = int(elem)
            elif int(elem) > year:
                year = int(elem)

    month = None
    months = os.listdir(f'D:/Thesis/petrol-price-analysis-germany/tankerkoenig/data/{category}/{year}')

    for elem in months:
        if elem.isnumeric():
            if month is None:
                month = int(elem)
            elif int(elem) > month:
                month = int(elem)


    day = None
    files = os.listdir(f'D:/Thesis/petrol-price-analysis-germany/tankerkoenig/data/{category}/{year}/{month:02}')

    for elem in files:
        file_day = int(elem.split('-')[2])
        if day is None:
            day = file_day
        elif file_day > day:
            day = file_day

    return f'D:/Thesis/petrol-price-analysis-germany/tankerkoenig/data/{category}/{year}/{month:02}/{year}-{month:02}-{day:02}-{category}.csv'

def format_date(elem: str):
    return datetime.datetime.strptime(elem[:-3], '%Y-%m-%d %H:%M:%S')

def load_prices(date: Optional[datetime.datetime] = None):
    new_fns: Dict[str, str] = {}
    change_columns = {'diesel': 'dieselchange', 'e5': 'e5change', 'e10': 'e10change'}
    if date is None:
        fn = find_recent_file('prices')
        data = pd.read_csv(fn)
    else:
        fn = f'D:/Thesis/petrol-price-analysis-germany/tankerkoenig/data/prices/{date.year}/{date.month:02}/{date.year}-{date.month:02}-{date.day:02}-prices.csv'
        if os.path.isfile(fn):
            data = pd.read_csv(fn)
        else:
            raise Exception(f"No file found for {date}")

    data['date'] = data['date'].apply(format_date)

    for fuel in ['e5', 'e10', 'diesel']:
        if date is None:
            new_fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/prices/{fuel}/recent.feather'
        else:
            new_fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/prices/{fuel}/{date.year}-{date.month:02}-{date.day:02}.feather'

        columns_drop = ['e5', 'e10', 'diesel']
        columns_drop.remove(fuel)
        columns_drop.extend(change_columns.values())
        columns_drop.remove(change_columns[fuel])
        data_new = data.drop(columns_drop, axis=1)

        data_new = data_new[(data_new[fuel] >= 0.5) & (data_new[fuel] <= 3.0)]

        feather.write_feather(data_new, new_fn)

        new_fns[fuel] = new_fn

    return new_fns

def load_stations(date: Optional[datetime.datetime] = None):
    if date is None:
        fn = find_recent_file('stations')
        data = pd.read_csv(fn)
        new_fn = 'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/recent.feather'
    else:
        fn = f'D:/Thesis/petrol-price-analysis-germany/tankerkoenig/data/stations/{date.year}/{date.month:02}/{date.year}-{date.month:02}-{date.day:02}-stations.csv'
        if os.path.isfile(fn):
            data = pd.read_csv(fn)
            new_fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/{date.year}-{date.month:02}-{date.day:02}.feather'
        else:
            raise Exception(f"No file found for {date}")

    data.drop(['street', 'house_number', 'post_code', 'city', 'first_active', 'openingtimes_json'], inplace=True, axis=1)

    feather.write_feather(data, new_fn)

    return new_fn
