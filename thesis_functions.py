from typing import Dict, List, Optional
import typing
import tankerkoenig.load as tk
import population.load as pop
import border.load as border
import road.load as road
import datetime
import os
import pyarrow.feather as feather
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from haversine import haversine
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
from linearmodels.panel import PooledOLS, PanelOLS
import numpy as np

tqdm.pandas()

def load_prices_per_day(start: datetime.datetime, end: datetime.datetime):
    current = start
    while current < end:
        tk.load_prices(current)
        print(f'{current.year}-{current.month:02}-{current.day:02}', end='\r')
        current += datetime.timedelta(days=1)

def load_stations_per_day(start: datetime.datetime, end: datetime.datetime):
    current = start
    while current < end:
        tk.load_stations(current)
        print(f'{current.year}-{current.month:02}-{current.day:02}', end='\r')
        current += datetime.timedelta(days=1)

def load_unique_stations_uuid_in_range(start: datetime.datetime, end: datetime.datetime):
    stations: Dict[str, List[str]] = {'diesel': [], 'e5': [], 'e10': []}
    fuels = ['diesel', 'e5', 'e10']
    for fuel in fuels:
        current = start
        while current < end:
            fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/prices/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather'
            df: pd.DataFrame = feather.read_feather(fn)
            stations[fuel].extend(df['station_uuid'].unique())
            print(f'{current.year}-{current.month:02}-{current.day:02}', end='\r')
            current += datetime.timedelta(days=1)

        stations[fuel] = list(set(stations[fuel]))

    return stations

def load_unique_stations_in_range(uuids: Dict[str, List[str]], start: datetime.datetime, end: datetime.datetime):
    fuels = ['diesel', 'e5', 'e10']
    for fuel in fuels:
        new_df: Optional[pd.DataFrame] = None
        for i, uuid in enumerate(uuids[fuel]):
            print(f'{(round(i/len(uuids[fuel])*100, 2))}%', end='\r')
            df = load_station_data_in_range(uuid, start, end)
            if len(df) > 0:
                if new_df is None:
                    new_df = df
                else:
                    new_df = pd.concat([new_df, df], ignore_index=True)

        new_fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}.feather'
        feather.write_feather(new_df, new_fn)

def load_station_data_in_range(uuid: str, start: datetime.datetime, end: datetime.datetime):
    current = end - datetime.timedelta(days=1)
    df: pd.DataFrame = pd.DataFrame()
    while current >= start:
        fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/{current.year}-{current.month:02}-{current.day:02}.feather'
        if os.path.isfile(fn):
            df_temp: pd.DataFrame = feather.read_feather(fn)
            df_temp = df_temp[df_temp['uuid'] == uuid]
            if(len(df_temp) > 0):
                return df_temp
            else:
                current -= datetime.timedelta(days=1)
        else:
            raise Exception(f"No file found for {current}")

    return df

def mean_price_per_station_per_day(start: datetime.datetime, end: datetime.datetime):
    for fuel in ['e5', 'e10', 'diesel']:
        current = start
        while current < end:
            fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/prices/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather'
            if os.path.isfile(fn):
                df: pd.DataFrame = feather.read_feather(fn)
                df_mean = df.groupby(['station_uuid'])[fuel].mean()
                df_mean = pd.DataFrame(df_mean, columns=[fuel])
                new_fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/prices_daily_avg/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather'
                feather.write_feather(df_mean, new_fn)
            else:
                raise Exception(f"No file found for {current}")

            print(f'{fuel}: {current.year}-{current.month:02}-{current.day:02}', end='\r')
            current += datetime.timedelta(days=1)

def categorize_station_by_price_in_range(start: datetime.datetime, end: datetime.datetime):
    stations = {'diesel': {'-0': [], '0': [], '0-1': [], '1-2': [], '2-3': [], '3-4': [], '4+': []}, 'e5': {'-0': [], '0': [], '0-1': [], '1-2': [], '2-3': [], '3-4': [], '4+': []}, 'e10': {'-0': [], '0': [], '0-1': [], '1-2': [], '2-3': [], '3-4': [], '4+': []}}
    for fuel in [ 'e5', 'e10', 'diesel']:
        current = start
        while current < end:
            fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/prices/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather'
            if os.path.isfile(fn):
                df: pd.DataFrame = feather.read_feather(fn)
                stations[fuel]['-0'].extend(df[df[fuel] < 0]['station_uuid'].unique())
                stations[fuel]['0'].extend(df[df[fuel] == 0]['station_uuid'].unique())
                stations[fuel]['0-1'].extend(df[(df[fuel] > 0) & (df[fuel] < 1)]['station_uuid'].unique())
                stations[fuel]['1-2'].extend(df[(df[fuel] >= 1) & (df[fuel] < 2)]['station_uuid'].unique())
                stations[fuel]['2-3'].extend(df[(df[fuel] >= 2) & (df[fuel] < 3)]['station_uuid'].unique())
                stations[fuel]['3-4'].extend(df[(df[fuel] >= 3) & (df[fuel] < 4)]['station_uuid'].unique())
                stations[fuel]['4+'].extend(df[df[fuel] >= 4]['station_uuid'].unique())
            else:
                raise Exception(f"No file found for {current}")

            print(f'{fuel}: {current.year}-{current.month:02}-{current.day:02}', end='\r')
            current += datetime.timedelta(days=1)

        for key in stations[fuel].keys():
            stations[fuel][key] = list(set(stations[fuel][key]))
            df = pd.DataFrame(stations[fuel][key], columns=['station_uuid'])
            fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations_by_price/{fuel}/{key}.feather'
            feather.write_feather(df, fn)

def count_stations_percentage_negative_zero(start: datetime.datetime, end: datetime.datetime):
    stations = {'diesel': {'negative': [], 'zero': []}, 'e5': {'negative': [], 'zero': []}, 'e10': {'negative': [], 'zero': []}}
    for fuel in ['diesel', 'e5', 'e10']:
        total_stations = []
        current = start
        while current < end:
            fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/prices/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather'
            if os.path.isfile(fn):
                df: pd.DataFrame = feather.read_feather(fn)
                total_stations.extend(df['station_uuid'].unique())
                stations[fuel]['negative'].extend(df[df[fuel] < 0]['station_uuid'].unique())
                stations[fuel]['zero'].extend(df[df[fuel] == 0]['station_uuid'].unique())
            else:
                raise Exception(f"No file found for {current}")

            print(f'{fuel}: {current.year}-{current.month:02}-{current.day:02}', end='\r')
            current += datetime.timedelta(days=1)


        negative = len(set(stations[fuel]['negative']))
        zero = len(set(stations[fuel]['zero']))
        total = len(set(total_stations))
        print(fuel)
        print(f'Negative: {negative}')
        print(f'Zero: {zero}')
        print(f'Total: {total}')
        print(f'Negative Percentage: {negative / total * 100}%')
        print(f'Zero Percentage: {zero / total * 100}%')

def load_population_data(min: Optional[int] = None):
    fn = 'D:/Thesis/petrol-price-analysis-germany/processed-data/population/population_germany.feather'

    if not os.path.isfile(fn):
        pop.prepare_data()

    df: pd.DataFrame = feather.read_feather(fn)

    if not min is None:
        df = df[df['population'] > min]

    return df

def create_bar_chart_for_stations_by_price():
    stations = {'diesel': {'-0': 0, '0': 0, '0-1': 0, '1-2': 0, '2-3': 0, '3-4': 0, '4+': 0}, 'e5': {'-0': 0, '0': 0, '0-1': 0, '1-2': 0, '2-3': 0, '3-4': 0, '4+': 0}, 'e10': {'-0': 0, '0': 0, '0-1': 0, '1-2': 0, '2-3': 0, '3-4': 0, '4+': 0}}
    for fuel in ['diesel', 'e5', 'e10']:
        for key in stations[fuel].keys():
            fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations_by_price/{fuel}/{key}.feather'
            df: pd.DataFrame = feather.read_feather(fn)
            stations[fuel][key] = len(df)
        ranges =list(stations[fuel].keys())
        values = list(stations[fuel].values())
        fig = plt.figure()
        plt.bar(ranges, values, color='blue')
        plt.xlabel("Price Range")
        plt.ylabel("Number of Stations")
        plt.title(f"Number of Stations by Price Range for {fuel.upper()}")
        plt.savefig(f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations_by_price/{fuel}.png')
        plt.close(fig)

@typing.overload
def load_border_data() -> None:
    ...

@typing.overload
def load_border_data(country: str) -> pd.DataFrame:
    ...

def load_border_data(country: Optional[str] = None):
    if country is None:
        border.prepare_data()
        return None

    fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/borders/{country}.feather'

    border.prepare_data([country])

    df: pd.DataFrame = feather.read_feather(fn)

    return df

def load_road_data(t = 1):
    fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/road/germany-{t}.feather'

    if not os.path.isfile(fn):
        road.prepare_data(t)

    df: pd.DataFrame = feather.read_feather(fn)

    return df

def calculate_shortest_distance_to_border(latitude: float, longitude: float, country: str) -> float:
    df = load_border_data(country)
    df = df[(df['lat'] <= latitude+0.25) & (df['lat'] >= latitude-0.25) & (df['lon'] <= longitude+0.5) & (df['lon'] >= longitude-0.5)]
    df['distance'] = df.apply(lambda row: haversine((latitude, longitude), (row['lat'], row['lon'])), axis=1)
    return df['distance'].min()

def calculate_distances_to_border(fuel: str, country: str):
    fn_new = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_distance_to_border_{country}.feather'

    if not os.path.isfile(fn_new):
        fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}.feather'
        df: pd.DataFrame = feather.read_feather(fn)

        df[f'distance_to_border_{country}'] = df.progress_apply(lambda row: calculate_shortest_distance_to_border(row['latitude'], row['longitude'], country), axis=1)

        feather.write_feather(df, fn_new)
        return df

    return feather.read_feather(fn_new)

def calculate_dummy_distance_to_border(fuel: str, country: str):
    fn_stations = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_distance_to_border_DEU.feather'
    fn_new = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_distance_to_border_dummy_{country}.feather'

    if not os.path.isfile(fn_new):
        df: pd.DataFrame = feather.read_feather(fn_stations)

        df = df[df['distance_to_border_DEU'] < 20]

        df[f'distance_to_border_{country}'] = df.progress_apply(lambda row: calculate_shortest_distance_to_border(row['latitude'], row['longitude'], country), axis=1)

        df[f'distance_to_border_{country}_dummy'] = np.where(df[f'distance_to_border_{country}'] < 20, 1, 0)

        feather.write_feather(df, fn_new)

        return df

    return feather.read_feather(fn_new)


def calculate_shortest_distance_to_highway(latitude: float, longitude: float) -> float:
    df = load_road_data()
    df = df[(df['lat'] <= latitude+0.5) & (df['lat'] >= latitude-0.5) & (df['lon'] <= longitude+1) & (df['lon'] >= longitude-1)]
    df['distance'] = df.apply(lambda row: haversine((latitude, longitude), (row['lat'], row['lon'])), axis=1)
    return df['distance'].min()

def calculate_distances_to_highway(fuel: str):
    fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}.feather'
    df: pd.DataFrame = feather.read_feather(fn)

    df['distance_to_highway'] = df.progress_apply(lambda row: calculate_shortest_distance_to_highway(row['latitude'], row['longitude']), axis=1)

    fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_distance_to_highway.feather'
    feather.write_feather(df, fn)

    return df

def calculate_competition_for_station(fuel: str, latitude: float, longitude:float):
    fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}.feather'
    df: pd.DataFrame = feather.read_feather(fn)

    df = df[(df['latitude'] <= latitude+0.1) & (df['latitude'] >= latitude-0.1) & (df['longitude'] <= longitude+0.2) & (df['longitude'] >= longitude-0.2)]

    df['distance'] = df.apply(lambda row: haversine((latitude, longitude), (row['latitude'], row['longitude'])), axis=1)

    count_05 = len(df[df['distance'] < 0.5]) - 1
    count_1 = len(df[df['distance'] < 1]) - 1
    count_2 = len(df[df['distance'] < 2]) - 1
    count_3 = len(df[df['distance'] < 3]) - 1
    count_5 = len(df[df['distance'] < 5]) - 1

    return pd.Series([count_05, count_1, count_2, count_3, count_5])

def calculate_competition(fuel: str):
    fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}.feather'
    df: pd.DataFrame = feather.read_feather(fn)

    df[['competition_05', 'competition_1', 'competition_2' , 'competition_3', 'competition_5']] = df.progress_apply(lambda row: calculate_competition_for_station(fuel, row['latitude'], row['longitude']), axis=1)

    fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_competition.feather'
    feather.write_feather(df, fn)

    return df

def calculate_population_for_station(latitude: float, longitude:float):
    df = load_population_data(0)

    df = df[(df['lat'] <= latitude+0.1) & (df['lat'] >= latitude-0.1) & (df['lon'] <= longitude+0.2) & (df['lon'] >= longitude-0.2)]

    df['distance'] = df.apply(lambda row: haversine((latitude, longitude), (row['lat'], row['lon'])), axis=1)

    df_1 = df[df['distance'] < 1]
    df_3 = df[df['distance'] < 3]
    df_5 = df[df['distance'] < 5]

    return pd.Series([df_1['population'].sum(), df_3['population'].sum(), df_5['population'].sum()])

def calculate_population(fuel: str):
    fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}.feather'
    df: pd.DataFrame = feather.read_feather(fn)

    df[['population_1', 'population_3', 'population_5']] = df.progress_apply(lambda row: calculate_population_for_station(row['latitude'], row['longitude']), axis=1)

    fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_populations.feather'
    feather.write_feather(df, fn)

    return df

def generate_station_data(fuel: str):
    df_unique_stations: pd.DataFrame = feather.read_feather(f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}.feather')
    df_unique_stations.set_index('uuid', inplace=True)

    df_distance_border: pd.DataFrame = feather.read_feather(f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_distance_to_border.feather')
    df_distance_border.set_index('uuid', inplace=True)
    df_distance_border.drop(columns=df_unique_stations.columns, inplace=True)

    df_distance_highway: pd.DataFrame = feather.read_feather(f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_distance_to_highway.feather')
    df_distance_highway.set_index('uuid', inplace=True)
    df_distance_highway.drop(columns=df_unique_stations.columns, inplace=True)

    df_population: pd.DataFrame = feather.read_feather(f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_populations.feather')
    df_population.set_index('uuid', inplace=True)
    df_population.drop(columns=df_unique_stations.columns, inplace=True)

    df_competition: pd.DataFrame = feather.read_feather(f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_competition.feather')
    df_competition.set_index('uuid', inplace=True)
    df_competition.drop(columns=df_unique_stations.columns, inplace=True)

    # Merge all dataframes
    df = pd.concat([df_unique_stations, df_distance_border, df_distance_highway, df_population, df_competition], axis=1)
    df = df[df['population_5'] != 0]
    feather.write_feather(df, f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_all_data.feather')

    return df

def generate_avg_prices_per_day_per_station(fuel: str, start_date: datetime.datetime, end_date: datetime.datetime):
    df: pd.DataFrame = feather.read_feather(f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/all_unique_{fuel}_with_all_data.feather')

    stations = df.index.unique()
    new_df = pd.DataFrame(columns=stations)
    stations_list = stations.to_list()
    new_df = new_df.T
    current = start_date

    while current < end_date:
        df_avg: pd.DataFrame = feather.read_feather(f'D:/Thesis/petrol-price-analysis-germany/processed-data/prices_daily_avg/{fuel}/{current.year}-{current.month:02}-{current.day:02}.feather')
        df_avg = df_avg.rename(columns={'e5': current.strftime('%Y-%m-%d')})
        new_df = pd.concat([new_df, df_avg], axis=1)
        current = current + datetime.timedelta(days=1)

    new_df:pd.DataFrame = new_df.T
    new_df.fillna(method='ffill', inplace=True)
    new_df['date'] = new_df.index
    new_df = pd.melt(new_df, id_vars=['date'], value_vars=stations_list, var_name='uuid', value_name='price')

    new_df = new_df[['uuid', 'date', 'price']]

    feather.write_feather(new_df, f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/avg_prices_{fuel}_{start_date.year}-{start_date.month:02}-{start_date.day:02}_{end_date.year}-{end_date.month:02}-{end_date.day:02}.feather')

    return new_df

def generate_avg_prices_for_range_per_station(fuel: str, start_date: datetime.datetime, end_date: datetime.datetime):
    df: pd.DataFrame = feather.read_feather(f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/avg_prices_{fuel}_{start_date.year}-{start_date.month:02}-{start_date.day:02}_{end_date.year}-{end_date.month:02}-{end_date.day:02}.feather')

    df = df.groupby(['uuid'])['price'].mean().reset_index()

    feather.write_feather(df, f'D:/Thesis/petrol-price-analysis-germany/processed-data/stations/avg_prices_{fuel}_range_{start_date.year}-{start_date.month:02}-{start_date.day:02}_{end_date.year}-{end_date.month:02}-{end_date.day:02}.feather')

    return df
def generate_shapefile(df: pd.DataFrame):
    df['geometry'] = df.apply(lambda x: Point((float(x['longitude']), float(x['latitude']))), axis=1) # type: ignore
    gdf = gpd.GeoDataFrame(df, geometry='geometry') # type: ignore
    gdf.to_file('D:/Thesis/petrol-price-analysis-germany/Stations.shp', driver='ESRI Shapefile')

def pooled_ols_regression(panel_data: pd.DataFrame):
    exog_var = ['tax', 'distance_to_highway', 'distance_to_border', 'population_1', 'population_3', 'population_5', 'competition_1', 'competition_3', 'competition_5', 'day', 'day_post' ]
    exog = sm.add_constant(panel_data[exog_var])
    mod = PooledOLS(panel_data['price'], exog)
    pooled_res = mod.fit()
    return pooled_res
    # cov_type='clustered', cluster_entity=True

def pooled_ols_regression_with_interaction(panel_data: pd.DataFrame):
    exog_var = ['tax', 'distance_to_highway', 'distance_to_border', 'population_1', 'population_3', 'population_5', 'competition_1', 'competition_3', 'competition_5', 'tax_distance_to_highway', 'tax_distance_to_border', 'tax_population_1', 'tax_population_3', 'tax_population_5', 'tax_competition_1', 'tax_competition_3', 'tax_competition_5', 'day', 'day_post']
    exog = sm.add_constant(panel_data[exog_var])
    mod = PooledOLS(panel_data['price'], exog)
    pooled_res = mod.fit()
    return pooled_res

def fixed_effects(panel_data: pd.DataFrame):
    exog_var = ['tax', 'day', 'day_post']
    exog = sm.add_constant(panel_data[exog_var])
    mod = PanelOLS(panel_data['price'], exog, entity_effects=True)
    fe_res = mod.fit()
    return fe_res

def fixed_effects_with_interaction(panel_data: pd.DataFrame):
    exog_var = ['tax', 'tax_distance_to_highway', 'tax_distance_to_border', 'tax_population_1', 'tax_population_3', 'tax_population_5', 'tax_competition_1', 'tax_competition_3', 'tax_competition_5', 'day', 'day_post']
    exog = sm.add_constant(panel_data[exog_var])
    mod = PanelOLS(panel_data['price'], exog, entity_effects=True)
    fe_res = mod.fit()
    return fe_res