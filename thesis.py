import datetime
from typing import List

import numpy as np
import pandas as pd
import pyarrow.feather as feather

import thesis_functions as thesis

PROCESSED_PATH = thesis.PROCESSED_PATH
START_DATE = datetime.datetime(2022, 4, 1)
END_DATE = datetime.datetime(2022, 8, 1)
COUNTRIES = ["AUT", "BEL", "CHE", "CZE", "DEU", "DNK", "FRA", "LUX", "NLD", "POL"]

# Format csv to feather file
thesis.load_prices_per_day(START_DATE, END_DATE)
thesis.load_stations_per_day(START_DATE, END_DATE)

# Get all stations in range
stations = thesis.load_unique_stations_uuid_in_range(START_DATE, END_DATE)

# Get recent stations data in range
thesis.load_unique_stations_in_range(stations, START_DATE, END_DATE)

# Calculate the mean price for diesel e5 e10 for every station
thesis.mean_price_per_station_per_day(START_DATE, END_DATE)

# Categorize stations by price range
thesis.categorize_station_by_price_in_range(START_DATE, END_DATE)

# Create a bar chart for the price range of every station
thesis.create_bar_chart_for_stations_by_price()

# Percentage of stations with negative prices or zero
thesis.count_stations_percentage_negative_zero(START_DATE, END_DATE)

# Create a dataframe from the population data
df_pop = thesis.load_population_data(0)

# Create a dataframe from the border data
borders = {}
for country in COUNTRIES:
    borders[country] = thesis.load_border_data(country)

# Create a dataframe from the road data
df_road_1 = thesis.load_road_data(1)

# Calculate the distance between every station and the border
df_deu = thesis.calculate_distances_to_border("e5", "DEU")

# Create booleans for every country if station is within 20km of the border
for country in COUNTRIES:
    df = thesis.calculate_dummy_distance_to_border("e5", country)
    df_deu[f"dummy_border_{country}"] = df[f"distance_to_border_{country}_dummy"]

df_deu = df_deu.drop(columns=["distance_to_border_DEU"])
feather.write_feather(
    df_deu, f"{PROCESSED_PATH}/stations/all_unique_e5_with_distance_to_border.feather"
)

# Calculate the distance between every station and the highway
df = thesis.calculate_distances_to_highway("e5")

# Calculate the amount of population in a radius of 1,3,5km
df = thesis.calculate_population("e5")

# Calculate the amount of competitors in a radius of 05,1,2,3,5km
df = thesis.calculate_competition("e5")

# --------------------------------------
# Generate AVG map
df_stations = thesis.generate_station_data("e5")
df = thesis.generate_avg_prices_for_range_per_station("e5", START_DATE, END_DATE)

# Add langitude longitude to the dataframe
map_data = pd.merge(df, df_stations, on="uuid")
columns_drop = df_stations.columns.to_list()
columns_drop.remove("latitude")
columns_drop.remove("longitude")
columns_drop.append("uuid")
map_data.drop(columns_drop, axis=1, inplace=True)

thesis.generate_shapefile(map_data)
# --------------------------------------

# START CREATING THE PANEL DATA

# load dataframes from feather files
df_stations = thesis.generate_station_data("e5")

# Create a dataframe with the mean price for every station in range
df_avg = thesis.generate_avg_prices_per_day_per_station("e5", START_DATE, END_DATE)

# Merge the dataframes
panel_data = pd.merge(df_avg, df_stations, on="uuid")

panel_data["date"] = pd.to_datetime(panel_data["date"])

# Save the dataframe to a feather file
feather.write_feather(panel_data, "panel_data_e5.feather")  # type: ignore

# Generate calculated panel data
panel_data["tax"] = np.where(panel_data["date"] < datetime.datetime(2022, 6, 1), 0.6545, 0.3590)
panel_data["pop_1"] = panel_data["pop_1"] / 10000
panel_data["pop_3"] = panel_data["pop_3"] / 10000
panel_data["pop_5"] = panel_data["pop_5"] / 10000
panel_data["dummy_distance_to_highway_500"] = np.where(
    panel_data["distance_to_highway"] <= 0.5, 1, 0
)

columns: List[str] = list(panel_data.columns)
columns_to_not_use = ["date", "uuid", "latitude", "longitude", "tax", "price", "brand", "name"]

for column in columns:
    if column not in columns_to_not_use:
        panel_data[f"tax_{column}"] = panel_data[column] * panel_data["tax"]

print(panel_data.head(10))  # type: ignore
feather.write_feather(panel_data, "panel_data_e5_tax.feather")

panel_data: pd.DataFrame = feather.read_feather("panel_data_e5_tax.feather")
panel_data["day"] = panel_data["date"].apply(lambda x: (x - START_DATE).days + 1)
panel_data["dummy"] = np.where(panel_data["date"] < datetime.datetime(2022, 6, 1), 0, 1)
panel_data["day_post"] = (
    panel_data["date"].apply(lambda x: (x - datetime.datetime(2022, 6, 1)).days + 1)
    * panel_data["dummy"]
)
panel_data = panel_data.set_index(["uuid", "date"])


# Do the regressions
print(thesis.pooled_ols_regression(panel_data))
print(thesis.pooled_ols_regression_with_interaction(panel_data))
print(thesis.pooled_ols_regression_fixed_effects(panel_data))
print(thesis.pooled_ols_regression_fixed_effects_with_interaction(panel_data))
