from typing import Any
import shapefile as shp
import pandas as pd
import pyarrow.feather as feather

def prepare_data(countries = ["AUT", "BEL", "CHE", "CZE", "DEU", "DNK", "FRA", "LUX", "NLD", "POL"]):
    for country in countries:
        sf = shp.Reader(f'D:/Thesis/petrol-price-analysis-germany/border/data/{country}/gadm41_{country}_0')
        records: shp.ShapeRecord = sf.shapeRecords()[0]

        geo_data: Any = records.__geo_interface__['geometry']
        lines = geo_data['coordinates']

        coordinates_list = []

        for line in lines:
            for coordinates in line[0]:
                coordinates_list.append(coordinates[::-1])

        coordinates_list = list(set(coordinates_list))

        df = pd.DataFrame(coordinates_list, columns=['lat', 'lon'])

        new_fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/borders/{country}.feather'

        feather.write_feather(df, new_fn)