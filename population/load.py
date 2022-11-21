import rasterio
import numpy as np
import pandas as pd
import pyarrow.feather as feather

def prepare_data():
    dataset = rasterio.open('D:/Thesis/population/data/Population_germany_2.tif')
    band1 = dataset.read(1)
    height, width = band1.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(dataset.transform, rows, cols) # type: ignore
    lons = np.array(xs)
    lats = np.array(ys)
    coordinates = np.dstack((lats,lons, band1))
    df = pd.DataFrame(coordinates.reshape(-1, 3), columns=['lat', 'lon', 'population'])

    new_fn = 'D:/Thesis/processed-data/population/population_germany.feather'

    feather.write_feather(df, new_fn)