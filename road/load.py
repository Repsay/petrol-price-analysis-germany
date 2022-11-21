import shapefile as shp
import pandas as pd
import pyarrow.feather as feather

def flatten_list(value):
    flat_list = []
    for elem in value:
        if type(elem) == list:
            flat_list.extend(flatten_list(elem))
        else:
            flat_list.append(elem)

    return flat_list

def prepare_data(filter_value = 1):
    sf = shp.Reader("D:/Thesis/petrol-price-analysis-germany/road/data/German_roads")

    filter_field = 'GP_RTP'

    records = []

    for rec in sf.iterRecords(fields=[filter_field]):
        if rec[filter_field] == filter_value:
            records.append(sf.shapeRecord(rec.oid))

    coordinates_list = []

    for record in records:
        geo_data = record.__geo_interface__['geometry']
        coordinates = flatten_list(geo_data['coordinates'])

        for item in coordinates:
            coordinates_list.append(item[::-1])

    coordinates_list = list(set(coordinates_list))

    df = pd.DataFrame(coordinates_list, columns=['lat', 'lon'])

    new_fn = f'D:/Thesis/petrol-price-analysis-germany/processed-data/road/germany-{filter_value}.feather'

    feather.write_feather(df, new_fn)