#%%
from pathlib import Path
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import Window
import numpy as np
from collections import Counter
from rasterio.mask import mask
from tqdm import tqdm
from shapely.geometry import box
from utils.preprocess_raster import read_raster, preprocess, filter_raster
import os

corine_data_path = "data/raw/corine"
nuts_data_path = Path('data/raw/NUTS2_2016_raster.tif')
preprocessed_data_path = "data/preprocessed/"
result_data_path = "data/results/"

agri_class_codes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 22]


def process_corine(corine_filepath, nuts_filepath, output_filepath):
    corine_year = os.path.basename(corine_filepath)
    clipped_corine_file = Path(preprocessed_data_path+'clipped_corine_'+corine_year)
    clipped_nuts_file = Path(preprocessed_data_path+'clipped_nuts_'+corine_year)
    preprocess(corine_filepath,
                nuts_filepath,
                clipped_corine_file,
                clipped_nuts_file)


    src_corine, raster_corine, transform_corine, crs_corine = read_raster(clipped_corine_file)
    src_nuts, raster_nuts, transform_nuts, crs_nuts = read_raster(clipped_nuts_file)


    raster_corine_filtered = filter_raster(raster_corine, agri_class_codes)
    raster_nuts = raster_nuts.squeeze()
    raster_corine_filtered = raster_corine_filtered.squeeze()
    output_data = np.zeros((raster_nuts.shape[0], raster_nuts.shape[1]), dtype=np.float32)

    window_shape = (10, 10)

    reshaped_corine = raster_corine_filtered.reshape(
        raster_nuts.shape[0], window_shape[0], raster_nuts.shape[1], window_shape[1]
    )
    avg_values = np.mean(reshaped_corine, axis=(1, 3))

    for i in tqdm(range(raster_nuts.shape[0])):
        for j in range(raster_nuts.shape[1]):
            output_data[i, j] = avg_values[i, j]


    meta_nuts = src_nuts.meta
    meta_output = meta_nuts
    meta_output.update(dtype=rio.float32, count=1)

    with rio.open(output_filepath, 'w', **meta_nuts) as dst:
        dst.write(output_data, 1)


if __name__ == "__main__":

    for filename in tqdm(os.listdir(corine_data_path)):
        corine_file_path = os.path.join(corine_data_path, filename)
        output_file = result_data_path+"output_raster_"+filename
        if os.path.isfile(corine_file_path):
            process_corine(corine_file_path, nuts_data_path, output_file)




