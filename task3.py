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
from preprocess_raster import read_raster, preprocess, filter_raster

#%%
original_corine_file = Path('data/U2018_CLC2018_V2020_20u1.tif')
original_nuts_file = Path('data/NUTS2_2016_raster.tif')
clipped_nuts_file = Path('data/clipped_nuts.tif')
clipped_corine_file = Path('data/clipped_corine.tif')
output_raster_path = Path('results/result.tif')

agri_class_codes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 22]


#%%
# Clipping both files, what's happening here is that I'm only using the bounds from the NUTS file
# to avoid computation for regions we don't need (Turkey, etc.)

if not clipped_corine_file.is_file() and not clipped_nuts_file.is_file():
    preprocess(original_corine_file,
            original_nuts_file,
            clipped_corine_file,
            clipped_nuts_file)

#%%
src_corine, raster_corine, transform_corine, crs_corine = read_raster(clipped_corine_file)
src_nuts, raster_nuts, transform_nuts, crs_nuts = read_raster(clipped_nuts_file)

#%%

raster_corine_filtered = filter_raster(raster_corine, agri_class_codes)
raster_corine_filtered.shape
#%%
np.unique(raster_corine_filtered)
# %%
show(raster_corine_filtered)


# %%
show(raster_nuts)

#%%
raster_nuts = raster_nuts.squeeze()
raster_corine_filtered = raster_corine_filtered.squeeze()
output_data = np.zeros((raster_nuts.shape[0], raster_nuts.shape[1]), dtype=np.float32)

#This works but the uncommented version is faster

# for i in tqdm(range(raster_nuts.shape[0])):
#     for j in range(raster_nuts.shape[1]):
#         row_start = i * 10
#         col_start = j * 10
#         window = (row_start, 10, col_start, 10)
#         avg_value = np.mean(raster_corine_filtered[slice(row_start,10),slice(col_start,10)])
#         output_data[i, j] = avg_value

window_shape = (10, 10)

reshaped_corine = raster_corine_filtered.reshape(
    raster_nuts.shape[0], window_shape[0], raster_nuts.shape[1], window_shape[1]
)
avg_values = np.mean(reshaped_corine, axis=(1, 3))

for i in tqdm(range(raster_nuts.shape[0])):
    for j in range(raster_nuts.shape[1]):
        output_data[i, j] = avg_values[i, j]

# %%
# Found a easier way to save
meta_nuts = src_nuts.meta
meta_output = meta_nuts
meta_output.update(dtype=rio.float32, count=1)

with rio.open(output_raster_path, 'w', **meta_nuts) as dst:
    dst.write(output_data, 1)

# %%

#Sanity Check - read saved raster and display

src_output, img, _ , _ = read_raster(output_raster_path)

show(img)
# %%
np.unique(img)

# %%
src_output.meta


# %%
import matplotlib.pyplot as plt
plt.hist(img.flatten())
# %%
