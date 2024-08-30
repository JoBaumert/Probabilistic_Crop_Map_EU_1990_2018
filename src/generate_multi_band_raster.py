#%%
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.plot import show
from rasterio import features
import numpy as np
import os
from pathlib import Path
#%%
main_path = str(Path(Path(os.path.abspath(__file__)).parents[1]))
result_dir = os.path.join(main_path, "data/results/")
os.makedirs(result_dir, exist_ok=True)
raw_dir = main_path+"/data/raw/"
os.makedirs(raw_dir, exist_ok=True)
preprocessed_dir = main_path+"/data/preprocessed/"
os.makedirs(preprocessed_dir, exist_ok=True)
# List of years to process
years = [2003, 2006, 2010, 2013, 2016, 2021]

#%%
# Open the reference raster to get the transform and shape
raster_src = rio.open(preprocessed_dir + "rasters/clipped_nuts_2018.tif")
transform = raster_src.transform
out_shape = raster_src.shape  

# DataFrame to store the combined dictionary for all years
combined_df = pd.DataFrame()
#%%
for year in years:
    file_path = os.path.join(raw_dir, f"nuts_shapedata/NUTS_RG_01M_{year}_3035.shp/NUTS_RG_01M_{year}_3035.shp")
    
    # Load the shapefile
    NUTS_shapefile = gpd.read_file(file_path, encoding='latin1')

    # Create a list to hold the rasters for each band (NUTS levels 0 to 3)
    bands = []
    for level in range(4):
        NUTS_level_df = NUTS_shapefile[NUTS_shapefile["LEVL_CODE"] == level].copy()
        NUTS_level_df.sort_values(by="NUTS_ID", inplace=True)
        NUTS_level_df["index"] = np.arange(len(NUTS_level_df)) + 1

        temp_df = NUTS_level_df[["NUTS_ID", "LEVL_CODE", "index"]].copy()
        temp_df["year"] = year
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        
        geom_value = (
            (geom, value) 
            for geom, value in zip(NUTS_level_df.geometry, NUTS_level_df["index"])
        )
        
        raster = features.rasterize(
            geom_value,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='uint16'
        )
        bands.append(raster)
    
    # Update metadata and write multi-band raster
    meta_output = raster_src.meta.copy()
    meta_output.update(dtype='uint16', count=4, compress='lzw')
    
    raster_path = os.path.join(result_dir, f"multi_band_raster/nuts_raster_{year}.tif")
    with rio.open(raster_path, 'w', **meta_output) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i + 1)
    
    print(f"Multi-band raster for {year} saved at {raster_path}")

# Save the combined dictionary to a CSV file
combined_df.to_csv(os.path.join(result_dir, "csv/nuts_regions_dictionary.csv"), index=False)


# %% sanity check
with rio.open(result_dir+'multi_band_raster/nuts_raster_2003.tif') as mbr:
    show(mbr.read(1))
    show(mbr.read(2))
    show(mbr.read(3))
    show(mbr.read(4))

# %%