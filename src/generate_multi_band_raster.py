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
main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))


main_path=str(Path(Path(os.path.abspath(__file__)).parents[0]))
data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]

raw_dir = data_main_path+"/raw"
preprocessed_dir = data_main_path+"/preprocessed"
preprocessed_csv_dir=preprocessed_dir+"/csv/"
preprocessed_raster_dir=preprocessed_dir+"/rasters/"
os.makedirs(preprocessed_raster_dir, exist_ok=True)

parameter_path=data_main_path+"/delineation_and_parameters/"
user_parameter_path=parameter_path+"user_parameters.xlsx"



# List of years to process
years = [2003, 2006, 2010, 2013, 2016, 2021]
#%%
geospatial_transformation=pd.read_excel(user_parameter_path,sheet_name="geospatial_transformation")
#%%
transform=tuple(np.array(geospatial_transformation["transform"]).astype(float))
out_shape=tuple(np.array(geospatial_transformation["shape"][:2]).astype(int))
#%%

# DataFrame to store the combined dictionary for all years
combined_df = pd.DataFrame()


#%%
for year in years:
    file_path = raw_dir+"/nuts_shapedata/NUTS_RG_01M_"+str(year)+"_3035.shp/NUTS_RG_01M_"+str(year)+"_3035.shp"
    
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

    
    with rio.open(preprocessed_raster_dir+"nuts_"+str(year)+".tif", 'w',
                width=int(out_shape[1]),height=int(out_shape[0]),
                transform=transform,count=4,dtype=rio.int16,crs="EPSG:3035") as dst:
        dst.write(np.array(bands).astype(rio.int16))

    print(f"Multi-band raster for {year} saved")

#%%
# Save the combined dictionary to a CSV file
combined_df.to_csv(preprocessed_csv_dir+"nuts_regions_dictionary.csv", index=False)



# %%
