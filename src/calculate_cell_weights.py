#%%
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.plot import show
from rasterio import features
import numpy as np
import os
from pathlib import Path
# %%

main_path = str(Path(Path(os.path.abspath(__file__)).parents[1]))
result_dir = os.path.join(main_path, "data/results/")
os.makedirs(result_dir, exist_ok=True)
raw_dir = main_path+"/data/raw/"
os.makedirs(raw_dir, exist_ok=True)
preprocessed_dir = main_path+"/data/preprocessed/"
os.makedirs(preprocessed_dir, exist_ok=True)
# %%
nuts_regions_dictionary=pd.read_csv(result_dir+"csv/nuts_regions_dictionary.csv")
# %%
crop_data=pd.read_csv(preprocessed_dir+"csv/relevant_crop_data.csv")
# %%
year=1990
relevant_nuts_ids=crop_data[crop_data["year"]==year]["NUTS_ID"].unique()
# %%
relevant_nuts_ids
# %%
nuts_regions_dictionary_year=nuts_regions_dictionary.iloc[
    np.where(np.isin(nuts_regions_dictionary["NUTS_ID"],relevant_nuts_ids))[0]
]
# %%
nuts_regions_dictionary_year=nuts_regions_dictionary_year.sort_values(by="year").drop_duplicates(["NUTS_ID"])
# %%
new_mapping_dict={"year":[],"NUTS_ID":[],"index":[]}
reference_raster=rio.open(result_dir+"multi_band_raster/nuts_raster_2003.tif")
output_raster=np.tile(-1,reference_raster.shape)
index_counter=0
for nuts_year in np.sort(nuts_regions_dictionary_year["year"].unique()):
    relevant_raster=rio.open(result_dir+"multi_band_raster/nuts_raster_"+str(nuts_year)+".tif").read()
    
    for level in np.sort(
            nuts_regions_dictionary_year[nuts_regions_dictionary_year["year"]==nuts_year]["LEVL_CODE"].unique()
        ):
        relevant_band=relevant_raster[level]

        nuts_regions_dictionary_year_level=nuts_regions_dictionary_year[(nuts_regions_dictionary_year["year"]==nuts_year)&
                                    (nuts_regions_dictionary_year["LEVL_CODE"]==level)]

        for i in range(len(nuts_regions_dictionary_year_level)):
            new_mapping_dict["year"].append(year)
            new_mapping_dict["NUTS_ID"].append(nuts_regions_dictionary_year_level["NUTS_ID"].iloc[i])
            new_mapping_dict["index"].append(index_counter)
            output_raster[np.where(relevant_band==nuts_regions_dictionary_year_level["index"].iloc[i])]=index_counter
            index_counter+=1
    
# %%
show(np.where(output_raster>=0,1,0))

# %%
new_mapping_df=pd.DataFrame(new_mapping_dict)

freq=np.ndarray(len(new_mapping_df))
for i in range(len(new_mapping_df)):
    freq[i]=len(np.where(output_raster.flatten()==i)[0])
    
new_mapping_df["freq"]=freq
new_mapping_df=new_mapping_df[new_mapping_df["freq"]>0]

new_mapping_df["country"]=np.array(new_mapping_df["NUTS_ID"]).astype("U2")
new_mapping_df["NUTS_LEVL"]=np.char.str_len(np.array(new_mapping_df["NUTS_ID"]).astype(str))-2
# %%
show(np.where(output_raster==2,1,0))
# %%
new_mapping_df
# %%
corine=rio.open(preprocessed_dir+"rasters/clipped_corine_1990.tif").read()[0]
# %%
show(corine)
#%%
np.unique(corine,return_counts=True)
# %%
