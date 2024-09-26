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
main_path_data_external="/home/baumert/fdiexchange/baumert/DGPCM_19902020/Data/raw/GEE_DGPCM/"
raw_data_path=main_path_data_external+"Raw_Data/"
delineation_and_parameter_path=main_path_data_external+"delineation_and_parameters/"
main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))
result_dir = os.path.join(main_path, "data/results/")
os.makedirs(result_dir, exist_ok=True)
raw_dir = main_path+"/data/raw/"
os.makedirs(raw_dir, exist_ok=True)
preprocessed_dir = main_path+"/data/preprocessed/"
os.makedirs(preprocessed_dir, exist_ok=True)

#%%
nuts_regions_dictionary=pd.read_csv(result_dir+"csv/nuts_regions_dictionary.csv")
crop_data=pd.read_csv(preprocessed_dir+"csv/relevant_crop_data.csv")
# %%
NUTS_code_mapping=crop_data[["CAPRI_code","NUTS_ID"]].drop_duplicates(["CAPRI_code","NUTS_ID"])
#%%
year=1990
relevant_nuts_ids=crop_data[crop_data["year"]==year]["NUTS_ID"].unique()

# %%
nuts_regions_dictionary_year=nuts_regions_dictionary.iloc[
    np.where(np.isin(nuts_regions_dictionary["NUTS_ID"],relevant_nuts_ids))[0]
]

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
"""preprocess CORINE data"""
corine_parameters=pd.read_excel(delineation_and_parameter_path+"user_parameters.xlsx",sheet_name="CORINE_years")
#%%
years=np.array(corine_parameters["year"])
corine_years=np.array(corine_parameters["CORINE_year"])
corine_years_unique=np.unique(corine_years)
#%%
corine_raw_allyears=[]
for i,cy in enumerate(corine_years_unique):
    corine_raw_allyears.append(rio.open(raw_data_path+"CORINE_"+str(cy)+".tif").read()[0])
#%%
corine_preprocessed_allyears=np.array(corine_raw_allyears)
#%%
nan_indices=np.where(np.isnan(corine_preprocessed_allyears[0]))
corine_preprocessed_allyears[0][nan_indices]=corine_preprocessed_allyears[1][nan_indices]
# %%
corine_preprocessed_19902018=[]
for year in corine_years:
    corine_preprocessed_19902018.append(corine_preprocessed_allyears[np.where(corine_years_unique==year)[0][0]])
# %%
corine_preprocessed_19902018=np.array(corine_preprocessed_19902018)
# %%
corine_preprocessed_19902018.shape
# %%
year=1990
output_raster_UAA=np.where(corine_preprocessed_19902018[np.where(years==year)[0][0]]>0,
                           output_raster,-1)
# %%
show(np.where(output_raster==43,1,0))
# %%
corine_preprocessed_19902018.shape
# %%
new_mapping_df=pd.DataFrame(new_mapping_dict)

freq=np.ndarray(len(new_mapping_df))
for i in range(len(new_mapping_df)):
    freq[i]=len(np.where(output_raster_UAA.flatten()==i)[0])
    
new_mapping_df["freq"]=freq
new_mapping_df=new_mapping_df[new_mapping_df["freq"]>0]

new_mapping_df["NUTS0"]=np.array(new_mapping_df["NUTS_ID"]).astype("U2")
new_mapping_df["NUTS1"]=np.array(new_mapping_df["NUTS_ID"]).astype("U3")
new_mapping_df["NUTS2"]=np.array(new_mapping_df["NUTS_ID"]).astype("U4")
new_mapping_df["NUTS_LEVL"]=np.char.str_len(np.array(new_mapping_df["NUTS_ID"]).astype(str))-2
# %%
new_mapping_df[new_mapping_df["NUTS0"]=="DE"]
# %%
excluded_nuts_ids=[]
for country in new_mapping_df["NUTS0"].unique():
    
    country_df=new_mapping_df[new_mapping_df["NUTS0"]==country]
    levels=np.sort(country_df["NUTS_LEVL"].unique())
    max_level=levels.max()
    if len(levels)>1:
        for level in levels:
            if level==max_level:
                continue
            for nuts_id in np.sort(country_df[country_df["NUTS_LEVL"]==level]["NUTS_ID"].unique()):
                own_freq=country_df[country_df["NUTS_ID"]==nuts_id]["freq"].iloc[0]
                subregion_freq=country_df.iloc[np.where(country_df["NUTS"+str(level)]==nuts_id)]["freq"].sum()

                if subregion_freq>0:
                    print(nuts_id+": "+str(own_freq/subregion_freq))
                    if (own_freq/subregion_freq)<0.01:
                        excluded_nuts_ids.append(nuts_id)

    else:
        continue

# %%
new_mapping_df=new_mapping_df[~new_mapping_df["NUTS_ID"].isin(excluded_nuts_ids)]
# %%
relevant_cropdata=crop_data[(crop_data["year"]==year)&(crop_data["NUTS_ID"].isin(new_mapping_df["NUTS_ID"]))]

# %%
relevant_cropdata
# %%
NUTS_code_mapping["NUTS_ID"].value_counts()
# %%
