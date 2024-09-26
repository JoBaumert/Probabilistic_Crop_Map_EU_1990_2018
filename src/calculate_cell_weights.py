#%%
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.plot import show
from rasterio import features
import numpy as np
import os
from pathlib import Path
from src.utils.preprocess_crop_data import cluster_crop_names

#%%
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
NUTS_code_mapping=crop_data[["CAPRI_code","NUTS_ID"]].drop_duplicates(["CAPRI_code","NUTS_ID"])
#%%
"""cluster crops accordingly"""
crop_delineation=pd.read_excel(delineation_and_parameter_path+"DGPCM_crop_delineation.xlsx")
all_crops=np.sort(crop_delineation.dropna(subset="DGPCM_code")["DGPCM_code"].unique())
#%%
preprocessed_crop_data=cluster_crop_names(crop_data=crop_data,crop_delineation=crop_delineation,
                                          all_crops=all_crops)
#%%
preprocessed_crop_data
#%%

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
#create one 3 dimensional matrix that contains all the corine data as bands (one per year)
corine_raw_allyears=[]
for i,cy in enumerate(corine_years_unique):
    corine_raw_allyears.append(rio.open(raw_data_path+"CORINE_"+str(cy)+".tif").read()[0])

corine_preprocessed_allyears=np.array(corine_raw_allyears)
nan_indices=np.where(np.isnan(corine_preprocessed_allyears[0]))
corine_preprocessed_allyears[0][nan_indices]=corine_preprocessed_allyears[1][nan_indices]

corine_preprocessed_19902018=[]
for year in corine_years:
    corine_preprocessed_19902018.append(corine_preprocessed_allyears[np.where(corine_years_unique==year)[0][0]])

corine_preprocessed_19902018=np.array(corine_preprocessed_19902018)
# %%
corine_preprocessed_19902018.shape
# %%

output_raster_UAA=np.where(corine_preprocessed_19902018[np.where(years==year)[0][0]]>0,
                           output_raster,-1)
# %%

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
show(np.where(np.isin(output_raster,new_mapping_df["index"].unique()),1,0))
#%%
relevant_cropdata=preprocessed_crop_data[
    (preprocessed_crop_data["year"]==year)&(preprocessed_crop_data["NUTS_ID"].isin(new_mapping_df["NUTS_ID"]))]

# %%
relevant_cropdata.drop_duplicates(subset=["CAPRI_code","DGPCM_crop_code"],inplace=True)
relevant_cropdata.drop("NUTS_ID",axis=1,inplace=True)
#%%
uaa_calculated=relevant_cropdata[["CAPRI_code","year","country","value"]].groupby(
    ["CAPRI_code","year","country"]
).sum().reset_index()

uaa_calculated.rename(columns={"value":"UAA_CAPRI"},inplace=True)


# %%
corine_data_relevant=corine_preprocessed_19902018[np.where(years==year)[0][0]]
# %%
uaa_CORINE_list=[]
index_list_old=[]
index_list_new=[]
index_list_new_wo_duplicates=[]
nuts_id_list=[]
for i,capri_code in enumerate(np.array(uaa_calculated["CAPRI_code"])):
    nuts_ids=np.array(NUTS_code_mapping[NUTS_code_mapping["CAPRI_code"]==capri_code]["NUTS_ID"])
    if len(nuts_ids)==1:
        nuts_id=nuts_ids[0]
        index=new_mapping_df[new_mapping_df["NUTS_ID"]==nuts_id]["index"].iloc[0]
        uaa_CORINE_list.append(np.nansum(corine_data_relevant[np.where(output_raster==index)])/10)
        index_list_old.append(index)
        index_list_new.append(index)
        nuts_id_list.append(nuts_id)
        index_list_new_wo_duplicates.append(index)
    else:
        uaa=0
        index_list_new_wo_duplicates.append(1000+i)
        for nuts_id in nuts_ids:
            index=new_mapping_df[new_mapping_df["NUTS_ID"]==nuts_id]["index"].iloc[0]
            uaa+=np.nansum(corine_data_relevant[np.where(output_raster==index)])/10
            index_list_old.append(index)
            index_list_new.append(1000+i)
            nuts_id_list.append(nuts_id)
        uaa_CORINE_list.append(uaa)

uaa_calculated["UAA_CORINE"]=uaa_CORINE_list
# %%
nuts_indices_relevant=output_raster.copy()
changed_index_positions=np.where(np.invert(np.array(index_list_old)==np.array(index_list_new)))[0]
for i,index in enumerate(np.array(index_list_old)[changed_index_positions]):
    nuts_indices_relevant[np.where(nuts_indices_relevant==index)]=index_list_new[changed_index_positions[i]]

nuts_indices_relevant=np.where((np.isin(nuts_indices_relevant,index_list_new))&(corine_data_relevant>0),nuts_indices_relevant,-1)

#%%
uaa_calculated["index"]=index_list_new_wo_duplicates
# %%
uaa_raster_relevant=np.where(nuts_indices_relevant>=0,corine_data_relevant,np.nan)
# %%
uaa_calculated["weight_factor"]=uaa_calculated["UAA_CAPRI"]/uaa_calculated["UAA_CORINE"]
# %%
weight_raster=np.ones_like(uaa_raster_relevant)*(-1)
for i,index in enumerate(np.array(uaa_calculated["index"])):
    weight_factor=uaa_calculated["weight_factor"].iloc[i]
    weight_raster=np.where(nuts_indices_relevant==index,uaa_raster_relevant*weight_factor,weight_raster)
# %%
show(np.where(weight_raster>1,1,0))
# %%
show(np.where(nuts_indices_relevant==87,weight_raster,0))
# %%
uaa_calculated
# %%
