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
#main_path_data_external="/home/baumert/fdiexchange/baumert/DGPCM_19902020/Data/raw/GEE_DGPCM/"
#raw_data_path=main_path_data_external+"Raw_Data/"
#delineation_and_parameter_path=main_path_data_external+"delineation_and_parameters/"
#main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))
#result_dir = os.path.join(main_path, "data/results/")
#os.makedirs(result_dir, exist_ok=True)
#raw_dir = main_path+"/data/raw/"
#os.makedirs(raw_dir, exist_ok=True)
#preprocessed_dir = main_path+"/data/preprocessed/"
#os.makedirs(preprocessed_dir, exist_ok=True)
#%%
main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))
data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]

raw_dir = data_main_path+"/raw"
preprocessed_dir = data_main_path+"/preprocessed"
preprocessed_csv_dir=preprocessed_dir+"/csv/"
preprocessed_raster_dir=preprocessed_dir+"/rasters/"
os.makedirs(preprocessed_raster_dir, exist_ok=True)

parameter_path=data_main_path+"/delineation_and_parameters/"
user_parameter_path=parameter_path+"user_parameters.xlsx"
#TODO: change the following path according to system
GEE_data_path="/home/baumert/fdiexchange/baumert/DGPCM_19902020/Data/raw/GEE_DGPCM/Raw_Data/"

#%%
nuts_regions_dictionary=pd.read_csv(preprocessed_csv_dir+"nuts_regions_dictionary.csv")
crop_data=pd.read_csv(preprocessed_csv_dir+"preprocessed_CAPREG_step2.csv")
NUTS_code_mapping=crop_data[["CAPRI_code","NUTS_ID"]].drop_duplicates(["CAPRI_code","NUTS_ID"])

#%%
"""preprocess CORINE data"""
corine_parameters=pd.read_excel(parameter_path+"user_parameters.xlsx",sheet_name="CORINE_years")

years=np.array(corine_parameters["year"])
corine_years=np.array(corine_parameters["CORINE_year"])
corine_years_unique=np.unique(corine_years)
#%%
#create one 3 dimensional matrix that contains all the corine data as bands (one per year)
corine_raw_allyears=[]
for i,cy in enumerate(corine_years_unique):
    corine_raw_allyears.append(rio.open(GEE_data_path+"CORINE_"+str(cy)+".tif").read()[0])

corine_preprocessed_allyears=np.array(corine_raw_allyears)
nan_indices=np.where(np.isnan(corine_preprocessed_allyears[0]))
corine_preprocessed_allyears[0][nan_indices]=corine_preprocessed_allyears[1][nan_indices]

corine_preprocessed_19902018=[]
for y in corine_years:
    corine_preprocessed_19902018.append(corine_preprocessed_allyears[np.where(corine_years_unique==y)[0][0]])

corine_preprocessed_19902018=np.array(corine_preprocessed_19902018)

#%%
"""cluster crops accordingly"""
crop_delineation=pd.read_excel(parameter_path+"DGPCM_crop_delineation.xlsx")
all_crops=np.sort(crop_delineation.dropna(subset="DGPCM_code")["DGPCM_code"].unique())

preprocessed_crop_data=cluster_crop_names(crop_data=crop_data,crop_delineation=crop_delineation,
                                          all_crops=all_crops)


countries=np.sort(np.array(pd.read_excel(parameter_path+"user_parameters.xlsx",sheet_name="countries")["country_code"]))
preprocessed_crop_data=preprocessed_crop_data[preprocessed_crop_data["country"].isin(countries)]


#%%
#output dfs and rasters:
uaa_calculated_allyears=pd.DataFrame()
weight_raster_allyears=np.ndarray(corine_preprocessed_19902018.shape)
nuts_indices_relevant_allyears=np.ndarray(corine_preprocessed_19902018.shape)
uaa_raster_relevant_allyears=np.ndarray(corine_preprocessed_19902018.shape)


for j,year in enumerate(years):
    print(year)

    #get all NUTS IDs that appear in the CAPRI data for the respective year
    relevant_nuts_ids=crop_data[crop_data["year"]==year]["NUTS_ID"].unique()


    #get the dictionary that links NUTS IDs and raster indices for the respective year
    nuts_regions_dictionary_year=nuts_regions_dictionary.iloc[
        np.where(np.isin(nuts_regions_dictionary["NUTS_ID"],relevant_nuts_ids))[0]
    ]
    #drop all duplicates so that in case a NUTS region definition exists throughout several years only the year where it occurs first is kept and
    #nuts_regions_dictionary_year only contains per NUTS region one year
    nuts_regions_dictionary_year=nuts_regions_dictionary_year.sort_values(by="year").drop_duplicates(["NUTS_ID"])


    new_mapping_dict={"year":[],"NUTS_ID":[],"index":[]}
    reference_raster=rio.open(preprocessed_raster_dir+"nuts_2003.tif")
    output_raster=np.tile(-1,reference_raster.shape)
    index_counter=0
    """a NUTS region that appears in the CAPRI data for, e.g., 1995 may exist in several NUTS region definitions (2003,2006,...,2021).
    here the spatial information is merged so that to each NUTS region that occurs in the CAPRI data grid cells can be allocated
    """
    for nuts_year in np.sort(nuts_regions_dictionary_year["year"].unique()):
        relevant_raster=rio.open(preprocessed_raster_dir+"nuts_"+str(nuts_year)+".tif").read()
        
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
        

    output_raster_UAA=np.where(corine_preprocessed_19902018[np.where(years==year)[0][0]]>0,
                            output_raster,-1)


    new_mapping_df=pd.DataFrame(new_mapping_dict)

    """calculate how many cells belong to each listed NUTS region. Some NUTS regions do not have agricultural area and therefore are discarded.
    Besides, we discard NUTS regions for which subregions information is available (if the subregions for which information is available cover more than
    1% of the region's area).
    """
    freq=np.ndarray(len(new_mapping_df))
    for i in range(len(new_mapping_df)):
        freq[i]=len(np.where(output_raster_UAA.flatten()==i)[0])
        
    new_mapping_df["freq"]=freq
    #discard NUTS regions without agricultural area
    new_mapping_df=new_mapping_df[new_mapping_df["freq"]>0]

    new_mapping_df["NUTS0"]=np.array(new_mapping_df["NUTS_ID"]).astype("U2")
    new_mapping_df["NUTS1"]=np.array(new_mapping_df["NUTS_ID"]).astype("U3")
    new_mapping_df["NUTS2"]=np.array(new_mapping_df["NUTS_ID"]).astype("U4")
    new_mapping_df["NUTS_LEVL"]=np.char.str_len(np.array(new_mapping_df["NUTS_ID"]).astype(str))-2


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
                        #print(nuts_id+": "+str(own_freq/subregion_freq))

                        #keep only the most disaggregated level
                        if (own_freq/subregion_freq)<0.01:
                            excluded_nuts_ids.append(nuts_id)

        else:
            continue



    new_mapping_df=new_mapping_df[~new_mapping_df["NUTS_ID"].isin(excluded_nuts_ids)]

    #show(np.where(np.isin(output_raster,new_mapping_df["index"].unique()),1,0))

    relevant_cropdata=preprocessed_crop_data[
        (preprocessed_crop_data["year"]==year)&(preprocessed_crop_data["NUTS_ID"].isin(new_mapping_df["NUTS_ID"]))]


    relevant_cropdata.drop_duplicates(subset=["CAPRI_code","DGPCM_crop_code"],inplace=True)
   
    relevant_cropdata.drop("NUTS_ID",axis=1,inplace=True)

    """CALCULATE UAA FROM CAPRI AND CORINE"""
    uaa_calculated=relevant_cropdata[["CAPRI_code","year","country","value"]].groupby(
        ["CAPRI_code","year","country"]
    ).sum().reset_index()

    uaa_calculated.rename(columns={"value":"UAA_CAPRI"},inplace=True)

    uaa_calculated["UAA_CAPRI"]=uaa_calculated["UAA_CAPRI"]*10

    corine_data_relevant=corine_preprocessed_19902018[np.where(years==year)[0][0]]

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
            uaa_CORINE_list.append(np.nansum(corine_data_relevant[np.where(output_raster==index)]))
            index_list_old.append(index)
            index_list_new.append(index)
            nuts_id_list.append(nuts_id)
            index_list_new_wo_duplicates.append(index)
        else:
            uaa=0
            index_list_new_wo_duplicates.append(1000+i)
            for nuts_id in nuts_ids:
                index=new_mapping_df[new_mapping_df["NUTS_ID"]==nuts_id]["index"].iloc[0]
                uaa+=np.nansum(corine_data_relevant[np.where(output_raster==index)])
                index_list_old.append(index)
                index_list_new.append(1000+i)
                nuts_id_list.append(nuts_id)
            uaa_CORINE_list.append(uaa)

    uaa_calculated["UAA_CORINE"]=uaa_CORINE_list

    nuts_indices_relevant=output_raster.copy()
    changed_index_positions=np.where(np.invert(np.array(index_list_old)==np.array(index_list_new)))[0]
    for i,index in enumerate(np.array(index_list_old)[changed_index_positions]):
        nuts_indices_relevant[np.where(nuts_indices_relevant==index)]=index_list_new[changed_index_positions[i]]

    nuts_indices_relevant=np.where((np.isin(nuts_indices_relevant,index_list_new))&(corine_data_relevant>0),nuts_indices_relevant,-1)

    uaa_calculated["index"]=index_list_new_wo_duplicates

    uaa_raster_relevant=np.where(nuts_indices_relevant>=0,corine_data_relevant,np.nan)

    uaa_calculated["weight_factor"]=uaa_calculated["UAA_CAPRI"]/uaa_calculated["UAA_CORINE"]
    
    weight_raster=np.ones_like(uaa_raster_relevant)*(-1)
    for i,index in enumerate(np.array(uaa_calculated["index"])):
        weight_factor=uaa_calculated["weight_factor"].iloc[i]
        weight_raster=np.where(nuts_indices_relevant==index,uaa_raster_relevant*weight_factor,weight_raster)

    uaa_calculated_allyears=pd.concat((uaa_calculated_allyears,uaa_calculated))
    weight_raster_allyears[j]=weight_raster
    nuts_indices_relevant_allyears[j]=nuts_indices_relevant
    uaa_raster_relevant_allyears[j]=uaa_raster_relevant
# %%
np.unique(nuts_indices_relevant_allyears[0]).astype(int)
#%%
show(np.where(weight_raster_allyears[20]>0,1,0))
# %%
uaa_calculated_allyears
# %%
#save uaa_calculated,weight_raster,nuts_indices_relevant and uaa_raster_relevant
uaa_calculated_allyears.to_csv(preprocessed_csv_dir+"uaa_calculated_allyears.csv")
# %%
transform_template=rio.open(preprocessed_raster_dir+"nuts_2003.tif")
if weight_raster_allyears.shape[1:]==transform_template.shape:
    transform=transform_template.transform
#%%
with rio.open(preprocessed_raster_dir+"cellweight_raster_allyears.tif", 'w',
            width=int(weight_raster_allyears.shape[2]),height=int(weight_raster_allyears.shape[1]),
            transform=transform,count=weight_raster_allyears.shape[0],dtype=rio.float32,crs="EPSG:3035") as dst:
    dst.write(weight_raster_allyears)
#%%
with rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif", 'w',
            width=int(nuts_indices_relevant_allyears.shape[2]),height=int(nuts_indices_relevant_allyears.shape[1]),
            transform=transform,count=nuts_indices_relevant_allyears.shape[0],dtype=rio.int16,crs="EPSG:3035") as dst:
    dst.write(nuts_indices_relevant_allyears.astype(int))
# %%
with rio.open(preprocessed_raster_dir+"uaa_raster_allyears.tif", 'w',
            width=int(uaa_raster_relevant_allyears.shape[2]),height=int(uaa_raster_relevant_allyears.shape[1]),
            transform=transform,count=uaa_raster_relevant_allyears.shape[0],dtype=rio.float32,crs="EPSG:3035") as dst:
    dst.write(uaa_raster_relevant_allyears)
# %%

# %%
