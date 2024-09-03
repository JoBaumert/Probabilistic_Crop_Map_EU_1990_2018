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
filtered_regional_cropdata=pd.read_csv(preprocessed_dir+"csv/filtered_final.csv")
CAPRI_Eurostat_NUTS_mapping=pd.read_csv(raw_dir+"CAPRI_Eurostat_NUTS_mapping.csv",header=None)
#%%
CAPRI_Eurostat_NUTS_mapping

#%%
filtered_regional_cropdata.rename(columns={"1":"CAPRI_code","2":"crop","3":"type","4":"year","5":"value"},inplace=True)
CAPRI_Eurostat_NUTS_mapping.rename(columns={0:"CAPRI_code",1:"NUTS_ID",2:"-"},inplace=True)
filtered_regional_cropdata=filtered_regional_cropdata[filtered_regional_cropdata["type"]=="LEVL"]
filtered_regional_cropdata=filtered_regional_cropdata[filtered_regional_cropdata["year"]>=1990]
#%%
yearwise_nuts_regions=pd.read_csv(result_dir+"csv/yearwise_nuts_regions.csv")
#%%
yearwise_nuts_regions
#%%
all_years_all_regions=pd.DataFrame()
for year in range(1990,2019):
    capri_nuts_ids=filtered_regional_cropdata[filtered_regional_cropdata["year"]==year]["CAPRI_code"].unique().astype(str)
    stripped_nuts_ids=np.char.rstrip(capri_nuts_ids,"0")
    matched_regions=capri_nuts_ids[np.where(np.isin(stripped_nuts_ids,yearwise_nuts_regions["NUTS_ID"]))]
    unmatched_regions=capri_nuts_ids[np.where(np.invert(np.isin(stripped_nuts_ids,yearwise_nuts_regions["NUTS_ID"])))]    
    direct_matches={"CAPRI_code":matched_regions,"NUTS_ID":np.char.rstrip(matched_regions,"0")}    
    #in some cases there is more than just one alternative, that's why we first store the alternatives as a list in a dictionary
    alternative_matches={"CAPRI_code":[],"NUTS_ID":[]}
    for reg in unmatched_regions:
        alternatives=np.array(CAPRI_Eurostat_NUTS_mapping[CAPRI_Eurostat_NUTS_mapping["CAPRI_code"]==reg]["NUTS_ID"])
        #we use year 2003 as it seems that all unmatched regions can be matched to a NUTS code in this year
        occurrence=np.isin(alternatives,yearwise_nuts_regions[yearwise_nuts_regions["2003"]==1]["NUTS_ID"])
        if len(np.where(occurrence)[0])>0:
            for alternative in alternatives[np.where(occurrence)]:
                alternative_matches["CAPRI_code"].append(reg)
                alternative_matches["NUTS_ID"].append(alternative)
            
    all_matches=pd.concat((pd.DataFrame(direct_matches),pd.DataFrame(alternative_matches)))
    all_matches["year"]=np.repeat(year,len(all_matches))
    #verify
    print("un-matched regions: "+str(np.where(np.invert(np.isin(all_matches["CAPRI_code"],capri_nuts_ids)))))
    all_years_all_regions=pd.concat((all_years_all_regions,all_matches))
#%%
all_years_all_regions
#%%
all_years_all_regions["country"]=np.array(all_years_all_regions["NUTS_ID"]).astype("U2")
all_years_all_regions["NUTS_level"]=np.char.str_len(np.array(all_years_all_regions["NUTS_ID"]).astype(str))-2

a=all_years_all_regions[["year","country","NUTS_level"]].groupby(["year","country"]).max().reset_index()
relevant_regions=pd.merge(a,all_years_all_regions,how="left",on=["year","country","NUTS_level"])

relevant_cropdata=pd.merge(relevant_regions,filtered_regional_cropdata,how="left",on=["CAPRI_code","year"])
#%%
relevant_cropdata
#%%
"""IMPORTANT: some regions were split over time, so that 'CAPRI_code' is matched with more than one 'NUTS_ID', e.g., see the following example:"""
relevant_cropdata[(relevant_cropdata["year"]==1990)&(relevant_cropdata["CAPRI_code"]=="IT310000")]
# %%
