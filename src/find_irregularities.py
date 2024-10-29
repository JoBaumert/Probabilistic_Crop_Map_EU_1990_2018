#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os 
from tqdm import tqdm

#main_path=str(Path(Path(os.path.abspath(__file__)).parents[1]))
#result_dir = main_path+"/data/results/"
#os.makedirs(result_dir, exist_ok=True)
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

# %% find unique years for unique nuts ids in combined csv
def create_yearwise_table_filtered_timeseries(regional_cropdata_df):
    nuts_codes = regional_cropdata_df[0].unique()
    years = list(range(1984, 2019))
    result_df = pd.DataFrame(0, index=regional_cropdata_df[0].unique(), columns=years)
   
    for nuts_code in nuts_codes:
        unique_years = regional_cropdata_df[regional_cropdata_df[0] == nuts_code][3].unique()
        result_df.loc[nuts_code, unique_years] = 1

    result_df.reset_index(inplace=True)
    result_df.rename(columns={'index': 'nuts_code'}, inplace=True)

    result_df.to_csv(preprocessed_csv_dir+'yearwise_regional_cropdata_overview.csv', index=False)


# %% find unique years for unique nuts ids in combined csv
def create_yearwise_table_nuts_region_dictionary(nuts_regions_dictionary_df):
    nuts_codes = nuts_regions_dictionary_df['NUTS_ID'].unique()
    years = [2003, 2006, 2010, 2013, 2016, 2021]
    result_df = pd.DataFrame(0, index=nuts_regions_dictionary_df['NUTS_ID'].unique(), columns=years)

    for nuts_id in tqdm(nuts_regions_dictionary_df['NUTS_ID'].unique()):
        for year in years:
            if ((nuts_regions_dictionary_df['NUTS_ID'] == nuts_id) & (nuts_regions_dictionary_df['year'] == year)).any():
                result_df.at[nuts_id, year] = 1

    result_df.reset_index(inplace=True)
    result_df.rename(columns={'index': 'NUTS_ID'}, inplace=True)

    result_df.to_csv(preprocessed_csv_dir+'yearwise_nuts_regions_overview.csv', index=False)
#%%

if __name__ == '__main__':
    # load preprocessed data
    regional_cropdata = pd.read_csv(preprocessed_csv_dir+'preprocessed_CAPREG_step1.csv', header=None)
    nuts_regions_dictionary = pd.read_csv(preprocessed_csv_dir+'nuts_regions_dictionary.csv')
    #%%
    regional_cropdata
    #%% create yearwise csv files
    create_yearwise_table_filtered_timeseries(regional_cropdata)
    #%%
    create_yearwise_table_nuts_region_dictionary(nuts_regions_dictionary)


# %%
