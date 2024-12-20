#%%
import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
#%%
try:
    main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))
    data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]
except:
    main_path = str(Path(Path(os.path.abspath(__file__)).parents[1]))
    data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]


#%%
raw_dir = data_main_path+"/raw"
os.makedirs(raw_dir, exist_ok=True)

preprocessed_dir = data_main_path+"/preprocessed"
os.makedirs(preprocessed_dir, exist_ok=True)

csv_output_dir=preprocessed_dir+"/csv/"
os.makedirs(csv_output_dir, exist_ok=True)

#the following is the raw CAPRI output:
input_path = raw_dir+"/res_time_series_17.csv"
output_file = csv_output_dir+"preprocessed_CAPREG_step1.csv"

parameter_path=data_main_path+"/delineation_and_parameters/"
crop_delineation_path=parameter_path+"DGPCM_crop_delineation.xlsx"

    
def to_table(element):
    try:
        element = str(element).split("'")
        return [element[1], element[3], element[5], element[7], element[8].split(" ")[1][:-1]]
    except IndexError:
        return None


#%%
crop_delineation=pd.read_excel(crop_delineation_path,sheet_name="CAPRI_crop_names")
chunk_size = 1000000

start_row = 0
with open(input_path, 'r') as file:
    for i, line in enumerate(file):
        if line.startswith("'"):
            start_row = i
            break



chunk=0
#will take 10-15 minutes
while True:
    
    
    print("chunk "+str(chunk))
    try:
        data = pd.read_csv(input_path, skiprows=start_row, nrows=chunk_size, header=None)
    except:
        break
    
    if data.empty:
        break
    
    
    data_array = [to_table(data.iloc[i, 0]) for i in range(len(data))]
    data_array = [x for x in data_array if x is not None]
    
    if not data_array:
        break
    
    
    data_df = pd.DataFrame(data_array, columns=["CAPRI_code","crop","type","year","value"])
    #select relevant types
    data_df=data_df[data_df["type"].isin(["LEVL","YILD","PROD"])]
    #select relevant production (i.e., the relevant crops)
    data_df=data_df[data_df["crop"].isin(np.array(crop_delineation["crop"]))]
    
    if start_row == 0:
        data_df.to_csv(output_file, index=False, mode='w', header=True)
    else:
        data_df.to_csv(output_file, index=False, mode='a', header=False)
    
    
    start_row += chunk_size
   
    chunk+=1
    
    del data
    del data_df
    gc.collect()

print("Processing complete")




# %%
