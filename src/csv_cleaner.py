#%%
import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
#%%
main_path=str(Path(Path(os.path.abspath(__file__)).parents[1]))
preprocessed_dir = main_path+"/data/preprocessed"
os.makedirs(preprocessed_dir, exist_ok=True)

#%% strip trailing zeros from NUTS column of time series dataset
def strip_zeros():
    filtered = pd.read_csv(preprocessed_dir+'filtered_final.csv', header=None)
    filtered[0] = filtered[0].astype(str).str.rstrip('0').replace('', '0')
    filtered.to_csv('filtered_new.csv', index=False, header=False)

    
def to_table(element):
    try:
        element = str(element).split("'")
        return [element[1], element[3], element[5], element[7], element[8].split(" ")[1][:-1]]
    except IndexError:
        return None

datapath = "res_time_series_17/res_time_series_17.csv"
output_file = "result_table.csv"
#%%
chunk_size = 1000000

start_row = 0
with open(datapath, 'r') as file:
    for i, line in enumerate(file):
        if line.startswith("'"):
            start_row = i
            break


if os.path.exists(output_file):

    with open(output_file, 'rb') as f:
        f.seek(-2, os.SEEK_END)  
        while f.read(1) != b'\n':  
            f.seek(-2, os.SEEK_CUR)  
        last_line = f.readline().decode()

    
    last_processed = last_line.strip().split(',')
    last_processed_str = f"'{last_processed[0]}'.'{last_processed[1]}'.'{last_processed[2]}'.'{last_processed[3]}'{last_processed[4]}"
    
    
    with open(datapath, 'r') as file:
        for i, line in enumerate(file):
            if line.strip() == last_processed_str:
                start_row = i + 1  
                break
else:
    start_row = 0


while True:
    
    data = pd.read_csv(datapath, skiprows=start_row, nrows=chunk_size, header=None)
    
    
    if data.empty:
        break
    
    
    data_array = [to_table(data.iloc[i, 0]) for i in range(len(data))]
    data_array = [x for x in data_array if x is not None]
    
    if not data_array:
        break
    
    
    data_df = pd.DataFrame(data_array, columns=['part1', 'part2', 'part3', 'part4', 'value'])
    
    
    if start_row == 0:
        data_df.to_csv(output_file, index=False, mode='w', header=True)
    else:
        data_df.to_csv(output_file, index=False, mode='a', header=False)
    
    
    start_row += chunk_size
    
    
    del data
    del data_df
    gc.collect()

print("Processing complete!")


