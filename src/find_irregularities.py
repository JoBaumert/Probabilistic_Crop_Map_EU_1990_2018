#%%
import numpy as np
import pandas as pd


filtered = pd.read_csv('/home/bajpai/Uni/Josef Hiwi/DGPCM_1990_2020/data/preprocessed/filtered_final.csv', header=None)
filtered[0] = filtered[0].astype(str).str.rstrip('0').replace('', '0')
filtered.to_csv('filtered_new.csv', index=False, header=False)


#%%
filtered = pd.read_csv('/home/bajpai/Uni/Josef Hiwi/DGPCM_1990_2020/data/preprocessed/filtered_new.csv', header=None)
combined_nuts = pd.read_csv('/home/bajpai/Uni/Josef Hiwi/DGPCM_1990_2020/data/preprocessed/combined_nuts_regions.csv')

#%%
filtered.head()
#%%
combined_nuts.head()

#%%
filtered.tail()
#%%
combined_nuts.tail()
# %%
unique_values = {}

# Loop through each unique value in column 0
for value in filtered[0].unique():
    # Find the corresponding unique values from column 3
    corresponding_values = filtered[filtered[0] == value][3].unique()
    # Store them in the dictionary
    unique_values[value] = corresponding_values

# Print the results
for key, values in unique_values.items():
    print(f"Unique values in column 3 corresponding to '{key}' in column 0: {list(values)}")
# %%
#Find unique years for unique nuts ids

nuts_codes = filtered[0].unique()
years = list(range(1984, 2019))
output_df = pd.DataFrame(0, index=nuts_codes, columns=years)

for nuts_code in nuts_codes:
    # Get unique years for this nuts_code
    unique_years = filtered[filtered[0] == nuts_code][3].unique()
    
    # Mark corresponding years with 1
    output_df.loc[nuts_code, unique_years] = 1

output_df.reset_index(inplace=True)
output_df.rename(columns={'index': 'nuts_code'}, inplace=True)

output_df.to_csv('output.csv', index=False)

# %%sanity check
output_df = pd.read_csv('/home/bajpai/Uni/Josef Hiwi/DGPCM_1990_2020/src/output.csv')

output_df.head()
# %%
#find unique years for unique nuts ids in combined csv

nuts_codes = combined_nuts[0].unique()
years = list(range(1984, 2019))
output_df = pd.DataFrame(0, index=nuts_codes, columns=years)

for nuts_code in nuts_codes:
    # Get unique years for this nuts_code
    unique_years = filtered[filtered['NUTS_ID'] == nuts_code]['year'].unique()
    
    # Mark corresponding years with 1
    output_df.loc[nuts_code, unique_years] = 1

output_df.reset_index(inplace=True)


output_df.to_csv('output.csv', index=False)
