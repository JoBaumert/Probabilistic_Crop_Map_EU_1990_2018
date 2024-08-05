#%%
import pandas as pd
import numpy as np
# %%
data=pd.read_csv("/home/baumert/fdiexchange/baumert/DGPCM_19902020/filtered_final.csv")
# %%
NUTS_regions=np.array(data["1"].value_counts().keys()).astype(str)
# %%
a=np.array(np.char.split(NUTS_regions[np.where(np.char.find(NUTS_regions,"0")==2)],"0"))
# %%
a
# %%
German_regions=np.sort(NUTS_regions[np.where(np.char.find(NUTS_regions,"DE")>=0)])
# %%
data[data["1"].isin(German_regions)]
# %%
