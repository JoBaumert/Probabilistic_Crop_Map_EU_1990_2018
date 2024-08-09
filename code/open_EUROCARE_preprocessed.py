#%%
import os.path
import pandas as pd
import numpy as np
from pathlib import Path
# %%
main_path=str(Path(Path(os.path.abspath(__file__)).parents[1]))

#%%
data=pd.read_csv(main_path+"/data/preprocessed/filtered_final.csv")
# %%

data=data[data["3"]=="LEVL"]
#%%

data["2"].value_counts()
#%%
NUTS_regions=np.array(data["1"].value_counts().keys()).astype(str)
# %%
NUTS_regions
# %%
German_regions=np.sort(NUTS_regions[np.where(np.char.find(NUTS_regions,"DE")>=0)])
# %%
data[data["1"].isin(German_regions)]
# %%
#only "LEVL" is relevant
list(data["3"].value_counts().keys())
# %%
NUTS_regions
# %%
NUTS2_regs
# %%
NUTS_regions
# %%
